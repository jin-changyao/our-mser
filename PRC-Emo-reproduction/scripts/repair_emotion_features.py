#!/usr/bin/env python3
"""Regenerate only invalid emotion-feature records.

The full emotion extractor intentionally stops when any record is invalid.
This utility locates those records in the saved feature JSON/raw JSONL files,
regenerates only them, and updates both files after a valid answer is found.
"""

import copy
import json
import os
import re
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DATASET = os.environ.get("PRC_EMO_DATASET", "meld").lower()
DATA_FOLDER = Path(os.environ.get("PRC_EMO_DATA_FOLDER", "./data")).expanduser()
MODEL_PATH = os.environ.get(
    "PRC_EMO_MODEL_PATH", "/home/pc/jcy/models/Qwen2.5-7B-Instruct"
)
MODEL_ID = Path(MODEL_PATH).name
PROMPT_TYPE = "ImplicitEmotion_V3"
SPLITS = [
    item.strip()
    for item in os.environ.get("PRC_EMO_SPLITS", "valid,test,train").split(",")
    if item.strip()
]
MAX_NEW_TOKENS = int(os.environ.get("PRC_EMO_REPAIR_MAX_NEW_TOKENS", "256"))
MAX_ATTEMPTS = max(1, int(os.environ.get("PRC_EMO_REPAIR_ATTEMPTS", "3")))
INPUT_MAX_LENGTH = int(os.environ.get("PRC_EMO_INPUT_MAX_LENGTH", "2048"))
DO_SAMPLE = os.environ.get("PRC_EMO_DO_SAMPLE", "0").lower() in {
    "1",
    "true",
    "yes",
}


def feature_path(split):
    return DATA_FOLDER / f"{DATASET}.{split}_{PROMPT_TYPE}_{MODEL_ID}.json"


def raw_path(split):
    return DATA_FOLDER / f"{DATASET}.{split}_{PROMPT_TYPE}_{MODEL_ID}.raw.jsonl"


def source_path(split):
    return DATA_FOLDER / f"{DATASET}.{split}.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_source(path):
    raw = load_json(path)
    if isinstance(raw, dict):
        result = []
        for conversation_id, value in raw.items():
            item = copy.deepcopy(value)
            item["s_id"] = conversation_id
            result.append(item)
        return result
    return raw


def iemocap_speaker_name(conversation_id, gender):
    names = {
        "Ses01": {"F": "MARY", "M": "JAMES"},
        "Ses02": {"F": "PATRICIA", "M": "JOHN"},
        "Ses03": {"F": "JENNIFER", "M": "ROBERT"},
        "Ses04": {"F": "LINDA", "M": "MICHAEL"},
        "Ses05": {"F": "ELIZABETH", "M": "WILLIAM"},
    }
    return names[conversation_id[:5]][gender]


def speaker_names(sample):
    if DATASET == "meld":
        return list(sample["speakers"])
    return [iemocap_speaker_name(sample["s_id"], gender) for gender in sample["genders"]]


def build_prompt(sample, utterance_index, window=5):
    names = speaker_names(sample)
    tagged = [
        f'{name}: "{text}"'
        for name, text in zip(names, sample["sentences"])
    ]
    start = max(0, utterance_index - window + 1)
    context = " ".join(tagged[start : utterance_index + 1])
    current_speaker = names[utterance_index]
    current_utterance = sample["sentences"][utterance_index]
    return (
        "You are an expert in analyzing human surface and implicit emotions through conversation context.\n"
        "### Task ###\n"
        "1. Analyze **explicit emotion** (outward expression based on their words and tone.) and **implicit emotion** (true inner feeling, even if not directly expressed.)\n"
        "2. Use **natural language descriptions** (no emotion labels like 'sappiness')\n"
        "3. Use concise but informative natural-language descriptions; no fixed word count is required.\n"
        "4. You MUST take into account the entire past conversation context, including what the current speaker and others have said earlier.\n"
        "### Conversation Context ###\n"
        f"{context}\n\n"
        "### Focus Utterance ###\n"
        f"{current_speaker}: {current_utterance}\n\n"
        "### Required JSON Format ###\n"
        '{\n"ExplicitEmotion": "<description>",\n"ImplicitEmotion": "<description>"\n}\n'
        "### Critical Rules ###\n"
        "1. Output **ONLY** the JSON object\n"
        "2. No explanations/thinking processes\n"
        "3. No markdown/code formatting\n"
        "4. Keys must be exactly as shown\n"
        "5. Both values are mandatory and must be non-empty natural-language descriptions.\n"
        "6. Stop immediately after the closing brace; do not generate Human/Assistant turns or another JSON object.\n\n"
        "Return exactly one object with both fields filled in:\n"
        '{\n"ExplicitEmotion": "<description>",\n'
        '"ImplicitEmotion": "<description>"\n}'
    )


def parse_output(text):
    text = str(text or "").strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        normalized = {
            str(key).lower().replace("_", "").replace(" ", ""): value
            for key, value in candidate.items()
        }
        surface = str(normalized.get("explicitemotion", "")).strip()
        implicit = str(normalized.get("implicitemotion", "")).strip()
        if surface and implicit:
            return surface, implicit, ""
    return "No prediction", "No prediction", "unable_to_parse_or_empty_field"


def is_invalid(prediction):
    if not isinstance(prediction, dict):
        return True
    invalid = {"", "No prediction", "unknown", "null", "None"}
    return (
        str(prediction.get("surface_emotion", "")).strip() in invalid
        or str(prediction.get("implicit_emotion", "")).strip() in invalid
        or bool(prediction.get("parse_error"))
    )


def backup_once(path):
    backup = Path(str(path) + ".bak_before_repair")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def load_raw_lines(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


print(
    f"Repair configuration: dataset={DATASET}, splits={SPLITS}, "
    f"attempts={MAX_ATTEMPTS}, do_sample={DO_SAMPLE}"
)
print("Repair validity: both emotion fields must be non-empty and parseable; description length is not enforced.")

targets = []
feature_data_by_split = {}
raw_records_by_split = {}
source_by_split = {}

for split in SPLITS:
    fpath = feature_path(split)
    rpath = raw_path(split)
    spath = source_path(split)
    if not fpath.exists():
        print(f"Skipping missing feature file: {fpath}")
        continue
    feature_data = load_json(fpath)
    raw_records = load_raw_lines(rpath)
    source_items = {item["s_id"]: item for item in load_source(spath)}
    feature_data_by_split[split] = feature_data
    raw_records_by_split[split] = raw_records
    source_by_split[split] = source_items

    for conversation_id, conversation in feature_data.items():
        for utterance_index, prediction in enumerate(
            conversation.get("emotion_predictions", [])
        ):
            if is_invalid(prediction):
                targets.append((split, conversation_id, utterance_index))

print(f"Located invalid records: {len(targets)}")
for target in targets:
    print("-", target)

if not targets:
    print("No invalid emotion records found.")
    raise SystemExit(0)

print("Loading repair model ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

repair_records = []
failed = []
for split, conversation_id, utterance_index in targets:
    sample = source_by_split[split].get(conversation_id)
    if sample is None or utterance_index >= len(sample.get("sentences", [])):
        failed.append((split, conversation_id, utterance_index, "source utterance not found"))
        continue

    prompt = build_prompt(sample, utterance_index)
    best_raw = ""
    best_surface = "No prediction"
    best_implicit = "No prediction"
    best_error = "unable_to_parse_or_empty_field"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=INPUT_MAX_LENGTH,
        ).to(device)
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": 1.2,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "num_return_sequences": 1,
            "do_sample": DO_SAMPLE,
        }
        if DO_SAMPLE:
            generation_kwargs.update(temperature=0.3, top_p=0.9)
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        raw_output = tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        surface, implicit, parse_error = parse_output(raw_output)
        best_raw = raw_output
        best_surface = surface
        best_implicit = implicit
        best_error = parse_error
        if not parse_error:
            break
        print(
            f"Invalid repair attempt {attempt}/{MAX_ATTEMPTS}: "
            f"{split}/{conversation_id}/{utterance_index}"
        )

    result = {
        "surface_emotion": best_surface,
        "implicit_emotion": best_implicit,
        "prompt": prompt,
        "raw_model_output": best_raw,
        "parse_error": best_error,
        "prompt_tokens_before_truncation": len(
            tokenizer(prompt, add_special_tokens=True)["input_ids"]
        ),
    }
    repair_records.append(
        {
            "dataset": DATASET,
            "split": split,
            "conversation_id": conversation_id,
            "utterance_index": utterance_index,
            "attempts": attempt,
            **result,
        }
    )

    if best_error:
        failed.append((split, conversation_id, utterance_index, best_error))
        continue

    feature_data_by_split[split][conversation_id]["emotion_predictions"][
        utterance_index
    ] = result
    for record in raw_records_by_split[split]:
        if (
            record.get("conversation_id") == conversation_id
            and record.get("utterance_index") == utterance_index
        ):
            record.update(result)
            break

for split, feature_data in feature_data_by_split.items():
    fpath = feature_path(split)
    rpath = raw_path(split)
    backup_once(fpath)
    backup_once(rpath)
    with fpath.open("w", encoding="utf-8") as handle:
        json.dump(feature_data, handle, indent=2, ensure_ascii=False)
    if rpath.exists():
        with rpath.open("w", encoding="utf-8") as handle:
            for record in raw_records_by_split[split]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

repair_log = DATA_FOLDER / (
    f"{DATASET}_{PROMPT_TYPE}_{MODEL_ID}_repair_raw.jsonl"
)
with repair_log.open("w", encoding="utf-8") as handle:
    for record in repair_records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Repair records saved to: {repair_log}")
print(f"Repaired successfully: {len(targets) - len(failed)} / {len(targets)}")
if failed:
    print("Still failed:")
    for item in failed:
        print("-", item)
    raise SystemExit(1)
print("All invalid emotion records were repaired successfully.")
