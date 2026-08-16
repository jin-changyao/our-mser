"""Generate explicit/implicit emotion descriptions for PRC-Emo.

The original script treated an existing conversation-level JSON file as
complete even when every prediction was ``No prediction``. This version
keeps the original prompt idea, but validates cached data, preserves raw
model answers, and fails loudly when an answer cannot be parsed.
"""

import copy
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DATASET = os.environ.get("PRC_EMO_DATASET", "iemocap").lower()
DATA_FOLDER = Path(os.environ.get("PRC_EMO_DATA_FOLDER", "./data/")).expanduser()
MODEL_PATH = os.environ.get(
    "PRC_EMO_MODEL_PATH", "/home/pc/jcy/models/Qwen2.5-7B-Instruct"
)
PROMPT_TYPE = "ImplicitEmotion_V3"
MODEL_ID = Path(MODEL_PATH).name

FORCE_REGEN = os.environ.get("PRC_EMO_FORCE_REGEN", "0").lower() in {
    "1",
    "true",
    "yes",
}
MAX_UTTERANCES = int(os.environ.get("PRC_EMO_MAX_UTTERANCES", "0"))
OUTPUT_SUFFIX = os.environ.get("PRC_EMO_OUTPUT_SUFFIX", "").strip()
SPLITS = [
    item.strip()
    for item in os.environ.get("PRC_EMO_SPLITS", "valid,test,train").split(",")
    if item.strip()
]
INPUT_MAX_LENGTH = int(os.environ.get("PRC_EMO_INPUT_MAX_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("PRC_EMO_MAX_NEW_TOKENS", "800"))
BATCH_SIZE = max(1, int(os.environ.get("PRC_EMO_BATCH_SIZE", "2")))

if DATASET not in {"iemocap", "meld"}:
    raise ValueError(f"This reproduction script currently supports iemocap/meld, got {DATASET}")
if set(SPLITS) - {"valid", "test", "train"}:
    raise ValueError(f"Invalid PRC_EMO_SPLITS={SPLITS}")
if MAX_UTTERANCES < 0:
    raise ValueError("PRC_EMO_MAX_UTTERANCES must be >= 0")


def output_path(split):
    return DATA_FOLDER / f"{DATASET}.{split}_{PROMPT_TYPE}_{MODEL_ID}{OUTPUT_SUFFIX}.json"


def raw_output_path(split):
    path = output_path(split)
    return path.with_name(path.stem + ".raw.jsonl")


def load_raw_data(path):
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        result = []
        for conversation_id, value in raw.items():
            item = copy.deepcopy(value)
            item["s_id"] = conversation_id
            result.append(item)
        return result
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unsupported raw data format: {path}")


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
    return [
        iemocap_speaker_name(sample["s_id"], gender)
        for gender in sample["genders"]
    ]


def build_prompt(sample, utterance_index, window=5):
    names = speaker_names(sample)
    tagged = [
        f'{name}: "{text}"'
        for name, text in zip(names, sample["sentences"])
    ]
    start = max(0, utterance_index - window + 1)
    context = " ".join(tagged[start : utterance_index + 1])
    current_speaker = names[utterance_index]
    current_utterance = tagged[utterance_index]
    return (
        "You are an expert in analyzing human surface and implicit emotions through conversation context.\n"
        "### Task ###\n"
        "1. Analyze **explicit emotion** (outward expression based on their words and tone.) and **implicit emotion** (true inner feeling, even if not directly expressed.)\n"
        "2. Use **natural language descriptions** (no emotion labels like 'sappiness')\n"
        "3. Use **at least 20 words**, but no more than 50 words each\n"
        "4. You MUST take into account the entire past conversation context, including what the current speaker and others have said earlier."
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
        "4. Keys must be exactly as shown\n\n"
        "Example Valid Response:\n"
        '{\n"ExplicitEmotion": "The speaker\'s cheerful tone and frequent use of positive adjectives suggest outwardly optimistic engagement...",\n'
        '"ImplicitEmotion": "Underneath the enthusiastic delivery, a slight hesitation in phrasing hints at unspoken reservations..."\n}'
    )


def parse_emotion_output(text):
    """Return (surface, implicit, parse_error), never silently accept a prompt."""
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
        surface = normalized.get("explicitemotion", normalized.get("surfaceemotion", ""))
        implicit = normalized.get("implicitemotion", "")
        if str(surface).strip() and str(implicit).strip():
            return str(surface).strip(), str(implicit).strip(), ""

    surface_match = re.search(
        r"(?:ExplicitEmotion|SurfaceEmotion)\s*:\s*(.+?)(?=\n|$)", text, re.I
    )
    implicit_match = re.search(
        r"ImplicitEmotion\s*:\s*(.+?)(?=\n|$)", text, re.I
    )
    if surface_match and implicit_match:
        return (
            surface_match.group(1).strip(),
            implicit_match.group(1).strip(),
            "non_json_format",
        )
    return "No prediction", "No prediction", "unable_to_parse_model_output"


def valid_prediction(prediction):
    if not isinstance(prediction, dict):
        return False
    invalid = {"", "No prediction", "unknown", "null", "None"}
    surface = str(prediction.get("surface_emotion", "")).strip()
    implicit = str(prediction.get("implicit_emotion", "")).strip()
    return surface not in invalid and implicit not in invalid


def valid_conversation(record):
    utterances = record.get("utterances", []) if isinstance(record, dict) else []
    predictions = record.get("emotion_predictions", []) if isinstance(record, dict) else []
    return (
        isinstance(utterances, list)
        and isinstance(predictions, list)
        and len(utterances) == len(predictions)
        and len(predictions) > 0
        and all(valid_prediction(item) for item in predictions)
    )


def truncate_samples(samples, limit):
    if limit <= 0:
        return samples
    selected = []
    remaining = limit
    for original in samples:
        if remaining <= 0:
            break
        sample = copy.deepcopy(original)
        count = min(len(sample.get("sentences", [])), remaining)
        if count == 0:
            continue
        for field in ("sentences", "labels", "genders", "speakers"):
            if isinstance(sample.get(field), list):
                sample[field] = sample[field][:count]
        selected.append(sample)
        remaining -= count
    return selected


def write_jsonl(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


print(
    f"Configuration: dataset={DATASET}, splits={SPLITS}, "
    f"max_utterances={MAX_UTTERANCES or 'all'}, force_regen={FORCE_REGEN}, "
    f"suffix={OUTPUT_SUFFIX or '<none>'}, model={MODEL_PATH}"
)

print("Loading model ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
    device_map="auto",
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "left"
print("Model loaded successfully from local path.")

raw_data = []
cached_by_split = {}
for split in SPLITS:
    source_path = DATA_FOLDER / f"{DATASET}.{split}.json"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    original = load_raw_data(source_path)
    processed_path = output_path(split)
    cached = {}
    if processed_path.exists() and not FORCE_REGEN:
        with processed_path.open("r", encoding="utf-8") as handle:
            candidate = json.load(handle)
        cached = {
            conv_id: record
            for conv_id, record in candidate.items()
            if valid_conversation(record)
        }
        print(
            f"{split}: cache={len(candidate)}, valid_cache={len(cached)}, "
            f"reprocess={len(candidate) - len(cached)}"
        )
    elif processed_path.exists():
        print(f"{split}: force regeneration; ignoring {processed_path}")

    remaining = [item for item in original if item["s_id"] not in cached]
    for item in remaining:
        item["type_data"] = split
    cached_by_split[split] = cached
    raw_data.extend(remaining)
    print(f"{split}: raw_conversations={len(original)}, to_generate={len(remaining)}")

raw_data = truncate_samples(raw_data, MAX_UTTERANCES)
if MAX_UTTERANCES:
    print(f"Smoke-test utterances selected: {sum(len(x['sentences']) for x in raw_data)}")
if not raw_data:
    print("No invalid or missing cached conversations require generation; exiting successfully.")
    raise SystemExit(0)

entries = []
for sample in raw_data:
    for utterance_index in range(len(sample["sentences"])):
        prompt = build_prompt(sample, utterance_index)
        entries.append(
            {
                "type_data": sample["type_data"],
                "conv_id": sample["s_id"],
                "utter_idx": utterance_index,
                "prompt": prompt,
                "prompt_tokens": len(TOKENIZER(prompt, add_special_tokens=True)["input_ids"]),
            }
        )
print(f"Prepared {len(entries)} prompts.")

raw_handles = {}
for split in SPLITS:
    raw_path = raw_output_path(split)
    raw_handles[split] = raw_path.open("w", encoding="utf-8")
    print(f"Raw answers: {raw_path}")

generated = {}
invalid_count = 0
first_example = True

try:
    offset = 0
    while offset < len(entries):
        current_batch_size = min(BATCH_SIZE, len(entries) - offset)
        batch_entries = entries[offset : offset + current_batch_size]
        while True:
            try:
                prompts = [item["prompt"] for item in batch_entries]
                inputs = TOKENIZER(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=INPUT_MAX_LENGTH,
                ).to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.3,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.2,
                        eos_token_id=TOKENIZER.eos_token_id,
                        pad_token_id=TOKENIZER.eos_token_id,
                        num_return_sequences=1,
                    )
                input_width = inputs["input_ids"].shape[1]
                raw_outputs = TOKENIZER.batch_decode(
                    output_ids[:, input_width:], skip_special_tokens=True
                )
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                batch_entries = entries[offset : offset + current_batch_size]
                print(f"CUDA OOM; retrying batch size {current_batch_size}")
                torch.cuda.empty_cache()

        for item, raw_output in zip(batch_entries, raw_outputs):
            surface, implicit, parse_error = parse_emotion_output(raw_output)
            if surface == "No prediction" or implicit == "No prediction":
                invalid_count += 1
            result = {
                "surface_emotion": surface,
                "implicit_emotion": implicit,
                "prompt": item["prompt"],
                "raw_model_output": raw_output,
                "parse_error": parse_error,
                "prompt_tokens_before_truncation": item["prompt_tokens"],
            }
            key = (item["type_data"], item["conv_id"], item["utter_idx"])
            generated[key] = result
            write_jsonl(
                raw_handles[item["type_data"]],
                {
                    "dataset": DATASET,
                    "split": item["type_data"],
                    "conversation_id": item["conv_id"],
                    "utterance_index": item["utter_idx"],
                    **result,
                },
            )
            if first_example:
                print("--- First generated example ---")
                print("Raw model output:", raw_output)
                print("Parsed:", json.dumps(result, ensure_ascii=False))
                print("--------------------------------")
                first_example = False
        offset += len(batch_entries)
finally:
    for handle in raw_handles.values():
        handle.close()

print(f"Generated={len(generated)}, invalid={invalid_count}")

for split in SPLITS:
    final_data = dict(cached_by_split[split])
    samples = [item for item in raw_data if item["type_data"] == split]
    grouped = {}
    for item in samples:
        grouped.setdefault(item["s_id"], item)

    for conversation_id, sample in grouped.items():
        names = speaker_names(sample)
        predictions = []
        for utterance_index in range(len(sample["sentences"])):
            key = (split, conversation_id, utterance_index)
            if key not in generated:
                raise RuntimeError(f"Missing result for {key}")
            predictions.append(generated[key])
        final_data[conversation_id] = {
            "utterances": [
                f"{name}: {text}"
                for name, text in zip(names, sample["sentences"])
            ],
            "emotion_predictions": predictions,
        }

    processed_path = output_path(split)
    with processed_path.open("w", encoding="utf-8") as handle:
        json.dump(final_data, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(final_data)} conversations for {split}: {processed_path}")

if invalid_count:
    raise RuntimeError(
        f"Generated {invalid_count} invalid emotion predictions. "
        "Inspect the .raw.jsonl files; refusing to report success."
    )

print("Emotion feature generation completed with valid predictions.")
