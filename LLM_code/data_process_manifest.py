import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


LABEL_TEXT = {
    "iemocap": "happy, sad, neutral, angry, excited, frustrated",
    "meld": "neutral, surprise, fear, sadness, joy, disgust, anger",
}


def label_guidance(dataset):
    if dataset == "iemocap":
        return (
            "Important label distinctions:\n"
            "- happy: positive and pleased, but not necessarily energetic.\n"
            "- excited: positive high-arousal emotion, energetic or enthusiastic.\n"
            "- angry: direct anger, irritation, or hostility.\n"
            "- frustrated: blocked, dissatisfied, helpless, or annoyed due to difficulty.\n"
            "- neutral: no clear emotional intensity.\n"
            "- sad: sorrow, disappointment, or low mood."
        )
    if dataset == "meld":
        return (
            "Important label distinctions:\n"
            "- joy: positive happiness or delight.\n"
            "- surprise: sudden unexpected reaction, not necessarily positive.\n"
            "- anger: anger, irritation, or hostility.\n"
            "- disgust: aversion, dislike, or contempt.\n"
            "- fear: worry, anxiety, or being scared.\n"
            "- sadness: sorrow, disappointment, or low mood.\n"
            "- neutral: no clear emotional intensity."
        )
    return ""


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def speaker_name(dataset, row):
    if row.get("speaker"):
        return row["speaker"]
    if dataset == "iemocap":
        speaker = row.get("speaker", "")
        if speaker == "M":
            return "Male Speaker"
        if speaker == "F":
            return "Female Speaker"
    return "Speaker"


def row_sort_key(row):
    if "sequence_index" in row and row["sequence_index"] != "":
        return int(row["sequence_index"])
    if "turn_id" in row and row["turn_id"] != "":
        return int(row["turn_id"])
    return row.get("utterance_id", "")


def group_dialogs(rows):
    dialogs = defaultdict(list)
    for row in rows:
        dialog_id = row.get("dialog_id") or row.get("local_dialog_id") or row["utterance_id"]
        dialogs[str(dialog_id)].append(row)
    for dialog_rows in dialogs.values():
        dialog_rows.sort(key=row_sort_key)
    return dialogs


def build_qwen_chat_prompt(dataset, context_lines, target_utterance):
    system_msg = (
        "You are an expert in emotion recognition in conversation. "
        "Classify the emotion of the target utterance only. "
        "Use the dialogue context and the target speaker. "
        "Audio and visual features are provided separately as learned prefix tokens. "
        "Do not classify the whole dialogue. Choose exactly one label from the given label set and output no other words.\n\n"
        f"{label_guidance(dataset)}"
    )
    user_msg = "\n\n".join(
        [
            f"Available emotion labels:\n{LABEL_TEXT[dataset]}",
            "Dialogue context:\n" + "\n".join(context_lines),
            f"Target utterance:\n{target_utterance}",
            "Question:\nWhat is the emotion label of the target utterance? "
            "Answer with exactly one label from the available labels.",
        ]
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_legacy_prompt(dataset, context_lines, target_utterance):
    context = " ".join(context_lines)
    return (
        "Now you are expert of sentiment and emotional analysis."
        "The following conversation noted between '### ###' involves several speakers. "
        f"### {context} ### "
        f"Please select the emotional label of <{target_utterance}> from <{LABEL_TEXT[dataset]}> "
        "based on the context. Respond with just one label:"
    )


def process_split(dataset, rows, window, prompt_style):
    examples = []
    dialogs = group_dialogs(rows)
    for dialog_rows in dialogs.values():
        for idx, row in enumerate(dialog_rows):
            start = max(idx - window, 0)
            context_lines = []
            for ctx_row in dialog_rows[start : idx + 1]:
                name = speaker_name(dataset, ctx_row)
                text = ctx_row.get("text", "")
                context_lines.append(f'{name}: "{text}"')
            target_utterance = context_lines[-1]
            if prompt_style == "qwen_chat":
                prompt = build_qwen_chat_prompt(dataset, context_lines, target_utterance)
            else:
                prompt = build_legacy_prompt(dataset, context_lines, target_utterance)
            examples.append(
                {
                    "path": row["utterance_id"],
                    "input": prompt,
                    "target": row["label"],
                }
            )
    return examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_dataset(dataset, manifest_dir, window, prompt_style, output_dir=None):
    manifest_dir = Path(manifest_dir)
    if output_dir is None:
        output_dir = f"../PROCESSED_DATASET/{dataset}/manifest/window_{window}_{prompt_style}"
    output_dir = Path(output_dir)
    split_files = {
        "train": manifest_dir / f"{dataset}_multimodal_train.jsonl",
        "test": manifest_dir / f"{dataset}_multimodal_test.jsonl",
        "valid": manifest_dir / f"{dataset}_multimodal_valid.jsonl",
    }
    report = {}
    for split, path in split_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Manifest split not found: {path}")
        rows = read_jsonl(path)
        examples = process_split(dataset, rows, window, prompt_style)
        write_jsonl(output_dir / f"{split}.json", examples)
        report[split] = len(examples)
    with (output_dir / "manifest_process_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Build Qwen training prompts directly from multimodal manifests.")
    parser.add_argument("--dataset", required=True, choices=["iemocap", "meld"])
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--historical_window", type=int, default=12)
    parser.add_argument("--prompt_style", choices=["legacy", "qwen_chat"], default="qwen_chat")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    output_dir = process_dataset(
        dataset=args.dataset,
        manifest_dir=args.manifest_dir,
        window=args.historical_window,
        prompt_style=args.prompt_style,
        output_dir=args.output_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
