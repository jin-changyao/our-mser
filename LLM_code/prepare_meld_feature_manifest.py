import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


STANDARD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

# The feature package's label.npz uses this numeric order, which is different
# from the standard MELD order used by the training code.
FEATURE_LABELS = {
    0: "anger",
    1: "joy",
    2: "sadness",
    3: "neutral",
    4: "disgust",
    5: "fear",
    6: "surprise",
}

SPLIT_TO_CORPUS = {
    "train": "train_corpus",
    "valid": "val_corpus",
    "test": "test_corpus",
}

GLOBAL_DIALOG_OFFSETS = {
    "train": 0,
    "valid": 1039,
    "test": 1153,
}

SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}


def clean_text(text):
    text = str(text)
    replacements = {
        "\x92": "'",
        "\u805e": "'",
        "\u806d": "",
        "\u806f": "",
        "\u8070": "",
        "\u8074": " ",
        "\u805f": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def read_transcriptions(path):
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "english"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing transcription columns: {sorted(missing)}")
        for row in reader:
            utterance_id = row["name"].strip()
            if not utterance_id:
                continue
            rows[utterance_id] = {
                "text": row.get("english", "").strip(),
                "text_zh": row.get("chinese", "").strip(),
            }
    return rows


def parse_utterance_id(utterance_id):
    match = re.match(r"^(train|val|test)_dia(\d+)_utt(\d+)$", utterance_id)
    if not match:
        raise ValueError(f"Unexpected MELD utterance id: {utterance_id}")
    raw_split, local_dialog_id, turn_id = match.groups()
    split = "valid" if raw_split == "val" else raw_split
    local_dialog_id = int(local_dialog_id)
    return {
        "split": split,
        "raw_split": raw_split,
        "local_dialog_id": local_dialog_id,
        "dialog_id": GLOBAL_DIALOG_OFFSETS[split] + local_dialog_id,
        "turn_id": int(turn_id),
    }


def sort_key_for_utterance_id(utterance_id):
    parsed = parse_utterance_id(utterance_id)
    return parsed["local_dialog_id"], parsed["turn_id"]


def feature_path(feature_root, feature_dir, utterance_id):
    path = feature_root / feature_dir / f"{utterance_id}.npy"
    return str(path) if path.is_file() else ""


def load_old_meld_pickle(path):
    if path is None or not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def build_old_sequences(old_data):
    if old_data is None:
        return {}
    split_ids = {
        "train": old_data[4],
        "test": old_data[5],
        "valid": old_data[6],
    }
    sequences = {}
    for split, dialog_ids in split_ids.items():
        rows = []
        for dialog_id in dialog_ids:
            for turn_id, text in enumerate(old_data[3][dialog_id]):
                label_id = int(old_data[1][dialog_id][turn_id])
                rows.append(
                    {
                        "dialog_id": dialog_id,
                        "turn_id": turn_id,
                        "text": str(text),
                        "label": STANDARD_LABELS[label_id],
                    }
                )
        sequences[split] = rows
    return sequences


def build_manifest(feature_root, label_path, transcription_path, feature_dirs, old_meld_pkl=None):
    payload = np.load(label_path, allow_pickle=True)
    transcriptions = read_transcriptions(transcription_path)
    old_sequences = build_old_sequences(load_old_meld_pickle(old_meld_pkl))

    rows = []
    missing_transcription = []
    missing_features = {name: [] for name in feature_dirs}
    old_text_mismatches = []
    old_label_mismatches = []

    for split in ["train", "valid", "test"]:
        corpus = payload[SPLIT_TO_CORPUS[split]].item()
        sorted_items = sorted(corpus.items(), key=lambda item: sort_key_for_utterance_id(item[0]))
        for sequence_index, (utterance_id, label_info) in enumerate(sorted_items):
            parsed = parse_utterance_id(utterance_id)
            text_info = transcriptions.get(utterance_id)
            if text_info is None:
                missing_transcription.append(utterance_id)
                text = ""
                text_zh = ""
            else:
                text = text_info["text"]
                text_zh = text_info["text_zh"]

            feature_label_id = int(label_info["emo"])
            label = FEATURE_LABELS.get(feature_label_id, "")
            label_id = STANDARD_LABELS.index(label) if label in STANDARD_LABELS else -1

            old_row = None
            if split in old_sequences and sequence_index < len(old_sequences[split]):
                old_row = old_sequences[split][sequence_index]
            old_text = old_row["text"] if old_row else None
            old_label = old_row["label"] if old_row else None
            old_text_match = old_text is not None and clean_text(old_text) == clean_text(text)
            old_label_match = old_label is not None and old_label == label

            if old_text is not None and not old_text_match:
                old_text_mismatches.append(
                    {
                        "utterance_id": utterance_id,
                        "sequence_index": sequence_index,
                        "old_dialog_id": old_row["dialog_id"],
                        "old_turn_id": old_row["turn_id"],
                        "old_text": old_text,
                        "new_text": text,
                    }
                )
            if old_label is not None and not old_label_match:
                old_label_mismatches.append(
                    {
                        "utterance_id": utterance_id,
                        "sequence_index": sequence_index,
                        "old_dialog_id": old_row["dialog_id"],
                        "old_turn_id": old_row["turn_id"],
                        "old_label": old_label,
                        "feature_label": label,
                    }
                )

            row = {
                "split": split,
                "utterance_id": utterance_id,
                "dialog_id": parsed["dialog_id"],
                "local_dialog_id": parsed["local_dialog_id"],
                "turn_id": parsed["turn_id"],
                "sequence_index": sequence_index,
                "text": text,
                "text_zh": text_zh,
                "label": label,
                "label_id": label_id,
                "feature_label_id": feature_label_id,
                "standard_label_order": ",".join(STANDARD_LABELS),
                "val": label_info.get("val", ""),
                "old_dialog_id": old_row["dialog_id"] if old_row else "",
                "old_turn_id": old_row["turn_id"] if old_row else "",
                "old_speechcue_label": old_label if old_label is not None else "",
                "old_text_match": old_text_match if old_text is not None else "",
                "old_label_match": old_label_match if old_label is not None else "",
            }
            for name in feature_dirs:
                path = feature_path(feature_root, name, utterance_id)
                row[f"feature_{name}"] = path
                if not path:
                    missing_features[name].append(utterance_id)
            rows.append(row)

    rows.sort(key=lambda item: (SPLIT_ORDER[item["split"]], item["local_dialog_id"], item["turn_id"]))
    report = {
        "num_manifest_rows": len(rows),
        "num_transcriptions": len(transcriptions),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "missing_transcription": len(missing_transcription),
        "missing_transcription_examples": missing_transcription[:10],
        "missing_features": {
            name: {"count": len(ids), "examples": ids[:10]}
            for name, ids in missing_features.items()
        },
        "old_speechcue_text_mismatches": len(old_text_mismatches),
        "old_speechcue_text_mismatch_examples": old_text_mismatches[:5],
        "old_speechcue_label_mismatches": len(old_label_mismatches),
        "old_speechcue_label_mismatch_examples": old_label_mismatches[:10],
    }
    return rows, report


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_split_jsonl(out_dir, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in ["train", "valid", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        out_path = out_dir / f"meld_multimodal_{split}.jsonl"
        write_jsonl(out_path, split_rows)
        written[split] = str(out_path)
    return written


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build a MELD feature manifest from label/transcription/features.")
    parser.add_argument("--feature_root", default=r"D:\datasets\meld\features")
    parser.add_argument("--label_file", default="label.npz")
    parser.add_argument("--transcription_file", default="transcription-engchi-polish.csv")
    parser.add_argument("--old_meld_pkl", default="../original_data/meld/meld.pkl")
    parser.add_argument("--out_jsonl", default="meld_feature_manifest_7way.jsonl")
    parser.add_argument("--out_csv", default="meld_feature_manifest_7way.csv")
    parser.add_argument("--out_split_dir", default="splits_7way")
    parser.add_argument(
        "--feature_dirs",
        default="chinese-hubert-large-UTT,clip-vit-large-patch14-UTT,Baichuan-13B-Base-UTT",
        help="Comma-separated feature subdirectories to include.",
    )
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    feature_dirs = [item.strip() for item in args.feature_dirs.split(",") if item.strip()]
    label_path = feature_root / args.label_file
    transcription_path = feature_root / args.transcription_file
    old_meld_pkl = Path(args.old_meld_pkl)
    if not old_meld_pkl.is_file():
        old_meld_pkl = None

    rows, report = build_manifest(feature_root, label_path, transcription_path, feature_dirs, old_meld_pkl)

    out_jsonl = feature_root / args.out_jsonl
    out_csv = feature_root / args.out_csv
    out_split_dir = feature_root / args.out_split_dir
    write_jsonl(out_jsonl, rows)
    write_csv(out_csv, rows)
    report["split_files"] = write_split_jsonl(out_split_dir, rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote JSONL: {out_jsonl}")
    print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
