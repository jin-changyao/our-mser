import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np


LABELS_6WAY = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]


def parse_utterance_id(utterance_id):
    match = re.match(r"^(Ses\d{2}[FM]_.+)_([FM])(\d+|XX\d+)$", utterance_id)
    if not match:
        return {
            "dialog_id": "",
            "speaker": "",
            "turn_code": "",
            "session": utterance_id[:5],
        }
    return {
        "dialog_id": match.group(1),
        "speaker": match.group(2),
        "turn_code": match.group(3),
        "session": utterance_id[:5],
    }


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
                "utterance_id": utterance_id,
                "text": row.get("english", "").strip(),
                "text_zh": row.get("chinese", "").strip(),
            }
    return rows


def read_labels(path):
    payload = np.load(path, allow_pickle=True)
    labels = payload["whole_corpus"].item()
    return labels


def read_split_csv(path):
    if path is None or not path.is_file():
        return {}, {}
    split_by_id = {}
    csv_meta_by_id = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "mode", "text", "emotion", "video_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing split csv columns: {sorted(missing)}")
        for row in reader:
            utterance_id = Path(row["path"]).stem
            split_by_id[utterance_id] = row["mode"]
            csv_meta_by_id[utterance_id] = row
    return split_by_id, csv_meta_by_id


def read_processed_targets(processed_data_dir):
    if processed_data_dir is None or not processed_data_dir.is_dir():
        return {}
    rows = {}
    for split in ["train", "valid", "test"]:
        path = processed_data_dir / f"{split}.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                utterance_id = Path(item["path"]).stem
                rows[utterance_id] = {
                    "split": split,
                    "label": item["target"],
                    "source_path": item["path"],
                }
    return rows


def feature_path(feature_root, feature_dir, utterance_id):
    path = feature_root / feature_dir / f"{utterance_id}.npy"
    return str(path) if path.is_file() else ""


def build_manifest(feature_root, label_path, transcription_path, feature_dirs, split_csv=None, processed_data_dir=None):
    transcriptions = read_transcriptions(transcription_path)
    labels = read_labels(label_path)
    split_by_id, csv_meta_by_id = read_split_csv(split_csv)
    processed_targets = read_processed_targets(processed_data_dir)

    rows = []
    missing_transcription = []
    missing_split = []
    missing_features = {name: [] for name in feature_dirs}
    row_ids = set()
    for utterance_id, label_info in labels.items():
        parsed = parse_utterance_id(utterance_id)
        text_info = transcriptions.get(utterance_id)
        if text_info is None:
            missing_transcription.append(utterance_id)
            text = ""
            text_zh = ""
        else:
            text = text_info["text"]
            text_zh = text_info["text_zh"]

        split = split_by_id.get(utterance_id, "")
        if split_csv is not None and not split:
            missing_split.append(utterance_id)

        label_id = int(label_info["emo"])
        row = {
            "split": split,
            "utterance_id": utterance_id,
            "dialog_id": parsed["dialog_id"],
            "session": parsed["session"],
            "speaker": parsed["speaker"],
            "turn_code": parsed["turn_code"],
            "text": text,
            "text_zh": text_zh,
            "label_id": label_id,
            "label": LABELS_6WAY[label_id] if 0 <= label_id < len(LABELS_6WAY) else "",
            "val": label_info.get("val", ""),
        }
        if utterance_id in csv_meta_by_id:
            row["source_csv_text"] = csv_meta_by_id[utterance_id].get("text", "")
            row["source_csv_emotion"] = csv_meta_by_id[utterance_id].get("emotion", "")
            row["source_csv_path"] = csv_meta_by_id[utterance_id].get("path", "")
        for name in feature_dirs:
            path = feature_path(feature_root, name, utterance_id)
            row[f"feature_{name}"] = path
            if not path:
                missing_features[name].append(utterance_id)
        rows.append(row)
        row_ids.add(utterance_id)

    supplemented = []
    missing_supplement_features = []
    for utterance_id, processed_info in processed_targets.items():
        if utterance_id in row_ids:
            continue
        feature_paths = {name: feature_path(feature_root, name, utterance_id) for name in feature_dirs}
        if not all(feature_paths.values()):
            missing_supplement_features.append(utterance_id)
            continue
        parsed = parse_utterance_id(utterance_id)
        text_info = transcriptions.get(utterance_id, {})
        csv_meta = csv_meta_by_id.get(utterance_id, {})
        label = processed_info["label"]
        label_id = LABELS_6WAY.index(label) if label in LABELS_6WAY else -1
        row = {
            "split": processed_info["split"],
            "utterance_id": utterance_id,
            "dialog_id": parsed["dialog_id"],
            "session": parsed["session"],
            "speaker": parsed["speaker"],
            "turn_code": parsed["turn_code"],
            "text": text_info.get("text", csv_meta.get("text", "")),
            "text_zh": text_info.get("text_zh", ""),
            "label_id": label_id,
            "label": label,
            "val": "",
            "supplement_source": "processed_json",
            "source_csv_text": csv_meta.get("text", ""),
            "source_csv_emotion": csv_meta.get("emotion", ""),
            "source_csv_path": csv_meta.get("path", processed_info.get("source_path", "")),
        }
        for name, path in feature_paths.items():
            row[f"feature_{name}"] = path
        rows.append(row)
        row_ids.add(utterance_id)
        supplemented.append(utterance_id)

    rows.sort(key=lambda item: item["utterance_id"])
    report = {
        "num_labels": len(labels),
        "num_transcriptions": len(transcriptions),
        "num_manifest_rows": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "missing_transcription": len(missing_transcription),
        "missing_transcription_examples": missing_transcription[:10],
        "missing_split": len(missing_split),
        "missing_split_examples": missing_split[:10],
        "missing_features": {
            name: {
                "count": len(ids),
                "examples": ids[:10],
            }
            for name, ids in missing_features.items()
        },
        "supplemented_from_processed_json": len(supplemented),
        "supplemented_examples": supplemented[:10],
        "missing_supplement_features": len(missing_supplement_features),
        "missing_supplement_feature_examples": missing_supplement_features[:10],
    }
    return rows, report


def write_split_jsonl(out_dir, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name_map = {"train": "train", "valid": "valid", "test": "test"}
    written = {}
    for split in ["train", "valid", "test"]:
        split_rows = [row for row in rows if row.get("split") == split]
        out_path = out_dir / f"iemocap_multimodal_{split_name_map[split]}.jsonl"
        write_jsonl(out_path, split_rows)
        written[split] = str(out_path)
    return written


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build an IEMOCAP feature manifest from label/transcription/features.")
    parser.add_argument("--feature_root", default=r"D:\datasets\iemocap\features")
    parser.add_argument("--label_file", default="label_6way.npz")
    parser.add_argument("--transcription_file", default="transcription-engchi-polish.csv")
    parser.add_argument("--split_csv", default="../data/IEMOCAP_full_release/iemocap_full_dataset.csv")
    parser.add_argument("--processed_data_dir", default="../PROCESSED_DATASET/iemocap/window/False_False_qwen_chat_prc")
    parser.add_argument("--out_jsonl", default="iemocap_feature_manifest_6way.jsonl")
    parser.add_argument("--out_csv", default="iemocap_feature_manifest_6way.csv")
    parser.add_argument("--out_split_dir", default="splits_6way")
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
    split_csv = Path(args.split_csv)
    if not split_csv.is_file():
        split_csv = None
    out_jsonl = feature_root / args.out_jsonl
    out_csv = feature_root / args.out_csv
    out_split_dir = feature_root / args.out_split_dir

    processed_data_dir = Path(args.processed_data_dir)
    if not processed_data_dir.is_dir():
        processed_data_dir = None

    rows, report = build_manifest(feature_root, label_path, transcription_path, feature_dirs, split_csv, processed_data_dir)
    write_jsonl(out_jsonl, rows)
    write_csv(out_csv, rows)
    split_files = write_split_jsonl(out_split_dir, rows) if split_csv is not None else {}
    report["split_files"] = split_files

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote JSONL: {out_jsonl}")
    print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
