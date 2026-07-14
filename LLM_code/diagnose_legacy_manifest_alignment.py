import argparse
import csv
import json
import os
import re
from pathlib import Path


DEFAULT_FEATURE_DIRS = [
    "chinese-hubert-large-UTT",
    "clip-vit-large-patch14-UTT",
]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_legacy_json(path):
    if not path.is_file():
        return []
    return read_jsonl(path)


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", text)
    return text.strip()


def compact_text(text, max_len=220):
    text = " ".join(str(text or "").split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def infer_split_from_file_name(file_name):
    stem = Path(file_name).stem.lower()
    if "train" in stem:
        return "train"
    if "valid" in stem or "dev" in stem:
        return "valid"
    if "test" in stem:
        return "test"
    return ""


def manifest_key_from_path(dataset, split, path):
    stem = Path(str(path)).stem
    if stem.startswith(("train_", "val_", "test_", "Ses")):
        return stem
    if dataset == "iemocap":
        return stem
    if dataset == "meld":
        split_prefix = {"train": "train", "valid": "val", "dev": "val", "test": "test"}.get(split, split)
        return f"{split_prefix}_{stem}"
    return stem


def load_manifest(manifest_dir, dataset):
    manifest_dir = Path(manifest_dir)
    by_split = {}
    for split in ["train", "test", "valid"]:
        path = manifest_dir / f"{dataset}_multimodal_{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Manifest split not found: {path}")
        rows = read_jsonl(path)
        by_split[split] = {row["utterance_id"]: row for row in rows}
    return by_split


def render_input(example_input):
    if isinstance(example_input, list):
        return "\n\n".join(str(message.get("content", "")) for message in example_input)
    return str(example_input)


def extract_target_utterance_text(rendered):
    marker = "Target utterance:"
    if marker in rendered:
        part = rendered.split(marker, 1)[1]
        for stop in ["\n\nTarget speech characteristics:", "\n\nSpeaker profiles:", "\n\nQuestion:"]:
            if stop in part:
                part = part.split(stop, 1)[0]
        return strip_speaker_quote(part)
    match = re.search(r"label of <(.+?)> from <", rendered, flags=re.S)
    if match:
        return strip_speaker_quote(match.group(1))
    return ""


def strip_speaker_quote(text):
    text = " ".join(str(text or "").strip().split())
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1]
    return text.strip()


def prompt_has_speech(rendered):
    markers = [
        "Target speech characteristics:",
        "speech characteristics",
        "pitch",
        "volume",
        "speaking rate",
        "audio features",
    ]
    rendered_lower = rendered.lower()
    return any(marker.lower() in rendered_lower for marker in markers)


def read_csv_rows(path):
    if not path or not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_speech_indexes(dataset, speech_feature_dir):
    speech_feature_dir = Path(speech_feature_dir)
    indexes = {}
    if dataset == "iemocap":
        rows = read_csv_rows(speech_feature_dir / "processed_iemocap_audio_features_5.csv")
        by_dialog_text = set()
        for row in rows:
            by_dialog_text.add((str(row.get("video_id", "")), normalize_text(row.get("text", ""))))
        indexes["iemocap_dialog_text"] = by_dialog_text
        indexes["row_count"] = len(rows)
        return indexes

    if dataset == "meld":
        for split, filename in [
            ("train", "meld_processed_5class_train_audio_features.csv"),
            ("test", "meld_processed_5class_test_audio_features.csv"),
            ("valid", "meld_processed_5class_dev_audio_features.csv"),
        ]:
            rows = read_csv_rows(speech_feature_dir / filename)
            indexes[split] = {row.get("filename", "") for row in rows}
        indexes["row_count"] = sum(len(v) for k, v in indexes.items() if isinstance(v, set))
        return indexes
    return indexes


def speech_feature_exists(dataset, split, utterance_id, manifest_row, target_text, speech_indexes):
    if dataset == "iemocap":
        dialog_id = str(manifest_row.get("dialog_id", ""))
        candidates = [
            normalize_text(target_text),
            normalize_text(manifest_row.get("text", "")),
            normalize_text(manifest_row.get("source_csv_text", "")),
        ]
        keys = speech_indexes.get("iemocap_dialog_text", set())
        return any((dialog_id, text) in keys for text in candidates if text)

    if dataset == "meld":
        stem = utterance_id
        for prefix in ["train_", "val_", "test_"]:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        filename = f"{stem}.wav"
        return filename in speech_indexes.get(split, set())
    return False


def compare_texts(target_text, manifest_text):
    old_norm = normalize_text(target_text)
    new_norm = normalize_text(manifest_text)
    if not old_norm or not new_norm:
        return "missing"
    if old_norm == new_norm:
        return "exact"
    if old_norm in new_norm or new_norm in old_norm:
        return "contains"
    return "different"


def inspect_split(dataset, split, legacy_rows, manifest_by_id, feature_dirs, speech_indexes, sample_limit):
    counts = {
        "legacy_rows": len(legacy_rows),
        "missing_manifest": 0,
        "label_mismatch": 0,
        "missing_feature": 0,
        "missing_speech_feature": 0,
        "prompt_without_speech_marker": 0,
        "text_exact": 0,
        "text_contains": 0,
        "text_different": 0,
        "text_missing": 0,
    }
    examples = []
    manifest_seen = set()
    for index, item in enumerate(legacy_rows):
        rendered = render_input(item.get("input", ""))
        target_text = extract_target_utterance_text(rendered)
        utterance_id = manifest_key_from_path(dataset, split, item.get("path", ""))
        manifest_row = manifest_by_id.get(utterance_id)
        problems = []

        if manifest_row is None:
            counts["missing_manifest"] += 1
            problems.append("missing_manifest")
        else:
            manifest_seen.add(utterance_id)
            if item.get("target") != manifest_row.get("label"):
                counts["label_mismatch"] += 1
                problems.append("label_mismatch")

            missing_features = []
            for feature_dir in feature_dirs:
                feature_path = manifest_row.get(f"feature_{feature_dir}", "")
                if not feature_path or not os.path.isfile(feature_path):
                    missing_features.append(feature_dir)
            if missing_features:
                counts["missing_feature"] += 1
                problems.append("missing_feature:" + ",".join(missing_features))

            text_status = compare_texts(target_text, manifest_row.get("text", ""))
            counts[f"text_{text_status}"] += 1
            if text_status in {"different", "missing"}:
                problems.append(f"text_{text_status}")

            if not speech_feature_exists(dataset, split, utterance_id, manifest_row, target_text, speech_indexes):
                counts["missing_speech_feature"] += 1
                problems.append("missing_speech_feature")

        if not prompt_has_speech(rendered):
            counts["prompt_without_speech_marker"] += 1
            problems.append("prompt_without_speech_marker")

        if problems and len(examples) < sample_limit:
            examples.append(
                {
                    "split": split,
                    "index": index,
                    "legacy_path": item.get("path", ""),
                    "utterance_id": utterance_id,
                    "target": item.get("target", ""),
                    "manifest_label": manifest_row.get("label", "") if manifest_row else "",
                    "target_text": compact_text(target_text),
                    "manifest_text": compact_text(manifest_row.get("text", "") if manifest_row else ""),
                    "problems": problems,
                }
            )

    extra_manifest_rows = len(set(manifest_by_id) - manifest_seen)
    counts["extra_manifest_rows_not_used_by_legacy"] = extra_manifest_rows
    return counts, examples


def main():
    parser = argparse.ArgumentParser(
        description="Check whether legacy SpeechCue prompts align with C-manifest audio/video features."
    )
    parser.add_argument("--dataset", required=True, choices=["iemocap", "meld"])
    parser.add_argument("--legacy_data_dir", required=True, help="Directory containing legacy train/test/valid.json.")
    parser.add_argument("--manifest_dir", required=True, help="Directory containing *_multimodal_train/test/valid.jsonl.")
    parser.add_argument("--speech_feature_dir", default="../speech_features")
    parser.add_argument("--feature_dirs", default=",".join(DEFAULT_FEATURE_DIRS))
    parser.add_argument("--out", default="")
    parser.add_argument("--sample_limit", type=int, default=20)
    args = parser.parse_args()

    legacy_data_dir = Path(args.legacy_data_dir)
    manifest = load_manifest(args.manifest_dir, args.dataset)
    speech_indexes = build_speech_indexes(args.dataset, args.speech_feature_dir)
    feature_dirs = [item.strip() for item in args.feature_dirs.split(",") if item.strip()]

    report = {
        "dataset": args.dataset,
        "legacy_data_dir": str(legacy_data_dir),
        "manifest_dir": args.manifest_dir,
        "speech_feature_dir": args.speech_feature_dir,
        "feature_dirs": feature_dirs,
        "speech_feature_rows": speech_indexes.get("row_count", 0),
        "splits": {},
        "problem_examples": [],
    }

    for split in ["train", "test", "valid"]:
        legacy_rows = read_legacy_json(legacy_data_dir / f"{split}.json")
        counts, examples = inspect_split(
            dataset=args.dataset,
            split=split,
            legacy_rows=legacy_rows,
            manifest_by_id=manifest[split],
            feature_dirs=feature_dirs,
            speech_indexes=speech_indexes,
            sample_limit=args.sample_limit - len(report["problem_examples"]),
        )
        report["splits"][split] = counts
        report["problem_examples"].extend(examples)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
