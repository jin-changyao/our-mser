import argparse
import csv
import json
import os
from pathlib import Path


SPLITS = ["train", "dev", "test"]
KNOWN_VIDEO_DIRS = {
    "train": ["train_splits", "train"],
    "dev": ["dev_splits_complete", "dev_splits", "dev"],
    "test": ["output_repeated_splits_test", "test_splits", "test"],
}


def clean_text(value):
    return " ".join(str(value).replace("\ufeff", "").split())


def get_first(row, names, default=""):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return default


def to_int(value, field_name):
    try:
        return int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Cannot parse {field_name}={value!r} as int.") from exc


def iter_mp4_files(root):
    for path in root.rglob("*.mp4"):
        if path.name.startswith("._"):
            continue
        yield path


def index_videos(raw_root):
    by_split_name = {}
    by_name = {}
    for split in SPLITS:
        by_split_name[split] = {}
        for dirname in KNOWN_VIDEO_DIRS[split]:
            candidate = raw_root / dirname
            if not candidate.is_dir():
                continue
            for path in iter_mp4_files(candidate):
                by_split_name[split][path.name] = path.resolve()

    for path in iter_mp4_files(raw_root):
        by_name.setdefault(path.name, []).append(path.resolve())
    return by_split_name, by_name


def find_video_path(split, dialog_id, utterance_id, by_split_name, by_name):
    file_name = f"dia{dialog_id}_utt{utterance_id}.mp4"
    split_match = by_split_name.get(split, {}).get(file_name)
    if split_match is not None:
        return str(split_match)

    candidates = by_name.get(file_name, [])
    if not candidates:
        return ""

    split_hints = KNOWN_VIDEO_DIRS.get(split, [])
    for path in candidates:
        path_text = str(path).replace("\\", "/").lower()
        if any(hint.lower() in path_text for hint in split_hints):
            return str(path)
    return str(candidates[0])


def read_split_csv(raw_root, split):
    path = raw_root / f"{split}_sent_emo.csv"
    if not path.is_file():
        raise FileNotFoundError(f"MELD CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), path


def build_manifest(raw_root, audio_root):
    raw_root = raw_root.resolve()
    audio_root = audio_root.resolve()
    by_split_name, by_name = index_videos(raw_root)

    rows = []
    missing_video = []
    for split in SPLITS:
        csv_rows, csv_path = read_split_csv(raw_root, split)
        for csv_index, row in enumerate(csv_rows):
            dialog_id = to_int(
                get_first(row, ["Dialogue_ID", "Dialogue Id", "DialogueID", "dialogue_id"]),
                "Dialogue_ID",
            )
            utterance_id = to_int(
                get_first(row, ["Utterance_ID", "Utterance Id", "UtteranceID", "utterance_id"]),
                "Utterance_ID",
            )
            name = f"{split}_dia{dialog_id}_utt{utterance_id}"
            video_path = find_video_path(split, dialog_id, utterance_id, by_split_name, by_name)
            if not video_path:
                missing_video.append(name)
            rows.append(
                {
                    "name": name,
                    "split": split,
                    "dialogue_id": dialog_id,
                    "utterance_id": utterance_id,
                    "csv_index": csv_index,
                    "speaker": clean_text(get_first(row, ["Speaker", "speaker"])),
                    "text": clean_text(get_first(row, ["Utterance", "utterance", "text"])),
                    "label": clean_text(get_first(row, ["Emotion", "emotion", "label"])).lower(),
                    "sentiment": clean_text(get_first(row, ["Sentiment", "sentiment"])).lower(),
                    "video_path": video_path,
                    "audio_path": str(audio_root / split / f"{name}.wav"),
                    "source_csv": str(csv_path.resolve()),
                }
            )

    report = {
        "raw_root": str(raw_root),
        "audio_root": str(audio_root),
        "num_rows": len(rows),
        "split_counts": {split: sum(1 for row in rows if row["split"] == split) for split in SPLITS},
        "missing_video": len(missing_video),
        "missing_video_examples": missing_video[:20],
    }
    return rows, report


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a MELD raw manifest for Omni caption generation.")
    parser.add_argument(
        "--raw_root",
        default=os.environ.get("MELD_RAW_ROOT", "/home/pc/jcy/MELD_raw/MELD.Raw"),
        help="Directory containing MELD CSV files and extracted split video folders.",
    )
    parser.add_argument(
        "--audio_root",
        default=None,
        help="Where extracted 16 kHz wav files should live. Defaults to ../audio_16k beside MELD.Raw.",
    )
    parser.add_argument(
        "--out",
        default="../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl",
        help="Output JSONL manifest path.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    audio_root = Path(args.audio_root) if args.audio_root else raw_root.parent / "audio_16k"
    rows, report = build_manifest(raw_root, audio_root)
    out_path = Path(args.out)
    write_jsonl(out_path, rows)
    report_path = out_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote manifest: {out_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
