import argparse
import json
import re
from pathlib import Path


LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_by_name(rows):
    latest = {}
    for row in rows:
        latest[row.get("name")] = row
    return latest


def contains_label_word(text):
    text = str(text).lower()
    return [label for label in LABELS if re.search(rf"\b{re.escape(label)}\b", text)]


def main():
    parser = argparse.ArgumentParser(description="Validate MELD Omni caption coverage and obvious quality issues.")
    parser.add_argument("--manifest", default="../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl")
    parser.add_argument("--captions", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl")
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--show_examples", type=int, default=10)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    if args.split:
        manifest_rows = [row for row in manifest_rows if row.get("split") == args.split]
    caption_rows = latest_by_name(read_jsonl(args.captions))

    missing = []
    errors = []
    empty_audio = []
    empty_video = []
    label_word_hits = []
    missing_files = []
    ok = 0

    for row in manifest_rows:
        name = row["name"]
        if not Path(row.get("video_path", "")).is_file():
            missing_files.append({"name": name, "field": "video_path", "path": row.get("video_path", "")})
        if not Path(row.get("audio_path", "")).is_file():
            missing_files.append({"name": name, "field": "audio_path", "path": row.get("audio_path", "")})
        cap = caption_rows.get(name)
        if cap is None:
            missing.append(name)
            continue
        if cap.get("status") != "ok":
            errors.append({"name": name, "error": cap.get("error", "")})
            continue
        audio_caption = str(cap.get("audio_caption", "")).strip()
        video_caption = str(cap.get("video_caption", "")).strip()
        if not audio_caption:
            empty_audio.append(name)
        if not video_caption:
            empty_video.append(name)
        hits = sorted(set(contains_label_word(audio_caption) + contains_label_word(video_caption)))
        if hits:
            label_word_hits.append({"name": name, "labels": hits})
        if audio_caption and video_caption:
            ok += 1

    report = {
        "manifest_rows": len(manifest_rows),
        "caption_records": len(caption_rows),
        "ok_with_audio_and_video": ok,
        "missing_caption": len(missing),
        "error_caption": len(errors),
        "empty_audio_caption": len(empty_audio),
        "empty_video_caption": len(empty_video),
        "missing_media_files": len(missing_files),
        "label_word_hits_for_review": len(label_word_hits),
        "examples": {
            "missing_caption": missing[: args.show_examples],
            "errors": errors[: args.show_examples],
            "empty_audio_caption": empty_audio[: args.show_examples],
            "empty_video_caption": empty_video[: args.show_examples],
            "missing_media_files": missing_files[: args.show_examples],
            "label_word_hits_for_review": label_word_hits[: args.show_examples],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
