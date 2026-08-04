import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows, split, limit):
    if split:
        rows = [row for row in rows if row.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    return rows


def extract_one(row, overwrite):
    video_path = Path(row["video_path"])
    audio_path = Path(row["audio_path"])
    if not video_path.is_file():
        return {"name": row["name"], "status": "missing_video", "video_path": str(video_path)}
    if audio_path.is_file() and audio_path.stat().st_size > 0 and not overwrite:
        return {"name": row["name"], "status": "exists", "audio_path": str(audio_path)}

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "name": row["name"],
            "status": "error",
            "audio_path": str(audio_path),
            "stderr": result.stderr.strip()[-1000:],
        }
    return {"name": row["name"], "status": "created", "audio_path": str(audio_path)}


def main():
    parser = argparse.ArgumentParser(description="Extract 16 kHz mono wav files from MELD mp4 files.")
    parser.add_argument("--manifest", default="../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl")
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="../PROCESSED_DATASET/meld/omni_caption/extract_audio_report.jsonl")
    args = parser.parse_args()

    rows = select_rows(read_jsonl(args.manifest), args.split, args.limit)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    with report_path.open("w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(extract_one, row, args.overwrite) for row in rows]
            for future in as_completed(futures):
                record = future.result()
                counts[record["status"]] = counts.get(record["status"], 0) + 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"num_rows": len(rows), "counts": counts, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
