import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path


PRED_RE = re.compile(r"preds_for_eval(?:_(\d+))?\.text$")
RUN_ID_RE = re.compile(r"_run_(\d{8}_\d{6})$")


def load_json(path):
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_score_file(pred_path):
    try:
        with pred_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            return {}
        score = json.loads(first_line)
        return score if isinstance(score, dict) else {}
    except Exception:
        return {}


def parse_epoch(pred_path):
    match = PRED_RE.match(pred_path.name)
    if not match or match.group(1) is None:
        return None
    return int(match.group(1))


def find_run_dirs(experiments_root):
    run_dirs = []
    for root, _dirs, files in os.walk(experiments_root):
        if any(PRED_RE.match(name) for name in files):
            run_dirs.append(Path(root))
    return sorted(set(run_dirs))


def infer_from_path(run_dir, experiments_root):
    try:
        rel_parts = run_dir.relative_to(experiments_root).parts
    except ValueError:
        rel_parts = run_dir.parts

    inferred = {
        "model_label": "",
        "experiment_setting": "",
        "dataset_from_path": "",
        "window_from_path": "",
        "run_name": run_dir.name,
    }
    if len(rel_parts) >= 1:
        inferred["model_label"] = rel_parts[0]
    if len(rel_parts) >= 2:
        inferred["experiment_setting"] = rel_parts[1]
    if len(rel_parts) >= 3:
        inferred["dataset_from_path"] = rel_parts[2]
    if len(rel_parts) >= 4:
        inferred["window_from_path"] = rel_parts[3]
    return inferred


def get_first(*values, default=""):
    for value in values:
        if value is not None and value != "":
            return value
    return default


def as_float(value, default=None):
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default=None):
    try:
        if value == "" or value is None:
            return default
        return int(value)
    except Exception:
        return default


def bool_text(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value) if value not in (None, "") else ""


def extract_run_name_flags(run_name):
    flags = {
        "uses_speech_from_name": "True" if "_speech" in run_name else "False",
        "uses_persona_from_name": "True" if "_persona" in run_name else "False",
        "uses_mm_prefix_from_name": "True" if "_avprefix" in run_name else "",
        "prompt_style_from_name": "qwen_chat" if "qwen_chat" in run_name else ("legacy" if "legacy" in run_name else ""),
        "data_source_from_name": "manifest" if "_manifest" in run_name else "",
        "data_format_from_name": "prc" if "_prc" in run_name else ("speechcue" if "_speechcue" in run_name else ""),
        "run_id_from_name": "",
    }
    run_id_match = RUN_ID_RE.search(run_name)
    if run_id_match:
        flags["run_id_from_name"] = run_id_match.group(1)

    patterns = {
        "lr_from_name": r"LR_([^_]+)",
        "batch_from_name": r"BS_([^_]+)",
        "data_percent_from_name": r"per_([^_]+)",
        "seed_from_name": r"class\d+_(\d+)_single_gpu",
        "mm_audio_tokens_from_name": r"_avprefix_a(\d+)_v\d+",
        "mm_video_tokens_from_name": r"_avprefix_a\d+_v(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, run_name)
        flags[key] = match.group(1) if match else ""
    return flags


def latest_mtime(paths):
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return max(times) if times else 0


def format_time(timestamp):
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def summarize_run(run_dir, experiments_root):
    pred_files = sorted(
        [p for p in run_dir.iterdir() if p.is_file() and PRED_RE.match(p.name)],
        key=lambda p: (-1 if parse_epoch(p) is None else parse_epoch(p), p.name),
    )
    if not pred_files:
        return [], None

    run_config = load_json(run_dir / "run_config.json")
    model_args = load_json(run_dir / "model_args.json")
    model_params = load_json(run_dir / "model_params.json")
    inferred = infer_from_path(run_dir, experiments_root)
    flags = extract_run_name_flags(run_dir.name)
    run_mtime = latest_mtime(pred_files + [run_dir / "run_config.json", run_dir / "model_args.json"])

    common = {
        "run_dir": str(run_dir),
        "run_name": inferred["run_name"],
        "run_id": get_first(run_config.get("run_id"), flags.get("run_id_from_name")),
        "last_modified": format_time(run_mtime),
        "dataset": get_first(run_config.get("dataset"), model_args.get("dataset"), inferred["dataset_from_path"]),
        "model_label": get_first(run_config.get("model_label"), inferred["model_label"]),
        "model_name": get_first(run_config.get("model_name")),
        "experiment_setting": get_first(run_config.get("experiments_setting"), inferred["experiment_setting"]),
        "data_source": get_first(run_config.get("data_source"), flags.get("data_source_from_name")),
        "prompt_style": get_first(run_config.get("prompt_style"), flags.get("prompt_style_from_name")),
        "data_format": get_first(run_config.get("data_format"), flags.get("data_format_from_name")),
        "model_path": get_first(run_config.get("model_path"), model_args.get("model_name_or_path")),
        "data_dir": get_first(run_config.get("data_path"), model_args.get("data_dir")),
        "batch_size": get_first(run_config.get("batch_size"), model_args.get("batch_size"), flags.get("batch_from_name")),
        "gradient_accumulation_steps": get_first(
            run_config.get("gradient_accumulation_steps"), model_args.get("gradient_accumulation_steps")
        ),
        "max_length": get_first(run_config.get("max_length"), model_args.get("max_length")),
        "learning_rate": get_first(run_config.get("learning_rate"), model_args.get("learning_rate"), flags.get("lr_from_name")),
        "num_train_epochs": get_first(run_config.get("num_train_epochs"), model_args.get("num_train_epochs")),
        "seed": get_first(run_config.get("seed"), model_args.get("seed"), flags.get("seed_from_name")),
        "lora": bool_text(get_first(run_config.get("lora"), model_args.get("lora"))),
        "lora_dim": get_first(run_config.get("lora_dim"), model_args.get("lora_dim")),
        "lora_alpha": get_first(run_config.get("lora_alpha"), model_args.get("lora_alpha")),
        "lora_dropout": get_first(run_config.get("lora_dropout"), model_args.get("lora_dropout")),
        "lora_module_name": get_first(run_config.get("lora_module_name"), model_args.get("lora_module_name")),
        "uses_speech": bool_text(get_first(run_config.get("audio_description"), flags.get("uses_speech_from_name"))),
        "uses_persona": bool_text(get_first(run_config.get("include_persona"), flags.get("uses_persona_from_name"))),
        "uses_mm_prefix": bool_text(get_first(run_config.get("use_mm_prefix"), model_args.get("use_mm_prefix"), flags.get("uses_mm_prefix_from_name"))),
        "text_guided_mm": bool_text(get_first(run_config.get("text_guided_mm"), model_args.get("text_guided_mm"))),
        "text_guided_audio": bool_text(get_first(run_config.get("text_guided_audio"), model_args.get("text_guided_audio"))),
        "text_guided_video": bool_text(get_first(run_config.get("text_guided_video"), model_args.get("text_guided_video"))),
        "text_guided_mode": get_first(run_config.get("text_guided_mode"), model_args.get("text_guided_mode")),
        "log_mm_gates": bool_text(get_first(run_config.get("log_mm_gates"), model_args.get("log_mm_gates"))),
        "mm_audio_tokens": get_first(run_config.get("mm_audio_tokens"), model_args.get("mm_audio_tokens"), flags.get("mm_audio_tokens_from_name")),
        "mm_video_tokens": get_first(run_config.get("mm_video_tokens"), model_args.get("mm_video_tokens"), flags.get("mm_video_tokens_from_name")),
        "skip_missing_mm": bool_text(get_first(run_config.get("skip_missing_mm"), model_args.get("skip_missing_mm"))),
        "mm_audio_feature_dir": get_first(run_config.get("mm_audio_feature_dir"), model_args.get("mm_audio_feature_dir")),
        "mm_video_feature_dir": get_first(run_config.get("mm_video_feature_dir"), model_args.get("mm_video_feature_dir")),
        "historical_window": get_first(run_config.get("historical_window"), inferred["window_from_path"].replace("window_", "")),
        "trainable_params": model_params.get("Trainable", ""),
        "total_params": model_params.get("Total", ""),
        "cuda_visible_devices": get_first(run_config.get("cuda_visible_devices")),
        "master_port": get_first(run_config.get("master_port")),
    }

    rows = []
    best = None
    for pred_path in pred_files:
        score = parse_score_file(pred_path)
        epoch = parse_epoch(pred_path)
        row = dict(common)
        row.update(
            {
                "pred_file": pred_path.name,
                "epoch": "" if epoch is None else epoch,
                "Acc_SA": score.get("Acc_SA", ""),
                "F1_SA": score.get("F1_SA", ""),
                "mode": score.get("mode", ""),
                "status": "has_score" if score else "no_score",
            }
        )
        rows.append(row)
        f1 = as_float(row["F1_SA"])
        if f1 is not None and (best is None or f1 > as_float(best["F1_SA"], -1)):
            best = row

    return rows, best


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, ""))
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compact_run_desc(row):
    parts = [
        f"model={row.get('model_label', '') or row.get('model_name', '')}",
        f"source={row.get('data_source', '')}",
        f"mm={row.get('uses_mm_prefix', '')}",
        f"a/v={row.get('mm_audio_tokens', '')}/{row.get('mm_video_tokens', '')}",
        f"lora={row.get('lora_dim', '')}/{row.get('lora_alpha', '')}",
        f"modules={row.get('lora_module_name', '')}",
    ]
    return "; ".join(parts)


def write_markdown(path, best_rows, all_rows, top_k):
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    complete_rows = [r for r in best_rows if as_float(r.get("F1_SA")) is not None]
    by_dataset = {}
    for row in complete_rows:
        by_dataset.setdefault(row.get("dataset", ""), []).append(row)

    lines = [
        "# Experiment Summary",
        "",
        f"Generated: {now}",
        "",
        f"Runs with scores: {len(complete_rows)}",
        f"Epoch-level rows: {len(all_rows)}",
        "",
    ]

    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda r: as_float(r.get("F1_SA"), -1), reverse=True)[:top_k]
        for row in rows:
            row["config"] = compact_run_desc(row)
        lines.extend(
            [
                f"## {dataset or 'unknown'} Top {min(top_k, len(rows))}",
                "",
                markdown_table(
                    rows,
                    [
                        "F1_SA",
                        "Acc_SA",
                        "epoch",
                        "model_label",
                        "data_source",
                        "uses_mm_prefix",
                        "mm_audio_tokens",
                        "mm_video_tokens",
                        "lora_dim",
                        "lora_alpha",
                        "lora_module_name",
                        "run_name",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- `all_results.csv` is epoch-level; one row per `preds_for_eval*.text`.",
            "- `best_results.csv` keeps the best F1 row for each run directory.",
            "- `run_config.json` is preferred when present; otherwise the script falls back to `model_args.json` and path parsing.",
            "",
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("dataset", "")),
            -as_float(r.get("F1_SA"), -1),
            str(r.get("last_modified", "")),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Collect and rank Our-MSER experiment results.")
    parser.add_argument("--experiments_root", default="../experiments", help="Root directory containing experiment runs.")
    parser.add_argument("--out_dir", default="../experiments/summary", help="Directory for summary files.")
    parser.add_argument("--all_csv", default="all_results.csv", help="Epoch-level CSV filename.")
    parser.add_argument("--best_csv", default="best_results.csv", help="Best-per-run CSV filename.")
    parser.add_argument("--md", default="experiment_summary.md", help="Markdown leaderboard filename.")
    parser.add_argument("--top_k", type=int, default=20, help="Rows per dataset in Markdown leaderboard.")
    parser.add_argument("--print_best", action="store_true", help="Print best rows to stdout.")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    run_dirs = find_run_dirs(experiments_root)
    all_rows = []
    best_rows = []
    for run_dir in run_dirs:
        rows, best = summarize_run(run_dir, experiments_root)
        all_rows.extend(rows)
        if best:
            best_rows.append(best)

    all_rows = sort_rows(all_rows)
    best_rows = sort_rows(best_rows)

    fieldnames = [
        "dataset",
        "model_label",
        "model_name",
        "experiment_setting",
        "data_source",
        "prompt_style",
        "data_format",
        "uses_speech",
        "uses_persona",
        "uses_mm_prefix",
        "mm_audio_tokens",
        "mm_video_tokens",
        "mm_audio_feature_dir",
        "mm_video_feature_dir",
        "historical_window",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "seed",
        "lora",
        "lora_dim",
        "lora_alpha",
        "lora_dropout",
        "lora_module_name",
        "Acc_SA",
        "F1_SA",
        "epoch",
        "mode",
        "status",
        "pred_file",
        "run_id",
        "last_modified",
        "trainable_params",
        "total_params",
        "cuda_visible_devices",
        "master_port",
        "model_path",
        "data_dir",
        "run_name",
        "run_dir",
    ]

    all_csv_path = out_dir / args.all_csv
    best_csv_path = out_dir / args.best_csv
    md_path = out_dir / args.md
    write_csv(all_csv_path, all_rows, fieldnames)
    write_csv(best_csv_path, best_rows, fieldnames)
    write_markdown(md_path, best_rows, all_rows, args.top_k)

    print(f"Experiments root: {experiments_root}")
    print(f"Found run dirs: {len(run_dirs)}")
    print(f"Epoch rows: {len(all_rows)}")
    print(f"Best rows: {len(best_rows)}")
    print(f"Wrote: {all_csv_path}")
    print(f"Wrote: {best_csv_path}")
    print(f"Wrote: {md_path}")

    if args.print_best:
        for row in best_rows:
            print(
                f"{row['dataset']}\tF1={row['F1_SA']}\tAcc={row['Acc_SA']}\t"
                f"epoch={row['epoch']}\t{compact_run_desc(row)}\t{row['run_name']}"
            )


if __name__ == "__main__":
    main()
