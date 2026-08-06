import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PRED_RE = re.compile(r"preds_for_eval(?:_(\d+))?\.text$")
RUN_ID_RE = re.compile(r"_run_(\d{8}_\d{6})$")


CORE_FIELDS = [
    "dataset",
    "method_family",
    "method_signature",
    "model_label",
    "experiment_setting",
    "F1_SA",
    "Acc_SA",
    "epoch",
    "seed",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "historical_window",
    "max_length",
    "lora_dim",
    "lora_alpha",
    "lora_module_name",
    "use_audio_caption",
    "use_video_caption",
    "uses_speech",
    "uses_persona",
    "uses_mm_prefix",
    "text_guided_mm",
    "mm_audio_tokens",
    "mm_video_tokens",
    "mm_audio_feature_dir",
    "mm_video_feature_dir",
    "run_id",
    "last_modified",
    "run_name",
    "run_dir",
]


def load_json(path):
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_score_file(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        score = json.loads(first) if first else {}
        return score if isinstance(score, dict) else {}
    except Exception:
        return {}


def parse_epoch(path):
    match = PRED_RE.match(path.name)
    if not match or match.group(1) is None:
        return None
    return int(match.group(1))


def as_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def get_first(*values, default=""):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def bool_text(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if value in (None, ""):
        return ""
    return str(value)


def latest_mtime(paths):
    values = [p.stat().st_mtime for p in paths if p.exists()]
    return max(values) if values else 0


def format_time(timestamp):
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def infer_from_path(run_dir, experiments_root):
    try:
        parts = run_dir.relative_to(experiments_root).parts
    except ValueError:
        parts = run_dir.parts
    return {
        "model_label": parts[0] if len(parts) > 0 else "",
        "experiment_setting": parts[1] if len(parts) > 1 else "",
        "dataset": parts[2] if len(parts) > 2 else "",
        "window": parts[3].replace("window_", "") if len(parts) > 3 else "",
        "run_name": run_dir.name,
    }


def run_name_flags(run_name):
    flags = {
        "uses_speech": "True" if "_speech" in run_name else "False",
        "uses_persona": "True" if "_persona" in run_name else "False",
        "uses_mm_prefix": "True" if "_avprefix" in run_name else "",
        "prompt_style": "qwen_chat" if "qwen_chat" in run_name else ("legacy" if "legacy" in run_name else ""),
        "data_source": "manifest" if "_manifest" in run_name else "",
        "run_id": "",
        "seed": "",
        "lr": "",
        "batch_size": "",
        "mm_audio_tokens": "",
        "mm_video_tokens": "",
    }
    patterns = {
        "run_id": RUN_ID_RE,
        "seed": re.compile(r"class\d+_(\d+)_single_gpu"),
        "lr": re.compile(r"LR_([^_]+)"),
        "batch_size": re.compile(r"BS_([^_]+)"),
        "mm_audio_tokens": re.compile(r"_avprefix_a(\d+)_v\d+"),
        "mm_video_tokens": re.compile(r"_avprefix_a\d+_v(\d+)"),
    }
    for key, pattern in patterns.items():
        match = pattern.search(run_name)
        if match:
            flags[key] = match.group(1)
    return flags


def method_family(row):
    data_source = str(row.get("data_source", "")).lower()
    run_name = str(row.get("run_name", "")).lower()
    uses_mm_prefix = str(row.get("uses_mm_prefix", "")).lower() == "true"
    uses_speech = str(row.get("uses_speech", "")).lower() == "true"
    uses_audio_caption = str(row.get("use_audio_caption", "")).lower() == "true"
    uses_video_caption = str(row.get("use_video_caption", "")).lower() == "true"
    text_guided = str(row.get("text_guided_mm", "")).lower() == "true"
    if data_source == "omni_caption" or "omni_caption" in run_name:
        if uses_audio_caption and uses_video_caption:
            return "omni_caption_av"
        if uses_audio_caption:
            return "omni_caption_audio"
        if uses_video_caption:
            return "omni_caption_video"
        return "omni_caption"
    if uses_mm_prefix and text_guided:
        return "av_prefix_text_guided"
    if uses_mm_prefix:
        return "av_prefix"
    if uses_speech:
        return "speech_description"
    if str(row.get("uses_persona", "")).lower() == "true":
        return "persona_text"
    return "text_or_legacy"


def method_signature(row):
    family = row.get("method_family") or method_family(row)
    parts = [family]
    for key in [
        "model_label",
        "dataset",
        "historical_window",
        "prompt_style",
        "lora_dim",
        "lora_alpha",
        "lora_module_name",
        "mm_audio_tokens",
        "mm_video_tokens",
        "text_guided_mm",
    ]:
        value = row.get(key, "")
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def find_run_dirs(experiments_root):
    run_dirs = []
    for root, _dirs, files in os.walk(experiments_root):
        if any(PRED_RE.match(name) for name in files):
            run_dirs.append(Path(root))
    return sorted(set(run_dirs))


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
    flags = run_name_flags(run_dir.name)
    mtime = latest_mtime(pred_files + [run_dir / "run_config.json", run_dir / "model_args.json"])

    common = {
        "run_dir": str(run_dir),
        "run_name": inferred["run_name"],
        "run_id": get_first(run_config.get("run_id"), flags.get("run_id")),
        "last_modified": format_time(mtime),
        "dataset": get_first(run_config.get("dataset"), model_args.get("dataset"), inferred.get("dataset")),
        "model_label": get_first(run_config.get("model_label"), inferred.get("model_label")),
        "model_name": get_first(run_config.get("model_name")),
        "experiment_setting": get_first(run_config.get("experiments_setting"), inferred.get("experiment_setting")),
        "data_source": get_first(run_config.get("data_source"), flags.get("data_source")),
        "prompt_style": get_first(run_config.get("prompt_style"), flags.get("prompt_style")),
        "model_path": get_first(run_config.get("model_path"), model_args.get("model_name_or_path")),
        "data_dir": get_first(run_config.get("data_path"), model_args.get("data_dir")),
        "caption_path": get_first(run_config.get("caption_path")),
        "use_audio_caption": bool_text(run_config.get("use_audio_caption")),
        "use_video_caption": bool_text(run_config.get("use_video_caption")),
        "caption_limit": get_first(run_config.get("caption_limit")),
        "video_fps": get_first(run_config.get("video_fps")),
        "video_max_pixels": get_first(run_config.get("video_max_pixels")),
        "batch_size": get_first(run_config.get("batch_size"), model_args.get("batch_size"), flags.get("batch_size")),
        "gradient_accumulation_steps": get_first(run_config.get("gradient_accumulation_steps"), model_args.get("gradient_accumulation_steps")),
        "max_length": get_first(run_config.get("max_length"), model_args.get("max_length")),
        "learning_rate": get_first(run_config.get("learning_rate"), model_args.get("learning_rate"), flags.get("lr")),
        "num_train_epochs": get_first(run_config.get("num_train_epochs"), model_args.get("num_train_epochs")),
        "seed": get_first(run_config.get("seed"), model_args.get("seed"), flags.get("seed")),
        "lora": bool_text(get_first(run_config.get("lora"), model_args.get("lora"))),
        "lora_dim": get_first(run_config.get("lora_dim"), model_args.get("lora_dim")),
        "lora_alpha": get_first(run_config.get("lora_alpha"), model_args.get("lora_alpha")),
        "lora_dropout": get_first(run_config.get("lora_dropout"), model_args.get("lora_dropout")),
        "lora_module_name": get_first(run_config.get("lora_module_name"), model_args.get("lora_module_name")),
        "uses_speech": bool_text(get_first(run_config.get("audio_description"), flags.get("uses_speech"))),
        "uses_persona": bool_text(get_first(run_config.get("include_persona"), flags.get("uses_persona"))),
        "uses_mm_prefix": bool_text(get_first(run_config.get("use_mm_prefix"), model_args.get("use_mm_prefix"), flags.get("uses_mm_prefix"))),
        "text_guided_mm": bool_text(get_first(run_config.get("text_guided_mm"), model_args.get("text_guided_mm"))),
        "mm_audio_tokens": get_first(run_config.get("mm_audio_tokens"), model_args.get("mm_audio_tokens"), flags.get("mm_audio_tokens")),
        "mm_video_tokens": get_first(run_config.get("mm_video_tokens"), model_args.get("mm_video_tokens"), flags.get("mm_video_tokens")),
        "mm_audio_feature_dir": get_first(run_config.get("mm_audio_feature_dir"), model_args.get("mm_audio_feature_dir")),
        "mm_video_feature_dir": get_first(run_config.get("mm_video_feature_dir"), model_args.get("mm_video_feature_dir")),
        "historical_window": get_first(run_config.get("historical_window"), inferred.get("window")),
        "trainable_params": model_params.get("Trainable", ""),
        "total_params": model_params.get("Total", ""),
    }

    rows = []
    best = None
    for pred_path in pred_files:
        score = parse_score_file(pred_path)
        row = dict(common)
        row.update(
            {
                "pred_file": pred_path.name,
                "epoch": "" if parse_epoch(pred_path) is None else parse_epoch(pred_path),
                "Acc_SA": score.get("Acc_SA", ""),
                "F1_SA": score.get("F1_SA", ""),
                "mode": score.get("mode", ""),
                "status": "has_score" if score else "no_score",
            }
        )
        row["method_family"] = method_family(row)
        row["method_signature"] = method_signature(row)
        rows.append(row)
        f1 = as_float(row["F1_SA"])
        if f1 is not None and (best is None or f1 > as_float(best.get("F1_SA"), -1)):
            best = row
    return rows, best


def sort_rows(rows):
    return sorted(rows, key=lambda r: (str(r.get("dataset", "")), -as_float(r.get("F1_SA"), -1), str(r.get("last_modified", ""))))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, ""))
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def group_stats(best_rows):
    groups = defaultdict(list)
    for row in best_rows:
        if as_float(row.get("F1_SA")) is not None:
            groups[(row.get("dataset", ""), row.get("method_family", ""))].append(row)
    stats = []
    for (dataset, family), rows in groups.items():
        f1s = [as_float(r.get("F1_SA")) for r in rows]
        accs = [as_float(r.get("Acc_SA")) for r in rows if as_float(r.get("Acc_SA")) is not None]
        best = max(rows, key=lambda r: as_float(r.get("F1_SA"), -1))
        stats.append(
            {
                "dataset": dataset,
                "method_family": family,
                "runs": len(rows),
                "best_F1": round(max(f1s), 3),
                "mean_F1": round(sum(f1s) / len(f1s), 3),
                "best_Acc": round(max(accs), 3) if accs else "",
                "best_run": best.get("run_name", ""),
                "best_epoch": best.get("epoch", ""),
                "best_seed": best.get("seed", ""),
            }
        )
    return sorted(stats, key=lambda r: (r["dataset"], -as_float(r["best_F1"], -1)))


def best_by_dataset(best_rows):
    by_dataset = defaultdict(list)
    for row in best_rows:
        if as_float(row.get("F1_SA")) is not None:
            by_dataset[row.get("dataset", "")].append(row)
    return {dataset: max(rows, key=lambda r: as_float(r.get("F1_SA"), -1)) for dataset, rows in by_dataset.items()}


def build_recommendations(best_rows, stats_rows):
    recs = []
    families = {row.get("method_family") for row in best_rows}
    complete = [r for r in best_rows if as_float(r.get("F1_SA")) is not None]
    if not complete:
        return ["No completed scored runs were found. First verify that `preds_for_eval*.text` files contain score JSON on the first line."]

    seeds_by_signature = defaultdict(set)
    for row in complete:
        seeds_by_signature[row.get("method_signature", "")].add(str(row.get("seed", "")))
    low_seed = [sig for sig, seeds in seeds_by_signature.items() if len({s for s in seeds if s}) < 3]
    if low_seed:
        recs.append("Repeat the strongest settings with at least 3 seeds before making a paper-level claim; several method signatures currently have fewer than 3 seeds.")

    if any(f and f.startswith("omni_caption") for f in families):
        recs.append("For the Omni-caption route, run audio-only and video-only ablations with the same seed/window/LoRA settings to determine whether gains come from voice cues, visual cues, or their combination.")
        recs.append("Audit captions containing label-like words before final reporting; high `label_word_hits_for_review` may reflect useful observable words or may indicate label leakage.")
    else:
        recs.append("No Omni-caption run was detected. Run `run_meld_omni_caption_experiment.sh` or check that its `run_config.json` uses `data_source=omni_caption`.")

    if "av_prefix" in families or "av_prefix_text_guided" in families:
        recs.append("Compare Omni-caption runs against AV-prefix runs under the same base LLM, window, seed, LoRA rank, and learning rate; otherwise improvements may be configuration effects rather than modality-use effects.")
    else:
        recs.append("Add an embedding/prefix baseline using the same MELD split and Qwen2.5 setup, so the caption route is compared against a continuous-feature multimodal route.")

    if "av_prefix_text_guided" not in families:
        recs.append("If the AV-prefix baseline is competitive, add a text-guided fusion run to test whether target text can gate unreliable audio/video cues.")

    return recs


def write_markdown(path, all_rows, best_rows, stats_rows, top_k):
    path.parent.mkdir(parents=True, exist_ok=True)
    complete_best = [r for r in best_rows if as_float(r.get("F1_SA")) is not None]
    top_rows = sort_rows(complete_best)[:top_k]
    best_dataset = best_by_dataset(complete_best)
    recs = build_recommendations(complete_best, stats_rows)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Experiment Analysis Report",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "",
        f"- Epoch-level scored rows: {len([r for r in all_rows if as_float(r.get('F1_SA')) is not None])}",
        f"- Completed run directories: {len(complete_best)}",
        "- Aggregation: each run is represented by its best epoch F1 unless otherwise noted.",
        "- Caution: this script summarizes logs; it does not establish statistical significance.",
        "",
    ]

    if best_dataset:
        lines.extend(["## Best By Dataset", ""])
        lines.append(markdown_table(list(best_dataset.values()), ["dataset", "F1_SA", "Acc_SA", "method_family", "epoch", "seed", "run_name"]))
        lines.append("")

    lines.extend(["## Top Runs", ""])
    if top_rows:
        lines.append(markdown_table(top_rows, ["dataset", "F1_SA", "Acc_SA", "method_family", "model_label", "epoch", "seed", "learning_rate", "run_name"]))
    else:
        lines.append("No scored runs found.")
    lines.append("")

    lines.extend(["## Method Family Summary", ""])
    if stats_rows:
        lines.append(markdown_table(stats_rows, ["dataset", "method_family", "runs", "best_F1", "mean_F1", "best_Acc", "best_epoch", "best_seed", "best_run"]))
    else:
        lines.append("No method-family statistics available.")
    lines.append("")

    lines.extend(["## Omni Caption Runs", ""])
    omni_rows = [r for r in complete_best if str(r.get("method_family", "")).startswith("omni_caption")]
    if omni_rows:
        lines.append(markdown_table(sort_rows(omni_rows), ["F1_SA", "Acc_SA", "method_family", "use_audio_caption", "use_video_caption", "video_fps", "video_max_pixels", "epoch", "seed", "run_name"]))
    else:
        lines.append("No Omni-caption scored run found.")
    lines.append("")

    lines.extend(["## Suggested Next Experiments", ""])
    for idx, rec in enumerate(recs, 1):
        lines.append(f"{idx}. {rec}")
    lines.append("")

    lines.extend(
        [
            "## Reading Guide",
            "",
            "- `best_by_run.csv`: one row per run directory, choosing the epoch with highest F1.",
            "- `all_epoch_results.csv`: one row per `preds_for_eval*.text` file.",
            "- `method_family_summary.csv`: grouped view by dataset and method family.",
            "- Check runs with the same seed and hyperparameters before interpreting method differences.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze Our-MSER experiment results and write a Markdown report.")
    parser.add_argument("--experiments_root", default="../experiments")
    parser.add_argument("--out_dir", default="../experiments/analysis")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--print_report", action="store_true")
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
    stats_rows = group_stats(best_rows)

    all_csv = out_dir / "all_epoch_results.csv"
    best_csv = out_dir / "best_by_run.csv"
    stats_csv = out_dir / "method_family_summary.csv"
    md_path = out_dir / "experiment_analysis.md"

    extra_fields = [
        "model_name",
        "data_source",
        "prompt_style",
        "caption_path",
        "caption_limit",
        "video_fps",
        "video_max_pixels",
        "data_dir",
        "pred_file",
        "status",
        "mode",
        "trainable_params",
        "total_params",
    ]
    fieldnames = CORE_FIELDS + [field for field in extra_fields if field not in CORE_FIELDS]
    write_csv(all_csv, all_rows, fieldnames)
    write_csv(best_csv, best_rows, fieldnames)
    write_csv(stats_csv, stats_rows, ["dataset", "method_family", "runs", "best_F1", "mean_F1", "best_Acc", "best_epoch", "best_seed", "best_run"])
    write_markdown(md_path, all_rows, best_rows, stats_rows, args.top_k)

    print(f"Experiments root: {experiments_root}")
    print(f"Run dirs found: {len(run_dirs)}")
    print(f"Epoch rows: {len(all_rows)}")
    print(f"Best rows: {len(best_rows)}")
    print(f"Wrote: {all_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {stats_csv}")
    print(f"Wrote: {md_path}")

    if args.print_report:
        print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
