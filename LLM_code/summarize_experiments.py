import argparse
import csv
import json
import os
import re
from pathlib import Path


PRED_RE = re.compile(r"preds_for_eval(?:_(\d+))?\.text$")


def load_json(path):
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_score(pred_path):
    try:
        with pred_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            return {}
        return json.loads(first_line)
    except Exception:
        return {}


def parse_epoch(pred_path):
    match = PRED_RE.match(pred_path.name)
    if not match:
        return None
    if match.group(1) is None:
        return None
    return int(match.group(1))


def infer_from_path(run_dir, experiments_root):
    try:
        rel_parts = run_dir.relative_to(experiments_root).parts
    except ValueError:
        rel_parts = run_dir.parts

    inferred = {}
    if len(rel_parts) >= 1:
        inferred["model_label"] = rel_parts[0]
    if len(rel_parts) >= 2:
        inferred["experiment_setting"] = rel_parts[1]
    if len(rel_parts) >= 3:
        inferred["dataset_from_path"] = rel_parts[2]
    if len(rel_parts) >= 4:
        inferred["window_from_path"] = rel_parts[3]
    inferred["run_name"] = run_dir.name
    return inferred


def extract_run_flags(run_name):
    flags = {
        "uses_speech": "_speech" in run_name,
        "uses_persona": "_persona" in run_name,
        "prompt_style": "qwen_chat" if "qwen_chat" in run_name else ("legacy" if "legacy" in run_name else ""),
        "data_format": "prc" if "_prc" in run_name else ("speechcue" if "_speechcue" in run_name else ""),
    }
    for key, pattern in [
        ("lr_from_name", r"LR_([^_]+)"),
        ("batch_from_name", r"BS_([^_]+)"),
        ("data_percent_from_name", r"per_([^_]+)"),
    ]:
        match = re.search(pattern, run_name)
        flags[key] = match.group(1) if match else ""
    return flags


def find_run_dirs(experiments_root):
    run_dirs = []
    for root, _dirs, files in os.walk(experiments_root):
        if any(PRED_RE.match(name) for name in files):
            run_dirs.append(Path(root))
    return sorted(run_dirs)


def summarize_run(run_dir, experiments_root):
    pred_files = sorted(
        [p for p in run_dir.iterdir() if p.is_file() and PRED_RE.match(p.name)],
        key=lambda p: (-1 if parse_epoch(p) is None else parse_epoch(p), p.name),
    )
    rows = []
    model_args = load_json(run_dir / "model_args.json")
    model_params = load_json(run_dir / "model_params.json")
    inferred = infer_from_path(run_dir, experiments_root)
    flags = extract_run_flags(run_dir.name)

    best = None
    for pred_path in pred_files:
        score = parse_score(pred_path)
        epoch = parse_epoch(pred_path)
        row = {
            "run_dir": str(run_dir),
            "run_name": inferred.get("run_name", ""),
            "pred_file": pred_path.name,
            "epoch": "" if epoch is None else epoch,
            "Acc_SA": score.get("Acc_SA", ""),
            "F1_SA": score.get("F1_SA", ""),
            "mode": score.get("mode", ""),
            "dataset": model_args.get("dataset", inferred.get("dataset_from_path", "")),
            "model_label": inferred.get("model_label", ""),
            "experiment_setting": inferred.get("experiment_setting", ""),
            "window": inferred.get("window_from_path", ""),
            "model_path": model_args.get("model_name_or_path", ""),
            "data_dir": model_args.get("data_dir", ""),
            "batch_size": model_args.get("batch_size", flags.get("batch_from_name", "")),
            "gradient_accumulation_steps": model_args.get("gradient_accumulation_steps", ""),
            "max_length": model_args.get("max_length", ""),
            "learning_rate": model_args.get("learning_rate", flags.get("lr_from_name", "")),
            "num_train_epochs": model_args.get("num_train_epochs", ""),
            "seed": model_args.get("seed", ""),
            "lora": model_args.get("lora", ""),
            "lora_dim": model_args.get("lora_dim", ""),
            "lora_alpha": model_args.get("lora_alpha", ""),
            "lora_dropout": model_args.get("lora_dropout", ""),
            "lora_module_name": model_args.get("lora_module_name", ""),
            "uses_speech": flags["uses_speech"],
            "uses_persona": flags["uses_persona"],
            "prompt_style": flags["prompt_style"],
            "data_format": flags["data_format"],
            "trainable_params": model_params.get("Trainable", ""),
            "total_params": model_params.get("Total", ""),
            "status": "has_score" if score else "no_score",
        }
        rows.append(row)
        if score and isinstance(score.get("F1_SA"), (int, float)):
            if best is None or score["F1_SA"] > best["F1_SA"]:
                best = row

    if not rows:
        return [], None
    return rows, best


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize Our-MSER experiment outputs.")
    parser.add_argument("--experiments_root", default="../experiments", help="Root directory containing experiment runs.")
    parser.add_argument("--out_csv", default="../experiments/experiment_summary.csv", help="CSV path for all epoch-level rows.")
    parser.add_argument("--best_csv", default="../experiments/experiment_best_summary.csv", help="CSV path for one best row per run.")
    parser.add_argument("--print_best", action="store_true", help="Print best rows to stdout.")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    run_dirs = find_run_dirs(experiments_root)

    all_rows = []
    best_rows = []
    for run_dir in run_dirs:
        rows, best = summarize_run(run_dir, experiments_root)
        all_rows.extend(rows)
        if best:
            best_rows.append(best)

    fieldnames = [
        "dataset", "model_label", "experiment_setting", "run_name", "epoch", "Acc_SA", "F1_SA", "mode",
        "uses_speech", "uses_persona", "prompt_style", "data_format", "batch_size", "gradient_accumulation_steps",
        "max_length", "learning_rate", "num_train_epochs", "seed", "lora", "lora_dim", "lora_alpha",
        "lora_dropout", "lora_module_name", "trainable_params", "total_params", "window", "model_path",
        "data_dir", "pred_file", "status", "run_dir",
    ]

    write_csv(Path(args.out_csv), all_rows, fieldnames)
    write_csv(Path(args.best_csv), best_rows, fieldnames)

    print(f"Found runs: {len(run_dirs)}")
    print(f"Epoch rows: {len(all_rows)}")
    print(f"Best rows: {len(best_rows)}")
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.best_csv}")

    if args.print_best:
        for row in sorted(best_rows, key=lambda r: (str(r["dataset"]), -float(r["F1_SA"] or 0))):
            print(
                f"{row['dataset']}\t{row['model_label']}\tF1={row['F1_SA']}\t"
                f"Acc={row['Acc_SA']}\tepoch={row['epoch']}\t{row['run_name']}"
            )


if __name__ == "__main__":
    main()
