"""Create a Markdown report from PRC-Emo run status and evaluation JSON files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="runs")
    parser.add_argument("--status", default="runs/run_status.tsv")
    parser.add_argument("--output", default="runs/RESULTS.md")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    status_path = Path(args.status)
    output_path = Path(args.output)
    lines = [
        "# PRC-Emo reproduction results",
        "",
        "该报告由 `scripts/run_reproduction_3gpu.sh` 自动生成。完整失败样本包括原始句子、Prompt、金标准、预测标签和模型原始输出。",
        "",
        "## Run status",
        "",
        "| Phase | Dataset | Seed | GPU | Exit code | Log | Output |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    statuses = []
    if status_path.exists():
        for raw in status_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.startswith("phase\t"):
                continue
            fields = raw.split("\t")
            if len(fields) >= 7:
                statuses.append(fields[:7])
                lines.append(
                    f"| {esc(fields[0])} | {esc(fields[1])} | {esc(fields[2])} | "
                    f"{esc(fields[3])} | {esc(fields[4])} | `{esc(fields[5])}` | `{esc(fields[6])}` |"
                )
    else:
        lines.append("| No status file found | | | | | | |")

    result_files = sorted(run_root.rglob("result_*.json")) if run_root.exists() else []
    lines.extend(["", "## Evaluation summary", "", "| Result file | Metrics | Failures |", "|---|---|---:|"])
    loaded = []
    for path in result_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            lines.append(f"| `{esc(path)}` | read error: {esc(exc)} | |")
            continue
        metrics = data.get("metrics", {})
        failures = data.get("failure_samples", [])
        loaded.append((path, data, failures))
        metric_text = "; ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items())
        lines.append(f"| `{esc(path)}` | {esc(metric_text)} | {len(failures)} |")

    lines.extend(["", "## Failure details", ""])
    if not loaded or not any(failures for _, _, failures in loaded):
        lines.append("没有发现评估失败样本，或者评估结果尚未生成。")
    else:
        for path, data, failures in loaded:
            if not failures:
                continue
            lines.extend([f"### `{path}`", ""])
            for failure in failures:
                index = esc(failure.get("index", ""))
                gold = esc(failure.get("gold_label", ""))
                pred = esc(failure.get("model_prediction", ""))
                utterance = esc(failure.get("utterance", ""))
                prompt = esc(failure.get("prompt", ""))
                raw_output = esc(failure.get("model_raw_output", ""))
                messages = esc(json.dumps(failure.get("messages", []), ensure_ascii=False, indent=2))
                lines.extend([
                    f"<details><summary>sample index={index}; gold={gold}; prediction={pred}</summary>",
                    "",
                    f"**Utterance**: {utterance}",
                    "",
                    f"**Gold label**: `{gold}`  ",
                    f"**Model prediction**: `{pred}`",
                    "",
                    "**Prompt**",
                    "",
                    f"<pre>{prompt}</pre>",
                    "",
                    "**Raw model output**",
                    "",
                    f"<pre>{raw_output}</pre>",
                    "",
                    "**Messages JSON**",
                    "",
                    f"<pre>{messages}</pre>",
                    "",
                    "</details>",
                    "",
                ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
