import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


REQUIRED_MARKERS = [
    "Available emotion labels",
    "Dialogue context",
    "Target utterance",
    "Question",
    "Answer with exactly one label",
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_chat_input(example_input, tokenizer):
    if isinstance(example_input, list):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                example_input,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered = []
        for message in example_input:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        rendered.append("assistant:")
        return "\n".join(rendered)
    return str(example_input)


def load_split(data_dir, split):
    path = Path(data_dir) / f"{split}.json"
    if not path.is_file():
        return []
    return read_jsonl(path)


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def target_text_from_prompt(text):
    marker = "Target utterance:"
    question = "\n\nQuestion:"
    if marker not in text:
        return ""
    part = text.split(marker, 1)[1]
    if question in part:
        part = part.split(question, 1)[0]
    return part.strip()


def inspect_with_side(rows, tokenizer, max_length, truncation_side):
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    limit = max_length - 1

    lengths = []
    truncated = []
    missing = {marker: [] for marker in REQUIRED_MARKERS}
    missing_target_text = []
    examples = []

    for idx, row in enumerate(rows):
        rendered = render_chat_input(row["input"], tokenizer)
        full_ids = tokenizer(rendered, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.append(len(full_ids))
        is_truncated = len(full_ids) > limit
        if is_truncated:
            truncated.append(idx)

        cut = tokenizer(
            rendered,
            max_length=limit,
            truncation=True,
            add_special_tokens=True,
        )
        decoded = tokenizer.decode(
            cut["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        missing_markers = []
        for marker in REQUIRED_MARKERS:
            if marker not in decoded:
                missing[marker].append(idx)
                missing_markers.append(marker)

        target_text = target_text_from_prompt(rendered)
        target_missing = bool(target_text and target_text not in decoded)
        if target_missing:
            missing_target_text.append(idx)

        if (missing_markers or target_missing) and len(examples) < 8:
            examples.append(
                {
                    "index": idx,
                    "path": row.get("path", ""),
                    "target": row.get("target", ""),
                    "full_tokens": len(full_ids),
                    "truncated": is_truncated,
                    "missing_markers": missing_markers,
                    "missing_target_text": target_missing,
                    "decoded_tail": decoded[-800:],
                }
            )

    tokenizer.truncation_side = old_side
    total = len(rows)
    return {
        "truncation_side": truncation_side,
        "total": total,
        "max_length": max_length,
        "input_token_limit_before_label": limit,
        "token_length": {
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
        },
        "truncated_count": len(truncated),
        "truncated_percent": round(len(truncated) * 100 / total, 3) if total else 0,
        "missing_marker_counts": {k: len(v) for k, v in missing.items()},
        "missing_marker_percent": {
            k: round(len(v) * 100 / total, 3) if total else 0 for k, v in missing.items()
        },
        "missing_target_text_count": len(missing_target_text),
        "missing_target_text_percent": round(len(missing_target_text) * 100 / total, 3) if total else 0,
        "problem_examples": examples,
    }


def inspect_split(rows, tokenizer, max_length, sides):
    return {
        side: inspect_with_side(rows, tokenizer, max_length=max_length, truncation_side=side)
        for side in sides
    }


def print_summary(report):
    print(f"Data dir: {report['data_dir']}")
    print(f"Model path: {report['model_path']}")
    print(f"Tokenizer default truncation_side: {report['tokenizer_default_truncation_side']}")
    print(f"Max length: {report['max_length']}")
    for split, split_report in report["splits"].items():
        print(f"\n[{split}]")
        for side, side_report in split_report.items():
            length = side_report["token_length"]
            print(
                f"  side={side} total={side_report['total']} "
                f"truncated={side_report['truncated_count']} "
                f"({side_report['truncated_percent']}%) "
                f"max={length['max']} p95={length['p95']} p99={length['p99']}"
            )
            markers = side_report["missing_marker_counts"]
            target_missing = side_report["missing_target_text_count"]
            print(f"    missing markers: {markers}; missing target text: {target_missing}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose whether prompt truncation removes task-critical text.")
    parser.add_argument("--data_dir", required=True, help="Processed data dir containing train/test/valid.json.")
    parser.add_argument("--model_path", required=True, help="Tokenizer/model path.")
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--splits", default="train,test,valid")
    parser.add_argument("--sides", default="right,left", help="Comma-separated truncation sides to compare.")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    report = {
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "max_length": args.max_length,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_default_truncation_side": tokenizer.truncation_side,
        "splits": {},
    }

    for split in splits:
        rows = load_split(args.data_dir, split)
        if rows:
            report["splits"][split] = inspect_split(rows, tokenizer, args.max_length, sides)

    print_summary(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()
