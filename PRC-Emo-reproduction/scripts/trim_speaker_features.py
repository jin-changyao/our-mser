"""Extract the first valid speaker-description Response from noisy feature files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RESPONSE_KEY = re.compile(r'"Response"\s*:\s*"', re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w'-]+\b")
POLLUTION_PATTERNS = {
    "human_rights_watch": re.compile(r"Human Rights Watch", re.IGNORECASE),
    "human_role": re.compile(r"\b(?:Human|Assistant)\s*:", re.IGNORECASE),
    "code_fence": re.compile(r"```(?:json|python|sql)?", re.IGNORECASE),
    "task_prompt": re.compile(
        r"\b(?:Question:|Multiple Choice|Essay Prompt|Fill-in-the-blank|Instruction:)\b",
        re.IGNORECASE,
    ),
    "unrelated_topic": re.compile(
        r"\b(?:Human[- ]robot|Human[- ]Animal Interaction|Jira|solar panels)\b",
        re.IGNORECASE,
    ),
}


def extract_first_response(raw: str) -> tuple[str | None, int]:
    """Return the first JSON Response value and the number of Response keys."""

    matches = list(RESPONSE_KEY.finditer(raw))
    if not matches:
        return None, 0

    quote_start = matches[0].end() - 1
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[quote_start:])
    except json.JSONDecodeError:
        # Be tolerant of malformed trailing JSON while still respecting escapes.
        chars: list[str] = []
        escaped = False
        for char in raw[quote_start + 1 :]:
            if escaped:
                chars.append(char)
                escaped = False
            elif char == "\\":
                chars.append(char)
                escaped = True
            elif char == '"':
                break
            else:
                chars.append(char)
        value = "".join(chars)

    return re.sub(r"\s+", " ", str(value)).strip(), len(matches)


def inspect_description(description: str | None) -> tuple[str, str | None]:
    if not description:
        return "no_response", None
    words = WORD_RE.findall(description)
    if len(words) < 5:
        return "too_short", f"only_{len(words)}_words"
    if len(description) > 800:
        return "too_long", f"{len(description)}_chars"
    for name, pattern in POLLUTION_PATTERNS.items():
        if pattern.search(description):
            return "first_response_polluted", name
    return "trimmed", None


def process_file(path: Path, output_dir: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    status_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    output: dict[str, list[str]] = {}

    for conversation_id, values in data.items():
        output_values: list[str] = []
        for index, raw_value in enumerate(values):
            raw = "" if raw_value is None else str(raw_value)
            description, response_count = extract_first_response(raw)
            status, reason = inspect_description(description)
            if status != "trimmed":
                description = ""
            status_counts[status] += 1
            if status != "trimmed" and len(examples.setdefault(status, [])) < 5:
                examples[status].append(
                    {
                        "conversation": conversation_id,
                        "index": index,
                        "response_count": response_count,
                        "reason": reason,
                        "raw_chars": len(raw),
                    }
                )
            output_values.append(description or "")
        output[conversation_id] = output_values

    output_path = output_dir / path.name.replace(
        "_spdescV6_Qwen2.5-7B-Instruct.json",
        "_spdescV6_trimmed_Qwen2.5-7B-Instruct.json",
    )
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "file": path.name,
        "output": output_path.name,
        "records": sum(len(values) for values in data.values()),
        "status_counts": dict(status_counts),
        "examples": examples,
    }


def write_report(report_path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Trimmed speaker feature report",
        "",
        "规则：提取第一个完整 `Response`，删除后续续写；异常样本输出为空字符串。",
        "",
        "| File | Records | Trimmed | No response | Too short | Too long | First response polluted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        counts = result["status_counts"]
        lines.append(
            f"| `{result['file']}` | {result['records']} | {counts.get('trimmed', 0)} | "
            f"{counts.get('no_response', 0)} | {counts.get('too_short', 0)} | "
            f"{counts.get('too_long', 0)} | {counts.get('first_response_polluted', 0)} |"
        )
    lines.extend(["", "## Examples", ""])
    for result in results:
        lines.append(f"### `{result['file']}`")
        lines.append("")
        for status, examples in result["examples"].items():
            lines.append(f"- `{status}`: `{json.dumps(examples, ensure_ascii=False)}`")
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*_spdescV6_Qwen2.5-7B-Instruct.json"))
    if len(files) != 6:
        raise SystemExit(f"Expected 6 speaker feature files, found {len(files)}")
    results = [process_file(path, args.output_dir) for path in files]
    write_report(args.report, results)
    print(f"Processed {len(files)} files")
    print(f"Report: {args.report}")
    for result in results:
        print(result["file"], result["status_counts"])


if __name__ == "__main__":
    main()
