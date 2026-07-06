import argparse
import json
import os
import re
from collections import Counter, defaultdict


LABELS = {
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
}

PROMPT_MARKERS = [
    "Now you are",
    "The following conversation",
    "Please select",
    "Respond with",
    "Speaker_",
    "###",
    "Target speech characteristics",
    "from <",
]


def load_prediction_items(path):
    text = open(path, "r", encoding="utf-8").read()

    stripped = text.strip()
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if isinstance(payload, list):
            return payload

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            required = {"input", "output", "target"}
            if required.issubset(payload[0].keys()):
                return payload

    raise ValueError(f"Could not find prediction JSON list in {path}")


def match_text(text, labels):
    if text is None:
        return []
    lower_text = text.lower()
    matches = []
    for idx in range(len(lower_text)):
        cache = []
        for span_length in range(1, 12):
            span = lower_text[idx: idx + span_length]
            if span in labels:
                cache.append(span)
        if cache:
            matches.append(cache[-1])
    return matches


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def optimize_output(output, labels):
    output = (output or "").lower().strip()
    return min(labels, key=lambda label: edit_distance(output, label))


def normalize_clean_output(output):
    output = (output or "").strip().lower()
    output = re.sub(r"^[\s\"'`<\[]+", "", output)
    output = re.sub(r"[\s\"'`>,.;:!\]\)]+$", "", output)
    return output


def first_word(output):
    output = normalize_clean_output(output)
    match = re.match(r"[a-zA-Z_+-]+", output)
    return match.group(0).lower() if match else ""


def has_prompt_residue(output):
    return any(marker in (output or "") for marker in PROMPT_MARKERS)


def choose_prediction(output, labels):
    matches = match_text(output, labels)
    if matches:
        return matches[0], "matched"
    return optimize_output(output, labels), "forced"


def shorten(text, limit=220):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def add_example(bucket, item, pred, reason, max_examples):
    if len(bucket) >= max_examples:
        return
    bucket.append({
        "index": item.get("index"),
        "target": item.get("target"),
        "pred": pred,
        "reason": reason,
        "output": shorten(item.get("output")),
        "input_tail": shorten(item.get("input", "")[-350:]),
    })


def analyze(items, dataset, max_examples):
    labels = LABELS[dataset]
    label_set = set(labels)

    counts = Counter()
    pred_counts = Counter()
    target_counts = Counter()
    wrong_counts = Counter()
    examples = defaultdict(list)

    y_true = []
    y_pred = []

    for item in items:
        output = item.get("output") or ""
        target = (item.get("target") or "").lower().strip()
        clean = normalize_clean_output(output)
        first = first_word(output)
        matches = match_text(output, labels)
        pred, parse_method = choose_prediction(output, labels)

        counts["total"] += 1
        target_counts[target] += 1
        pred_counts[pred] += 1
        y_true.append(target)
        y_pred.append(pred)

        if not output.strip():
            counts["empty_output"] += 1
            add_example(examples["empty_output"], item, pred, "empty output", max_examples)
        if clean in label_set:
            counts["exact_label_output"] += 1
        else:
            add_example(examples["not_exact_label"], item, pred, "output is not exactly one label", max_examples)
        if first in label_set:
            counts["first_word_is_label"] += 1
        else:
            counts["first_word_not_label"] += 1
        if parse_method == "forced":
            counts["forced_by_edit_distance"] += 1
            add_example(examples["forced_by_edit_distance"], item, pred, "no label substring matched", max_examples)
        if len(set(matches)) > 1:
            counts["multiple_distinct_labels_in_output"] += 1
            add_example(examples["multiple_labels"], item, pred, f"labels={sorted(set(matches))}", max_examples)
        if has_prompt_residue(output):
            counts["prompt_residue_suspected"] += 1
            add_example(examples["prompt_residue"], item, pred, "prompt marker appears in output", max_examples)
        if len(output.split()) > 5 or len(output) > 40:
            counts["long_output"] += 1
            add_example(examples["long_output"], item, pred, "more than 5 words or 40 chars", max_examples)
        if pred != target:
            wrong_counts[(target, pred)] += 1
            add_example(examples["wrong_prediction"], item, pred, "pred != target", max_examples)

    metrics = {}
    try:
        from sklearn.metrics import accuracy_score, f1_score
        metrics["accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 3)
        metrics["weighted_f1"] = round(f1_score(y_true, y_pred, labels=labels, average="weighted") * 100, 3)
    except Exception as exc:
        metrics["sklearn_error"] = str(exc)

    return {
        "dataset": dataset,
        "counts": dict(counts),
        "metrics_recomputed_from_parsed_outputs": metrics,
        "target_distribution": dict(target_counts),
        "prediction_distribution": dict(pred_counts),
        "top_wrong_pairs": [
            {"target": target, "pred": pred, "count": count}
            for (target, pred), count in wrong_counts.most_common(20)
        ],
        "examples": dict(examples),
    }


def resolve_prediction_file(path):
    if os.path.isfile(path):
        return path
    candidates = []
    for name in os.listdir(path):
        if name.startswith("preds_for_eval") and name.endswith(".text"):
            full_path = os.path.join(path, name)
            epoch_match = re.search(r"preds_for_eval_(\d+)\.text$", name)
            epoch = int(epoch_match.group(1)) if epoch_match else -1
            candidates.append((epoch, os.path.getmtime(full_path), full_path))
    if not candidates:
        raise FileNotFoundError(f"No preds_for_eval*.text files found in {path}")
    candidates.sort()
    return candidates[-1][2]


def print_summary(report, pred_file):
    counts = Counter(report["counts"])
    total = counts["total"] or 1

    def pct(name):
        return f"{counts[name]} ({counts[name] / total * 100:.2f}%)"

    print(f"Prediction file: {pred_file}")
    print(f"Dataset: {report['dataset']}")
    print(f"Total: {counts['total']}")
    print(f"Exact one-label outputs: {pct('exact_label_output')}")
    print(f"First word is a label: {pct('first_word_is_label')}")
    print(f"Forced by edit distance: {pct('forced_by_edit_distance')}")
    print(f"Multiple labels in output: {pct('multiple_distinct_labels_in_output')}")
    print(f"Prompt residue suspected: {pct('prompt_residue_suspected')}")
    print(f"Long outputs: {pct('long_output')}")
    print(f"Empty outputs: {pct('empty_output')}")
    print(f"Recomputed metrics: {report['metrics_recomputed_from_parsed_outputs']}")
    print(f"Prediction distribution: {report['prediction_distribution']}")
    print(f"Target distribution: {report['target_distribution']}")
    if report["top_wrong_pairs"]:
        print("Top wrong pairs:")
        for pair in report["top_wrong_pairs"][:10]:
            print(f"  {pair['target']} -> {pair['pred']}: {pair['count']}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SpeechCueLLM/Our-MSER prediction files.")
    parser.add_argument("--dataset", choices=sorted(LABELS), required=True)
    parser.add_argument("--pred_file", required=True, help="A preds_for_eval*.text file or an output directory.")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    parser.add_argument("--max_examples", type=int, default=8)
    args = parser.parse_args()

    pred_file = resolve_prediction_file(args.pred_file)
    items = load_prediction_items(pred_file)
    report = analyze(items, args.dataset, args.max_examples)
    report["prediction_file"] = pred_file
    print_summary(report, pred_file)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved diagnostic report: {args.out}")


if __name__ == "__main__":
    main()
