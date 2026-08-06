import argparse
import json
import re
from collections import Counter
from pathlib import Path


LABEL_WORDS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
INFERENCE_WORDS = [
    "emotion",
    "emotional",
    "mood",
    "affect",
    "demeanor",
    "attitude",
    "suggesting",
    "suggests",
    "indicating",
    "indicates",
    "indicated",
    "implying",
    "implies",
    "reflecting",
    "reflects",
    "calm",
    "happy",
    "sad",
    "angry",
    "anger",
    "annoyed",
    "annoyance",
    "frustrated",
    "frustration",
    "upset",
    "surprised",
    "surprise",
    "fearful",
    "scared",
    "nervous",
    "anxious",
    "disgusted",
    "concerned",
    "confused",
    "pleading",
    "soothing",
    "serious",
    "uncertain",
    "agitation",
    "agitated",
    "confidence",
    "confident",
    "positive",
    "playful",
    "amused",
    "light-hearted",
]
VIDEO_INFERENCE_WORDS = [
    "appears",
    "seems",
    "suggesting",
    "indicating",
    "indicates",
    "indicated",
    "suggests",
    "likely",
]
CHAT_MARKERS = [
    "Human:",
    "User:",
    "Assistant:",
    "\nsystem\n",
    "\nuser\n",
    "\nassistant\n",
    "system\n",
    "user\n",
    "assistant\n",
]
STOP_LAST_WORDS = {
    "and",
    "or",
    "with",
    "in",
    "on",
    "at",
    "of",
    "the",
    "a",
    "an",
    "to",
    "for",
    "by",
    "as",
    "while",
    "that",
    "who",
    "has",
    "is",
    "are",
}
MECHANICAL_FIXES = {
    "gesturinguringing": "gesturing",
    "gesturinguring": "gesturing",
    "gesturinging": "gesturing",
    "holdinging": "holding",
    "conversinging": "conversing",
    "talkinging": "talking",
    "lookinging": "looking",
    "smilinging": "smiling",
    "noddinging": "nodding",
    "laughinging": "laughing",
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_by_name(rows):
    latest = {}
    for row in rows:
        name = row.get("name")
        if name:
            latest[name] = row
    return latest




def has_prompt_residue(text):
    text = str(text)
    return any(marker in text for marker in CHAT_MARKERS)


def extract_assistant_answer(text):
    text = str(text or "")
    lowered = text.lower()
    assistant_marker = "assistant\n"
    if assistant_marker in lowered:
        start = lowered.find(assistant_marker) + len(assistant_marker)
        text = text[start:]
    for marker in ["Human:", "User:", "Assistant:", "\nsystem\n", "\nuser\n", "\nassistant\n"]:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    return text


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def remove_parenthetical_empty(text):
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\bis\s*,", "is even,", text, flags=re.IGNORECASE)
    text = re.sub(r"\bare\s*,", "are visible,", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(or|and)\s*([,.!?])", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s*([,.!?])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban\s*([,.!?])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\btone is\s+and\b", "tone is varied and", text, flags=re.IGNORECASE)
    text = re.sub(r"\btone is\s*\." , "tone is varied.", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text


def strip_inference_clauses(text):
    patterns = [
        r"\s*,?\s*\b(?:suggesting|suggests|indicating|indicates|indicated|implying|implies|reflecting|reflects)\b[^.!?]*",
        r"\s*,?\s*\bwith\s+(?:a\s+)?(?:calm|happy|sad|angry|annoyed|frustrated|surprised|fearful|nervous|confident|positive|serious|uncertain)[^.!?]*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def strip_absent_audio_clauses(text):
    patterns = [
        r"\s*,?\s*with no (?:noticeable|notable|audible)?\s*[^.!?]*",
        r"\s*,?\s*and no (?:noticeable|notable|audible)?\s*[^.!?]*",
        r"\s*There (?:is|are) no (?:noticeable|notable|audible)?\s*[^.!?]*[.!?]?",
        r"\s*No (?:noticeable|notable|audible)?\s*[^.!?]*[.!?]?",
        r"\s*without (?:noticeable|notable|audible)?\s*[^.!?]*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def remove_audio_label_terms(text):
    replacements = [
        (r"\bneutral tone\b", "even tone"),
        (r"\bneutral voice\b", "even voice"),
        (r"\bcalm demeanor\b", "steady delivery"),
        (r"\bcalm tone\b", "steady tone"),
        (r"\bserious tone\b", "lower tone"),
        (r"\buncertain tone\b", "uneven tone"),
        (r"\bpleading tone\b", "strained tone"),
        (r"\bsoothing tone\b", "soft tone"),
        (r"\blight-hearted mood\b", "lighter delivery"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    banned = sorted(set(LABEL_WORDS + INFERENCE_WORDS), key=len, reverse=True)
    for word in banned:
        text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
    return remove_parenthetical_empty(text)


def remove_video_inference_terms(text):
    text = re.sub(
        r"\b(?:appears to be|seems to be|is|are)?\s*expressing\b[^,.!?]*(?:,?\s*(?:indicated|shown)\s+by|through)\s+",
        "shows ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bappears to be\b", "is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bseems to be\b", "is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bappears\b", "is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bseems\b", "is", text, flags=re.IGNORECASE)
    text = strip_inference_clauses(text)
    text = re.sub(r"\b(indicating|indicates|suggesting|suggests|likely)\b[^.!?]*", "", text, flags=re.IGNORECASE)
    banned = sorted(set(LABEL_WORDS + VIDEO_INFERENCE_WORDS + INFERENCE_WORDS), key=len, reverse=True)
    for word in banned:
        text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
    return remove_parenthetical_empty(text)


def fix_mechanical_words(text):
    lowered = text
    for bad, good in MECHANICAL_FIXES.items():
        lowered = re.sub(re.escape(bad), good, lowered, flags=re.IGNORECASE)
    return lowered


def ensure_sentence(text):
    text = normalize_space(text)
    if not text:
        return text
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)
    if words and words[-1].lower() in STOP_LAST_WORDS:
        parts = re.split(r"[.!?]", text)
        parts = [part.strip(" ,;") for part in parts if part.strip(" ,;")]
        if len(parts) > 1:
            text = parts[0]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def truncate_words(text, max_words):
    tokens = text.split()
    if len(tokens) <= max_words:
        return text
    return " ".join(tokens[:max_words]).rstrip(" ,;") + "."


def contains_any_word(text, words):
    text = str(text).lower()
    return [word for word in words if re.search(rf"\b{re.escape(word)}\b", text)]


def maybe_truncated(text):
    text = str(text or "").strip()
    if not text:
        return False
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)
    last_word = words[-1].lower() if words else ""
    return text[-1] not in ".!?" or last_word in STOP_LAST_WORDS


def clean_audio(text, max_words):
    flags = []
    original = str(text or "")
    if has_prompt_residue(original):
        flags.append("prompt_residue_removed")
    cleaned = extract_assistant_answer(original)
    cleaned = strip_inference_clauses(cleaned)
    before_absent = cleaned
    cleaned = strip_absent_audio_clauses(cleaned)
    if cleaned != before_absent:
        flags.append("absent_audio_clauses_removed")
    if contains_any_word(cleaned, LABEL_WORDS + INFERENCE_WORDS):
        flags.append("label_or_inference_terms_removed")
    cleaned = remove_audio_label_terms(cleaned)
    cleaned = fix_mechanical_words(cleaned)
    cleaned = truncate_words(ensure_sentence(cleaned), max_words)
    if not cleaned:
        flags.append("empty_after_cleaning")
    if contains_any_word(cleaned, LABEL_WORDS + INFERENCE_WORDS):
        flags.append("label_or_inference_terms_remaining")
    if has_prompt_residue(cleaned):
        flags.append("prompt_residue_remaining")
    return cleaned, flags


def clean_video(text, max_words):
    flags = []
    original = str(text or "")
    if has_prompt_residue(original):
        flags.append("prompt_residue_removed")
    if maybe_truncated(original):
        flags.append("possibly_truncated_original")
    if any(bad in original.lower() for bad in MECHANICAL_FIXES):
        flags.append("mechanical_words_fixed")
    cleaned = extract_assistant_answer(original)
    cleaned = fix_mechanical_words(cleaned)
    if contains_any_word(cleaned, LABEL_WORDS + INFERENCE_WORDS + VIDEO_INFERENCE_WORDS):
        flags.append("label_or_inference_terms_removed")
    cleaned = remove_video_inference_terms(cleaned)
    cleaned = fix_mechanical_words(cleaned)
    cleaned = truncate_words(ensure_sentence(cleaned), max_words)
    if not cleaned:
        flags.append("empty_after_cleaning")
    if maybe_truncated(cleaned):
        flags.append("possibly_truncated_after_cleaning")
    if contains_any_word(cleaned, LABEL_WORDS + INFERENCE_WORDS + VIDEO_INFERENCE_WORDS):
        flags.append("label_or_inference_terms_remaining")
    if has_prompt_residue(cleaned):
        flags.append("prompt_residue_remaining")
    return cleaned, flags


def rerun_reasons(row, audio_flags, video_flags):
    reasons = []
    if row.get("status") != "ok":
        reasons.append("status_not_ok")
    for prefix, flags in [("audio", audio_flags), ("video", video_flags)]:
        if "empty_after_cleaning" in flags:
            reasons.append(f"{prefix}_empty_after_cleaning")
        if "prompt_residue_removed" in flags:
            reasons.append(f"{prefix}_prompt_residue_in_original")
        if "prompt_residue_remaining" in flags:
            reasons.append(f"{prefix}_prompt_residue_remaining")
        if "label_or_inference_terms_remaining" in flags:
            reasons.append(f"{prefix}_label_or_inference_terms_remaining")
    if "possibly_truncated_original" in video_flags or "possibly_truncated_after_cleaning" in video_flags:
        reasons.append("video_possibly_truncated")
    return sorted(set(reasons))


def build_report(cleaned_rows, rerun_rows, output_path, rerun_path, manifest_rows=None):
    status_counts = Counter(row.get("status") for row in cleaned_rows)
    flag_counts = Counter()
    label_remaining = {"audio": 0, "video": 0}
    prompt_remaining = {"audio": 0, "video": 0}
    empty = {"audio": 0, "video": 0}
    for row in cleaned_rows:
        for flag in row.get("cleaning_flags", []):
            flag_counts[flag] += 1
        if contains_any_word(row.get("audio_caption", ""), LABEL_WORDS + INFERENCE_WORDS):
            label_remaining["audio"] += 1
        if contains_any_word(row.get("video_caption", ""), LABEL_WORDS + INFERENCE_WORDS + VIDEO_INFERENCE_WORDS):
            label_remaining["video"] += 1
        if has_prompt_residue(row.get("audio_caption", "")):
            prompt_remaining["audio"] += 1
        if has_prompt_residue(row.get("video_caption", "")):
            prompt_remaining["video"] += 1
        if not row.get("audio_caption", "").strip():
            empty["audio"] += 1
        if not row.get("video_caption", "").strip():
            empty["video"] += 1

    report = {
        "output_path": str(output_path),
        "rerun_path": str(rerun_path),
        "cleaned_records": len(cleaned_rows),
        "manifest_rows": len(manifest_rows) if manifest_rows is not None else None,
        "missing_from_captions": (
            len(set(row["name"] for row in manifest_rows) - set(row["name"] for row in cleaned_rows))
            if manifest_rows is not None
            else None
        ),
        "status_counts": dict(status_counts),
        "rerun_recommended": len(rerun_rows),
        "flag_counts": dict(flag_counts),
        "post_clean_remaining": {
            "label_or_inference_terms": label_remaining,
            "prompt_residue": prompt_remaining,
            "empty_caption": empty,
        },
        "rerun_reason_counts": dict(Counter(reason for row in rerun_rows for reason in row["rerun_reasons"])),
        "rerun_examples": rerun_rows[:20],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Clean MELD Omni captions and mark rows that should be regenerated.")
    parser.add_argument("--captions", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl")
    parser.add_argument("--manifest", default="../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl")
    parser.add_argument("--out", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.clean.jsonl")
    parser.add_argument("--rerun_out", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.rerun_needed.jsonl")
    parser.add_argument("--report_out", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.clean_report.json")
    parser.add_argument("--audio_max_words", type=int, default=32)
    parser.add_argument("--video_max_words", type=int, default=42)
    args = parser.parse_args()

    caption_rows = latest_by_name(read_jsonl(args.captions))
    manifest_path = Path(args.manifest)
    manifest_rows = read_jsonl(manifest_path) if manifest_path.is_file() else None
    ordered_names = [row["name"] for row in manifest_rows] if manifest_rows is not None else sorted(caption_rows)

    cleaned_rows = []
    rerun_rows = []
    for name in ordered_names:
        row = dict(caption_rows.get(name, {"name": name, "status": "missing"}))
        audio_caption, audio_flags = clean_audio(row.get("audio_caption", ""), args.audio_max_words)
        video_caption, video_flags = clean_video(row.get("video_caption", ""), args.video_max_words)
        flags = [f"audio:{flag}" for flag in audio_flags] + [f"video:{flag}" for flag in video_flags]
        reasons = rerun_reasons(row, audio_flags, video_flags)
        row["audio_caption"] = audio_caption
        row["video_caption"] = video_caption
        row["cleaning_version"] = "meld_omni_caption_clean_v1"
        row["cleaning_flags"] = flags
        row["rerun_recommended"] = bool(reasons)
        row["rerun_reasons"] = reasons
        cleaned_rows.append(row)
        if reasons:
            rerun_rows.append(
                {
                    "name": name,
                    "split": row.get("split"),
                    "status": row.get("status"),
                    "rerun_reasons": reasons,
                    "cleaning_flags": flags,
                    "audio_caption": audio_caption,
                    "video_caption": video_caption,
                    "error": row.get("error", ""),
                }
            )

    write_jsonl(args.out, cleaned_rows)
    write_jsonl(args.rerun_out, rerun_rows)
    report = build_report(cleaned_rows, rerun_rows, args.out, args.rerun_out, manifest_rows)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
