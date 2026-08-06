import argparse
import json
import re
from collections import Counter
from pathlib import Path


LABEL_AND_INFERENCE_WORDS = [
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
    "emotion",
    "emotional",
    "mood",
    "demeanor",
    "attitude",
    "appears",
    "seems",
    "suggesting",
    "indicating",
    "indicated",
    "suggests",
    "indicates",
    "likely",
]
CHAT_MARKERS = ["system\n", "user\n", "assistant\n", "Human:", "User:", "Assistant:"]
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
FILLER_PATTERNS = [
    r"^the speaker(?:'s)?\s+",
    r"^the target speaker(?:'s)?\s+",
    r"^speaker(?:'s)?\s+",
    r"^the individual\s+",
    r"^the person\s+",
]
DROP_AUDIO_PHRASES = [
    "there is",
    "there are",
    "noticeable",
    "notable",
]


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


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sentence_chunks(text):
    text = normalize_space(text)
    text = re.sub(r"[.!?]+", ";", text)
    return [chunk.strip(" ,;") for chunk in text.split(";") if chunk.strip(" ,;")]


def split_clauses(text, mode):
    clauses = []
    text = re.sub(r"\bsteady and measured\b", "steady measured", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\bsteady and even\b", "steady even", str(text), flags=re.IGNORECASE)
    if mode == "audio":
        text = re.sub(r"\band\s+(?:the\s+)?pace\s+is\b", ", pace is", text, flags=re.IGNORECASE)
        text = re.sub(r"\band\s+(?:the\s+)?pitch\s+is\b", ", pitch is", text, flags=re.IGNORECASE)
    for chunk in sentence_chunks(text):
        if mode == "audio":
            splitter = r"\s*(?:,|;|\bwhile\b|\bwith\b)\s*"
        else:
            splitter = r"\s*(?:,|;|\bwhile\b)\s*"
        parts = re.split(splitter, chunk, flags=re.IGNORECASE)
        clauses.extend(part.strip(" ,;") for part in parts if part.strip(" ,;"))
    return clauses


def fix_mechanical_words(text):
    for bad, good in MECHANICAL_FIXES.items():
        text = re.sub(re.escape(bad), good, text, flags=re.IGNORECASE)
    return text


def remove_filler(text):
    text = normalize_space(text)
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:is|are|has|have)\b\s*", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:a|an|the)\b\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:and|with)\s+", "", text, flags=re.IGNORECASE)
    return normalize_space(text).strip(" ,;")


def remove_bad_words(text):
    for word in LABEL_AND_INFERENCE_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;")


def canonical_audio_phrase(phrase):
    phrase = remove_filler(fix_mechanical_words(phrase.lower()))
    phrase = remove_bad_words(phrase)
    phrase = phrase.replace("speaking pace", "pace")
    phrase = phrase.replace("pace of speech", "pace")
    phrase = phrase.replace("pace is", "pace")
    phrase = phrase.replace("voice is", "voice")
    phrase = phrase.replace("tone is", "tone")
    phrase = phrase.replace("pitch is", "pitch")
    phrase = phrase.replace("loudness is", "loudness")
    phrase = phrase.replace("voice quality is", "voice quality")
    for drop in DROP_AUDIO_PHRASES:
        phrase = phrase.replace(drop, "")
    phrase = normalize_space(phrase).strip(" ,;")
    if not phrase:
        return ""
    replacements = [
        (r"\bsteady and measured tone\b", "steady measured tone"),
        (r"\bsteady and even pitch\b", "steady even pitch"),
        (r"\bmoderate speaking rate\b", "moderate pace"),
        (r"\bmoderate pace\b", "moderate pace"),
        (r"\bsteady pace\b", "steady pace"),
        (r"\bpace is steady\b", "steady pace"),
        (r"\bpitch moderate\b", "moderate pitch"),
        (r"\blow and steady voice\b", "low steady voice"),
        (r"\btone steady measured\b", "steady measured tone"),
        (r"\btone steady\b", "steady tone"),
        (r"\btone even\b", "even tone"),
        (r"\btone varied\b", "varied tone"),
        (r"\bpace moderate\b", "moderate pace"),
        (r"\bpace steady\b", "steady pace"),
    ]
    for pattern, repl in replacements:
        phrase = re.sub(pattern, repl, phrase, flags=re.IGNORECASE)
    return normalize_space(phrase).strip(" ,;")


def canonical_video_phrase(phrase):
    phrase = remove_filler(fix_mechanical_words(phrase.lower()))
    phrase = remove_bad_words(phrase)
    phrase = phrase.replace("engaged in a conversation", "conversation")
    phrase = phrase.replace("engaging in conversation", "conversation")
    phrase = phrase.replace("conversing with", "talking with")
    phrase = phrase.replace(" and is ", " and ")
    phrase = phrase.replace("facial expressions", "facial movement")
    phrase = phrase.replace("facial expression", "facial movement")
    phrase = normalize_space(phrase).strip(" ,;")
    if not phrase:
        return ""
    phrase = re.sub(r"\bother other\b", "other", phrase)
    phrase = re.sub(r"\bhas has\b", "has", phrase)
    return normalize_space(phrase).strip(" ,;")


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def trim_cues(cues, max_cues, max_words_per_cue):
    trimmed = []
    for cue in cues:
        words = cue.split()
        if len(words) > max_words_per_cue:
            cue = " ".join(words[:max_words_per_cue]).strip(" ,;")
        if cue:
            trimmed.append(cue)
    return trimmed[:max_cues]


def audio_to_cues(text, max_cues, max_words_per_cue):
    raw_clauses = split_clauses(text, mode="audio")
    cues = []
    for clause in raw_clauses:
        cue = canonical_audio_phrase(clause)
        if cue:
            cues.append(cue)
    cues = dedupe_keep_order(cues)
    return trim_cues(cues, max_cues, max_words_per_cue)


def video_to_cues(text, max_cues, max_words_per_cue):
    raw_clauses = split_clauses(text, mode="video")
    cues = []
    for clause in raw_clauses:
        cue = canonical_video_phrase(clause)
        if cue:
            cues.append(cue)
    cues = dedupe_keep_order(cues)
    return trim_cues(cues, max_cues, max_words_per_cue)


def format_cues(prefix, cues):
    if not cues:
        return f"{prefix}: unavailable."
    return f"{prefix}: " + "; ".join(cues) + "."


def contains_any_word(text, words):
    text = str(text).lower()
    return [word for word in words if re.search(rf"\b{re.escape(word)}\b", text)]


def has_prompt_residue(text):
    return any(marker in str(text or "") for marker in CHAT_MARKERS)


def convert_row(row, args):
    converted = dict(row)
    source_audio = str(row.get("audio_caption", "") or "")
    source_video = str(row.get("video_caption", "") or "")
    audio_cues = audio_to_cues(source_audio, args.audio_max_cues, args.max_words_per_cue)
    video_cues = video_to_cues(source_video, args.video_max_cues, args.max_words_per_cue)

    if args.keep_original:
        converted["source_audio_caption"] = source_audio
        converted["source_video_caption"] = source_video
    converted["audio_cues"] = audio_cues
    converted["video_cues"] = video_cues
    converted["audio_caption"] = format_cues("Audio cues", audio_cues)
    converted["video_caption"] = format_cues("Video cues", video_cues)
    converted["cue_format_version"] = "meld_omni_cuelist_v1"
    converted["cue_format_flags"] = []
    if not audio_cues:
        converted["cue_format_flags"].append("empty_audio_cues")
    if not video_cues:
        converted["cue_format_flags"].append("empty_video_cues")
    if contains_any_word(converted["audio_caption"], LABEL_AND_INFERENCE_WORDS):
        converted["cue_format_flags"].append("audio_label_or_inference_terms_remaining")
    if contains_any_word(converted["video_caption"], LABEL_AND_INFERENCE_WORDS):
        converted["cue_format_flags"].append("video_label_or_inference_terms_remaining")
    if has_prompt_residue(converted["audio_caption"]) or has_prompt_residue(converted["video_caption"]):
        converted["cue_format_flags"].append("prompt_residue_remaining")
    return converted


def build_report(rows, out_path):
    flag_counts = Counter(flag for row in rows for flag in row.get("cue_format_flags", []))
    audio_cue_counts = [len(row.get("audio_cues", [])) for row in rows]
    video_cue_counts = [len(row.get("video_cues", [])) for row in rows]
    report = {
        "output_path": str(out_path),
        "records": len(rows),
        "status_counts": dict(Counter(row.get("status", "") for row in rows)),
        "rerun_recommended": sum(bool(row.get("rerun_recommended")) for row in rows),
        "flag_counts": dict(flag_counts),
        "post_convert_remaining": {
            "audio_label_or_inference_hits": sum(
                bool(contains_any_word(row.get("audio_caption", ""), LABEL_AND_INFERENCE_WORDS)) for row in rows
            ),
            "video_label_or_inference_hits": sum(
                bool(contains_any_word(row.get("video_caption", ""), LABEL_AND_INFERENCE_WORDS)) for row in rows
            ),
            "prompt_residue": sum(
                has_prompt_residue(row.get("audio_caption", "")) or has_prompt_residue(row.get("video_caption", ""))
                for row in rows
            ),
            "empty_audio_cues": sum(count == 0 for count in audio_cue_counts),
            "empty_video_cues": sum(count == 0 for count in video_cue_counts),
        },
        "cue_count_summary": {
            "audio_min": min(audio_cue_counts) if audio_cue_counts else 0,
            "audio_max": max(audio_cue_counts) if audio_cue_counts else 0,
            "audio_mean": round(sum(audio_cue_counts) / len(audio_cue_counts), 3) if audio_cue_counts else 0,
            "video_min": min(video_cue_counts) if video_cue_counts else 0,
            "video_max": max(video_cue_counts) if video_cue_counts else 0,
            "video_mean": round(sum(video_cue_counts) / len(video_cue_counts), 3) if video_cue_counts else 0,
        },
        "examples": [
            {
                "name": row.get("name"),
                "audio_caption": row.get("audio_caption"),
                "video_caption": row.get("video_caption"),
                "cue_format_flags": row.get("cue_format_flags", []),
            }
            for row in rows[:10]
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Convert cleaned MELD Omni captions into a unified cue-list format.")
    parser.add_argument("--captions", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.clean.jsonl")
    parser.add_argument("--out", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.cuelist.jsonl")
    parser.add_argument("--report_out", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.cuelist_report.json")
    parser.add_argument("--audio_max_cues", type=int, default=5)
    parser.add_argument("--video_max_cues", type=int, default=5)
    parser.add_argument("--max_words_per_cue", type=int, default=10)
    parser.add_argument("--keep_original", default="True")
    args = parser.parse_args()

    args.keep_original = str(args.keep_original).lower() == "true"
    rows = [convert_row(row, args) for row in read_jsonl(args.captions)]
    write_jsonl(args.out, rows)
    report = build_report(rows, args.out)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
