import argparse
import json
from collections import defaultdict
from pathlib import Path


LABEL_TEXT = "neutral, surprise, fear, sadness, joy, disgust, anger"
LABEL_GUIDANCE = (
    "Important label distinctions:\n"
    "- joy: positive happiness or delight.\n"
    "- surprise: sudden unexpected reaction, not necessarily positive.\n"
    "- anger: anger, irritation, or hostility.\n"
    "- disgust: aversion, dislike, or contempt.\n"
    "- fear: worry, anxiety, or being scared.\n"
    "- sadness: sorrow, disappointment, or low mood.\n"
    "- neutral: no clear emotional intensity."
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_ok_captions(path):
    captions = {}
    for row in read_jsonl(path):
        name = row.get("name")
        if not name:
            continue
        if row.get("status") == "ok":
            captions[name] = row
    return captions


def normalize_split(split):
    return "valid" if split == "dev" else split


def speaker_name(row):
    speaker = str(row.get("speaker", "")).strip()
    return speaker if speaker else "Speaker"


def group_dialogs(rows):
    dialogs = defaultdict(list)
    for row in rows:
        key = (row["split"], int(row["dialogue_id"]))
        dialogs[key].append(row)
    for dialog_rows in dialogs.values():
        dialog_rows.sort(key=lambda item: int(item["utterance_id"]))
    return dialogs


def format_context_line(row, audio_caption="", include_audio=False):
    line = f'{speaker_name(row)}: "{row.get("text", "")}"'
    if include_audio and audio_caption:
        line += f" (Audio description: {audio_caption.strip()})"
    return line


def build_caption_parts(audio_caption, video_caption, caption_prompt_mode):
    if caption_prompt_mode == "context_audio_target_video":
        parts = []
        if video_caption:
            parts.append("Target video description:\n" + video_caption.strip())
        return parts

    if caption_prompt_mode == "unified_cues":
        cue_lines = []
        if audio_caption:
            cue_lines.append(audio_caption.strip())
        if video_caption:
            cue_lines.append(video_caption.strip())
        return ["Target observable multimodal cues:\n" + "\n".join(cue_lines)] if cue_lines else []

    parts = []
    if audio_caption:
        parts.append("Target audio description:\n" + audio_caption.strip())
    if video_caption:
        parts.append("Target video description:\n" + video_caption.strip())
    return parts


def build_qwen_chat_prompt(context_lines, target_utterance, audio_caption, video_caption, caption_prompt_mode):
    if caption_prompt_mode == "context_audio_target_video":
        system_msg = (
            "You are an expert in emotion recognition in conversation. "
            "Classify the emotion of the target utterance only. "
            "Use the dialogue text as the primary semantic evidence. "
            "Use audio descriptions attached to utterances in the dialogue window as auxiliary speech-delivery evidence, "
            "and use the target video description as auxiliary visible-action evidence. "
            "Do not treat multimodal descriptions as emotion labels. "
            "Do not classify the whole dialogue. Choose exactly one label from the given label set and output no other words.\n\n"
            f"{LABEL_GUIDANCE}"
        )
    elif caption_prompt_mode == "unified_cues":
        system_msg = (
            "You are an expert in emotion recognition in conversation. "
            "Classify the emotion of the target utterance only. "
            "Use the dialogue context, the target speaker, and the provided observable audio and visual cues as auxiliary evidence. "
            "Do not treat multimodal cues as emotion labels. "
            "Do not classify the whole dialogue. Choose exactly one label from the given label set and output no other words.\n\n"
            f"{LABEL_GUIDANCE}"
        )
    else:
        system_msg = (
            "You are an expert in emotion recognition in conversation. "
            "Classify the emotion of the target utterance only. "
            "Use the dialogue context, the target speaker, and the provided audio and visual descriptions. "
            "Do not classify the whole dialogue. Choose exactly one label from the given label set and output no other words.\n\n"
            f"{LABEL_GUIDANCE}"
        )
    user_parts = [
        f"Available emotion labels:\n{LABEL_TEXT}",
        "Dialogue context:\n" + "\n".join(context_lines),
        f"Target utterance:\n{target_utterance}",
    ]
    user_parts.extend(build_caption_parts(audio_caption, video_caption, caption_prompt_mode))
    user_parts.append(
        "Question:\n"
        "What is the emotion label of the target utterance? "
        "Answer with exactly one label from the available labels."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_legacy_prompt(context_lines, target_utterance, audio_caption, video_caption, caption_prompt_mode):
    prompt = (
        "Now you are expert of sentiment and emotional analysis."
        "The following conversation noted between '### ###' involves several speakers. "
        "### "
        + " ".join(context_lines)
        + " ### "
    )
    for caption_part in build_caption_parts(audio_caption, video_caption, caption_prompt_mode):
        prompt += f" {caption_part.replace(chr(10), ' ')}"
    prompt += (
        f" Please select the emotional label of <{target_utterance}> from <{LABEL_TEXT}> "
        "based on the dialogue context and multimodal descriptions. Respond with just one label:"
    )
    return prompt


def get_caption(captions, name, field):
    return str(captions.get(name, {}).get(field, "") or "").strip()


def build_examples(
    manifest_rows,
    captions,
    window,
    prompt_style,
    caption_prompt_mode,
    use_audio_caption,
    use_video_caption,
    skip_missing,
):
    dialogs = group_dialogs(manifest_rows)
    examples = []
    missing_caption = []
    for dialog_rows in dialogs.values():
        for idx, row in enumerate(dialog_rows):
            name = row["name"]
            start = max(idx - window, 0)
            context_rows = dialog_rows[start : idx + 1]
            cap = captions.get(name, {})
            audio_caption = cap.get("audio_caption", "").strip() if use_audio_caption else ""
            video_caption = cap.get("video_caption", "").strip() if use_video_caption else ""

            if caption_prompt_mode == "context_audio_target_video":
                missing_for_example = []
                context_lines = []
                for ctx_row in context_rows:
                    ctx_audio = get_caption(captions, ctx_row["name"], "audio_caption") if use_audio_caption else ""
                    if use_audio_caption and not ctx_audio:
                        missing_for_example.append(ctx_row["name"])
                    context_lines.append(format_context_line(ctx_row, ctx_audio, include_audio=use_audio_caption))
                if use_video_caption and not video_caption:
                    missing_for_example.append(name)
                if missing_for_example:
                    missing_caption.extend(missing_for_example)
                    if skip_missing:
                        continue
                    raise ValueError(f"Missing required caption for {name}: {sorted(set(missing_for_example))}")
                audio_caption = ""
            else:
                if (use_audio_caption and not audio_caption) or (use_video_caption and not video_caption):
                    missing_caption.append(name)
                    if skip_missing:
                        continue
                    raise ValueError(f"Missing required caption for {name}")
                context_lines = [format_context_line(ctx_row) for ctx_row in context_rows]
            target_utterance = format_context_line(row)
            if prompt_style == "qwen_chat":
                prompt = build_qwen_chat_prompt(
                    context_lines,
                    target_utterance,
                    audio_caption,
                    video_caption,
                    caption_prompt_mode,
                )
            else:
                prompt = build_legacy_prompt(
                    context_lines,
                    target_utterance,
                    audio_caption,
                    video_caption,
                    caption_prompt_mode,
                )
            examples.append(
                {
                    "split": normalize_split(row["split"]),
                    "path": row["video_path"],
                    "utterance_id": name,
                    "target_utterance": target_utterance,
                    "input": prompt,
                    "target": row["label"],
                    "audio_caption": audio_caption,
                    "video_caption": video_caption,
                }
            )
    return examples, missing_caption


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build MELD ERC prompts with Omni audio/video captions.")
    parser.add_argument("--manifest", default="../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl")
    parser.add_argument("--captions", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl")
    parser.add_argument("--out_dir", default="../PROCESSED_DATASET/meld/omni_caption_prompt/window_12_qwen_chat_av")
    parser.add_argument("--historical_window", type=int, default=12)
    parser.add_argument("--prompt_style", choices=["legacy", "qwen_chat"], default="qwen_chat")
    parser.add_argument(
        "--caption_prompt_mode",
        choices=["separate_fields", "unified_cues", "context_audio_target_video"],
        default="separate_fields",
    )
    parser.add_argument("--use_audio_caption", default="True")
    parser.add_argument("--use_video_caption", default="True")
    parser.add_argument("--skip_missing", default="False")
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    captions = latest_ok_captions(args.captions)
    examples, missing_caption = build_examples(
        manifest_rows=manifest_rows,
        captions=captions,
        window=args.historical_window,
        prompt_style=args.prompt_style,
        caption_prompt_mode=args.caption_prompt_mode,
        use_audio_caption=str_to_bool(args.use_audio_caption),
        use_video_caption=str_to_bool(args.use_video_caption),
        skip_missing=str_to_bool(args.skip_missing),
    )

    split_examples = defaultdict(list)
    for example in examples:
        split = example.pop("split")
        split_examples[split].append(example)

    out_dir = Path(args.out_dir)
    for split in ["train", "valid", "test"]:
        write_jsonl(out_dir / f"{split}.json", split_examples.get(split, []))

    report = {
        "manifest_rows": len(manifest_rows),
        "caption_rows": len(captions),
        "examples": len(examples),
        "split_counts": {split: len(split_examples.get(split, [])) for split in ["train", "valid", "test"]},
        "missing_caption": len(missing_caption),
        "missing_caption_examples": missing_caption[:20],
        "use_audio_caption": str_to_bool(args.use_audio_caption),
        "use_video_caption": str_to_bool(args.use_video_caption),
        "historical_window": args.historical_window,
        "prompt_style": args.prompt_style,
        "caption_prompt_mode": args.caption_prompt_mode,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "caption_process_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(out_dir))


if __name__ == "__main__":
    main()
