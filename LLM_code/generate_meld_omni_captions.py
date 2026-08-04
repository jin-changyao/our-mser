import argparse
import json
import os
from pathlib import Path

import torch


AUDIO_PROMPT = (
    "Listen to the audio only. Describe directly audible properties of the speaker's voice, such as "
    "pitch, loudness, speaking pace, pauses, laughter, sighs, breathing, and voice quality. "
    "If laughter or noise is from the background, audience, or laugh track rather than the speaker, call it background sound. "
    "Mention absent cues only if their absence is unusual or important; do not list things that are not present. "
    "Do not transcribe words, infer emotion, or explain what the cues indicate. Avoid emotion and mood words. "
    "Write one complete English sentence under 30 words."
)

VIDEO_PROMPT = (
    "Watch the video only. Describe directly visible actions and appearance, such as facial movement, gaze, "
    "head movement, hand gestures, posture, body movement, and interaction with objects or people. "
    "Ignore audio and do not transcribe words. Do not infer emotion, mood, attitude, intention, relationship, or mental state. "
    "Do not explain what the cues indicate. Avoid emotion and mood words. "
    "Write one complete English sentence under 35 words."
)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing(path):
    existing = {}
    if not path.is_file():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing[row.get("name")] = row
    return existing


def select_rows(rows, split, limit):
    if split:
        rows = [row for row in rows if row.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_conversation(media_type, media_path, prompt, video_fps=None, video_max_pixels=None):
    media_item = {"type": media_type, media_type: media_path}
    if media_type == "video":
        if video_fps is not None:
            media_item["fps"] = video_fps
        if video_max_pixels is not None:
            media_item["max_pixels"] = video_max_pixels
    return [
        {
            "role": "user",
            "content": [
                media_item,
                {"type": "text", "text": prompt},
            ],
        },
    ]


def move_inputs_to_model(inputs, model):
    device = next(model.parameters()).device
    dtype = getattr(model, "dtype", None)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            value = value.to(device)
            if dtype is not None and value.dtype.is_floating_point:
                value = value.to(dtype)
            inputs[key] = value
    return inputs


def clean_caption(text):
    text = str(text).strip()
    for marker in ["\nHuman:", "\nUser:", "\nAssistant:", "\nhuman:", "\nuser:", "\nassistant:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip().strip("`")


def generate_one(
    model,
    processor,
    process_mm_info,
    media_type,
    media_path,
    prompt,
    max_new_tokens,
    return_audio_arg,
    video_fps=None,
    video_max_pixels=None,
):
    if not Path(media_path).is_file():
        raise FileNotFoundError(media_path)
    use_audio_in_video = False
    conversation = build_conversation(
        media_type,
        str(media_path),
        prompt,
        video_fps=video_fps,
        video_max_pixels=video_max_pixels,
    )
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = move_inputs_to_model(inputs, model)
    generate_kwargs = {
        "use_audio_in_video": use_audio_in_video,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if return_audio_arg:
        generate_kwargs["return_audio"] = False
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    if isinstance(output_ids, tuple):
        output_ids = output_ids[0]
    input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_ids = output_ids[:, input_length:] if input_length else output_ids
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    caption = clean_caption(decoded[0])
    del inputs, output_ids, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return caption


def load_omni(model_path, attn_implementation, model_class):
    try:
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        from qwen_omni_utils import process_mm_info
    except Exception as exc:
        raise RuntimeError(
            "Qwen2.5-Omni dependencies are missing. Install the Omni-compatible transformers build "
            "and qwen-omni-utils[decord] on the server before running this script."
        ) from exc

    kwargs = {"device_map": "auto", "torch_dtype": "auto"}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if model_class == "full":
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if hasattr(model, "disable_talker"):
            model.disable_talker()
        return_audio_arg = True
    else:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **kwargs)
        return_audio_arg = False
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor, process_mm_info, return_audio_arg


def main():
    parser = argparse.ArgumentParser(description="Generate MELD audio/video captions with Qwen2.5-Omni.")
    parser.add_argument("--manifest", default="../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl")
    parser.add_argument("--out", default="../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl")
    parser.add_argument("--model_path", default=os.environ.get("OMNI_MODEL_PATH", "/home/pc/jcy/models/Qwen2.5-Omni-7B"))
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--caption_type", choices=["audio", "video", "both"], default="both")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--prompt_version", default="omni_caption_v2_observable")
    parser.add_argument("--attn_implementation", default="", help="Example: flash_attention_2")
    parser.add_argument("--model_class", choices=["thinker", "full"], default="thinker")
    parser.add_argument("--video_fps", type=float, default=1.0, help="Frames per second sampled from each video.")
    parser.add_argument("--video_max_pixels", type=int, default=200704, help="Maximum pixels per sampled video frame.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    existing = load_existing(out_path)
    rows = select_rows(read_jsonl(args.manifest), args.split, args.limit)
    model, processor, process_mm_info, return_audio_arg = load_omni(
        args.model_path,
        args.attn_implementation or None,
        args.model_class,
    )

    written = 0
    skipped = 0
    for row in rows:
        old = existing.get(row["name"])
        needs_audio = args.caption_type in {"audio", "both"}
        needs_video = args.caption_type in {"video", "both"}
        if old and not args.overwrite:
            has_audio = (not needs_audio) or bool(old.get("audio_caption"))
            has_video = (not needs_video) or bool(old.get("video_caption"))
            if has_audio and has_video:
                skipped += 1
                continue

        record = dict(old) if old and not args.overwrite else {}
        record.update(
            {
                "name": row["name"],
                "split": row["split"],
                "prompt_version": args.prompt_version,
                "model_path": args.model_path,
                "status": "ok",
            }
        )
        record.pop("error", None)
        try:
            if needs_audio:
                record["audio_caption"] = generate_one(
                    model,
                    processor,
                    process_mm_info,
                    "audio",
                    row["audio_path"],
                    AUDIO_PROMPT,
                    args.max_new_tokens,
                    return_audio_arg,
                )
            if needs_video:
                record["video_caption"] = generate_one(
                    model,
                    processor,
                    process_mm_info,
                    "video",
                    row["video_path"],
                    VIDEO_PROMPT,
                    args.max_new_tokens,
                    return_audio_arg,
                    video_fps=args.video_fps,
                    video_max_pixels=args.video_max_pixels,
                )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)
        append_jsonl(out_path, record)
        written += 1
        print(json.dumps({"written": written, "skipped": skipped, "last": record["name"], "status": record["status"]}))

    print(json.dumps({"rows_selected": len(rows), "written": written, "skipped": skipped, "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()






