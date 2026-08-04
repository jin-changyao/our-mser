import argparse
import json
import os
from pathlib import Path

import torch


SYSTEM_PROMPT = (
    "You are an expert multimodal annotator for emotion recognition in conversation. "
    "Describe observable audio or visual cues. Do not output an emotion label."
)

AUDIO_PROMPT = (
    "Listen to the audio only. Describe speech and acoustic cues useful for emotion recognition, "
    "such as tone, pitch, loudness, pace, pauses, laughter, sighs, and vocal affect. "
    "Do not transcribe or quote the spoken words. Do not infer or output an emotion label. "
    "Write one concise English sentence."
)

VIDEO_PROMPT = (
    "Watch the video only. Describe visible cues useful for emotion recognition, such as facial "
    "expression, gaze, gestures, posture, movement, and interaction cues. Ignore audio. "
    "Do not transcribe or quote the spoken words. Do not infer or output an emotion label. "
    "Write one concise English sentence."
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


def build_conversation(media_type, media_path, prompt):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": media_type, media_type: media_path},
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


def generate_one(model, processor, process_mm_info, media_type, media_path, prompt, max_new_tokens, return_audio_arg):
    if not Path(media_path).is_file():
        raise FileNotFoundError(media_path)
    use_audio_in_video = False
    conversation = build_conversation(media_type, str(media_path), prompt)
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
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


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
    parser.add_argument("--prompt_version", default="omni_caption_v1")
    parser.add_argument("--attn_implementation", default="", help="Example: flash_attention_2")
    parser.add_argument("--model_class", choices=["thinker", "full"], default="thinker")
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
            if old.get("status") == "ok" and has_audio and has_video:
                skipped += 1
                continue

        record = {
            "name": row["name"],
            "split": row["split"],
            "prompt_version": args.prompt_version,
            "model_path": args.model_path,
            "status": "ok",
        }
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

