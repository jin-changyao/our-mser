#!/usr/bin/env python3
"""Build controlled IEMOCAP ablation prompts.

The four variants intentionally share the same local conversation context and
the same labels. They differ only in which optional information is included:

    none          local context only
    speaker       local context + speaker profile
    explicit      local context + explicit emotion description
    retrieval     local context + retrieved labelled examples

This file is separate from the original prompt formatter so the original
reproduction path remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VARIANTS = ("none", "speaker", "explicit", "retrieval")
EMOTION_MODEL_ID = "Qwen2.5-7B-Instruct"
SPEAKER_FEATURE_ID = "Qwen2.5-7B-Instruct"


def speaker_name(conversation_id: str, gender: str) -> str:
    names = {
        "Ses01": {"F": "Mary", "M": "James"},
        "Ses02": {"F": "Patricia", "M": "John"},
        "Ses03": {"F": "Jennifer", "M": "Robert"},
        "Ses04": {"F": "Linda", "M": "Michael"},
        "Ses05": {"F": "Elizabeth", "M": "William"},
    }
    return names[conversation_id[:5]][gender].upper()


def labels() -> list[str]:
    return ["happy", "sad", "neutral", "angry", "excited", "frustrated"]


def emotion_map() -> dict[int, str]:
    return {
        0: "happy",
        1: "sad",
        2: "neutral",
        3: "angry",
        4: "excited",
        5: "frustrated",
    }


def clean_text(value: Any) -> str:
    return str(value).replace("\n", " ").strip()


def context_lines(conv: dict[str, Any], conversation_id: str, index: int, window: int) -> list[str]:
    return [
        f" {speaker_name(conversation_id, conv['genders'][j])}: {conv['sentences'][j]}"
        for j in range(max(0, index - window), min(len(conv["sentences"]), index + window + 1))
    ]


def calculate_difficulty(conv: dict[str, Any]) -> float:
    """Match the original curriculum difficulty calculation for IEMOCAP."""
    import similarity_matrix

    emotion_lookup = emotion_map()
    matrix, emotion_to_index = similarity_matrix.get_similarity_matrix("iemocap")
    mapped_labels = [emotion_lookup[label] for label in conv["labels"]]
    genders = conv["genders"]
    num_utterances = len(mapped_labels)
    num_speakers = len(set(genders))

    speaker_emotions: dict[str, list[str]] = {}
    for gender, emotion in zip(genders, mapped_labels):
        speaker_emotions.setdefault(gender, []).append(emotion)

    emotion_shift_weighted = 0.0
    number_of_shifts = 0
    for emotions in speaker_emotions.values():
        for current, following in zip(emotions, emotions[1:]):
            if current == following or current == "null" or following == "null":
                continue
            number_of_shifts += 1
            score = abs(matrix[emotion_to_index[current]][emotion_to_index[following]]) + 0.4
            emotion_shift_weighted += score

    speaker_turns = sum(genders[i] != genders[i - 1] for i in range(1, len(genders)))
    cross_speaker_shift = 0.0
    for i in range(1, num_utterances):
        if genders[i] == genders[i - 1]:
            continue
        current = mapped_labels[i]
        previous = mapped_labels[i - 1]
        if current == previous or current == "null" or previous == "null":
            continue
        score = abs(matrix[emotion_to_index[current]][emotion_to_index[previous]]) + 0.4
        cross_speaker_shift += score

    # Keep the same expression used by reformat_data_ft_llm_combine.py.
    _ = number_of_shifts, speaker_turns
    return (emotion_shift_weighted + num_speakers + cross_speaker_shift) / (num_utterances + num_speakers)


class Retriever:
    def __init__(self, retrieval_path: Path, embedding_path: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        with retrieval_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        vectors = np.asarray([record["vector"] for record in records], dtype="float32")
        self.metadata = [
            {
                "text": record["text"],
                "label": record["label"],
                "dataset": record["dataset"],
                "conversation_id": record["conversation_id"],
                "utterance_id": record["utterance_id"],
            }
            for record in records
        ]
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.encoder = SentenceTransformer(embedding_path, device="cuda")
        print(f"Loaded retrieval library: {len(self.metadata)} samples")

    def search(
        self,
        text: str,
        conversation_id: str,
        utterance_id: int,
        exclude_self: bool,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        query = np.asarray(self.encoder.encode([text], convert_to_tensor=False), dtype="float32")
        distances, indices = self.index.search(query, k + 5)
        results = []
        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(self.metadata):
                continue
            sample = self.metadata[index]
            if exclude_self and (
                sample["conversation_id"] == conversation_id
                and sample["utterance_id"] == utterance_id
            ):
                continue
            results.append(
                {
                    "text": sample["text"],
                    "label": sample["label"],
                    "dataset": sample["dataset"],
                    "distance": float(distance),
                }
            )
            if len(results) >= k:
                break
        return results


def load_feature(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing feature file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def emotion_records(features: dict[str, Any], conversation_id: str) -> list[dict[str, str]]:
    predictions = features[conversation_id]["emotion_predictions"]
    return [
        {
            "surface": clean_text(item.get("surface_emotion", "")),
            "implicit": clean_text(item.get("implicit_emotion", "")),
        }
        for item in predictions
    ]


def build_messages(
    conv: dict[str, Any],
    conversation_id: str,
    index: int,
    window: int,
    variant: str,
    speaker_features: dict[str, Any] | None,
    emotion_features: dict[str, Any] | None,
    retriever: Retriever | None,
    split: str,
) -> list[dict[str, str]]:
    target_name = speaker_name(conversation_id, conv["genders"][index])
    target_sentence = conv["sentences"][index]
    context = "\n".join(context_lines(conv, conversation_id, index, window))
    system = (
        "### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\n"
        "Your goal is to infer the most accurate **emotion label** for a given utterance.\n"
    )

    additions = []
    if variant == "speaker":
        description = clean_text(speaker_features[conversation_id][index])
        additions.append(
            f"\n### Given the characteristic of this speaker, {target_name}:\n{description}\n"
        )
    elif variant == "explicit":
        description = emotion_records(emotion_features, conversation_id)[index]["surface"]
        additions.append(
            f"\n### Given the speaker's Explicit Emotion Interpretation in the utterance "
            f"\"{target_sentence}\":\n"
            f"- Explicit Emotion Interpretation: {description}\n"
        )
    elif variant == "retrieval":
        references = retriever.search(
            target_sentence,
            conversation_id,
            index,
            exclude_self=(split == "train"),
        )
        demonstration = "### Reference Similar Emotional Expressions:\n"
        for number, sample in enumerate(references, 1):
            demonstration += (
                f'{number}. "{sample["text"]}" → {sample["label"]} '
                f'(from {sample["dataset"]}, distance: {sample["distance"]:.2f})\n'
            )
        additions.append("\n" + demonstration)

    system_content = system + "\n### Given the following conversation as a context\n" + context
    if additions:
        system_content += "\n" + "\n".join(additions)
    if variant == "retrieval":
        system_content += "\n### Available emotion labels: " + ", ".join(labels()) + "\n"

    question = (
        f'Based on the above conversation, which emotional label of {target_name} '
        f'in the utterance "{target_sentence}".'
    )
    label = labels()[conv["labels"][index]]
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
        {"role": "assistant", "content": label},
    ]


def generate(args: argparse.Namespace) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant {args.variant}; choose from {', '.join(VARIANTS)}")

    data_dir = Path(args.data_folder)
    model_id = Path(args.emotion_model_id).name
    emotion_by_split: dict[str, dict[str, Any]] = {}
    speaker_by_split: dict[str, dict[str, Any]] = {}

    if args.variant == "explicit":
        for split in ("train", "valid", "test"):
            emotion_by_split[split] = load_feature(
                data_dir / f"iemocap.{split}_ImplicitEmotion_V3_{model_id}.json"
            )
    if args.variant == "speaker":
        for split in ("train", "valid", "test"):
            speaker_by_split[split] = load_feature(
                data_dir / f"iemocap.{split}_spdescV6_{SPEAKER_FEATURE_ID}.json"
            )

    retriever = None
    if args.variant == "retrieval":
        retriever = Retriever(Path(args.retrieval_path), args.embedding_model_path)

    for split in ("train", "valid", "test"):
        raw_path = data_dir / f"iemocap.{split}.json"
        with raw_path.open("r", encoding="utf-8") as handle:
            conversations = json.load(handle)

        output_path = data_dir / f"iemocap.{split}.0shot_w{args.window}_ablation_{args.variant}.jsonl"
        matrix_difficulty = {
            conversation_id: calculate_difficulty(conv) for conversation_id, conv in conversations.items()
        } if split == "train" else {}

        with output_path.open("w", encoding="utf-8") as handle:
            count = 0
            for conversation_id, conv in conversations.items():
                for index in range(len(conv["sentences"])):
                    messages = build_messages(
                        conv,
                        conversation_id,
                        index,
                        args.window,
                        args.variant,
                        speaker_by_split.get(split),
                        emotion_by_split.get(split),
                        retriever,
                        split,
                    )
                    record = {"messages": messages, "difficulty": matrix_difficulty.get(conversation_id, 0.0)}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        print(f"Generated {count} samples: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data_folder", default=str(ROOT / "data"))
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--emotion_model_id", default=EMOTION_MODEL_ID)
    parser.add_argument("--retrieval_path", default=str(ROOT / "data" / "Emotion_Retrieval_Library_3.json"))
    parser.add_argument(
        "--embedding_model_path",
        default=os.environ.get("PRC_EMO_EMBEDDING_MODEL_PATH", str(ROOT / "models" / "all-MiniLM-L6-v2")),
    )
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
