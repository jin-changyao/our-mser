#!/usr/bin/env python3
"""Generate comparable prompts for the five single-seed experiments."""

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

METHODS = ("none", "emotion", "speaker", "retrieval", "full")
PLACEHOLDER = "No reliable speaker description is available."
LABELS = {
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
}
EMOTION_MAP = {
    "iemocap": dict(enumerate(LABELS["iemocap"])),
    "meld": dict(enumerate(LABELS["meld"])),
}


def clean(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def speaker_name(dataset: str, conversation_id: str, speaker: Any) -> str:
    if dataset == "meld":
        return str(speaker)
    names = {
        "Ses01": {"F": "MARY", "M": "JAMES"},
        "Ses02": {"F": "PATRICIA", "M": "JOHN"},
        "Ses03": {"F": "JENNIFER", "M": "ROBERT"},
        "Ses04": {"F": "LINDA", "M": "MICHAEL"},
        "Ses05": {"F": "ELIZABETH", "M": "WILLIAM"},
    }
    return names[conversation_id[:5]][str(speaker)]


def speaker_key(dataset: str, conversation: dict[str, Any], index: int) -> Any:
    return conversation["speakers"][index] if dataset == "meld" else conversation["genders"][index]


def context_lines(dataset: str, conversation: dict[str, Any], conversation_id: str, index: int, window: int) -> list[str]:
    return [
        f" {speaker_name(dataset, conversation_id, speaker_key(dataset, conversation, j))}: {conversation['sentences'][j]}"
        for j in range(max(0, index - window), min(len(conversation["sentences"]), index + window + 1))
    ]


def calculate_difficulty(dataset: str, conversation: dict[str, Any]) -> float:
    """Match the existing curriculum difficulty calculation."""
    import similarity_matrix

    matrix, emotion_to_index = similarity_matrix.get_similarity_matrix(dataset)
    labels = [EMOTION_MAP[dataset][label] for label in conversation["labels"]]
    speakers = [speaker_key(dataset, conversation, i) for i in range(len(labels))]
    num_speakers = len(set(str(item) for item in speakers))
    by_speaker: dict[str, list[str]] = {}
    for speaker, label in zip(speakers, labels):
        by_speaker.setdefault(str(speaker), []).append(label)

    emotion_shift_weighted = 0.0
    for sequence in by_speaker.values():
        for current, following in zip(sequence, sequence[1:]):
            if current == following or current == "null" or following == "null":
                continue
            emotion_shift_weighted += abs(
                matrix[emotion_to_index[current]][emotion_to_index[following]]
            ) + 0.4

    cross_speaker_shift = 0.0
    for i in range(1, len(labels)):
        if str(speakers[i]) == str(speakers[i - 1]):
            continue
        current, previous = labels[i], labels[i - 1]
        if current == previous or current == "null" or previous == "null":
            continue
        cross_speaker_shift += abs(
            matrix[emotion_to_index[current]][emotion_to_index[previous]]
        ) + 0.4

    return (emotion_shift_weighted + num_speakers + cross_speaker_shift) / (
        len(labels) + num_speakers
    )


class Retriever:
    def __init__(self, path: Path, embedding_model_path: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        with path.open("r", encoding="utf-8") as handle:
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
        device = os.environ.get("PRC_EMO_EMBEDDING_DEVICE", "cuda")
        self.encoder = SentenceTransformer(embedding_model_path, device=device)

    def search(self, text: str, conversation_id: str, index: int, split: str) -> list[dict[str, Any]]:
        query = np.asarray(self.encoder.encode([text], convert_to_tensor=False), dtype="float32")
        distances, indices = self.index.search(query, 8)
        results = []
        for distance, record_index in zip(distances[0], indices[0]):
            if record_index < 0 or record_index >= len(self.metadata):
                continue
            record = self.metadata[record_index]
            if split == "train" and (
                record["conversation_id"] == conversation_id
                and record["utterance_id"] == index
            ):
                continue
            results.append({**record, "distance": float(distance)})
            if len(results) == 3:
                break
        return results


def emotion_info(emotion_data: dict[str, Any], conversation_id: str, index: int) -> tuple[str, str]:
    item = emotion_data[conversation_id]["emotion_predictions"][index]
    surface = item.get("surface_emotion", item.get("explicit_emotion", ""))
    implicit = item.get("implicit_emotion", "")
    return clean(surface), clean(implicit)


def speaker_info(speaker_data: dict[str, Any], conversation_id: str, index: int) -> str:
    if conversation_id not in speaker_data or index >= len(speaker_data[conversation_id]):
        return PLACEHOLDER
    description = clean(speaker_data[conversation_id][index])
    return description or PLACEHOLDER


def build_messages(
    dataset: str,
    conversation: dict[str, Any],
    conversation_id: str,
    index: int,
    window: int,
    method: str,
    emotion_data: dict[str, Any] | None,
    speaker_data: dict[str, Any] | None,
    retriever: Retriever | None,
    split: str,
) -> list[dict[str, str]]:
    target = conversation["sentences"][index]
    target_name = speaker_name(dataset, conversation_id, speaker_key(dataset, conversation, index))
    system = (
        "### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\n"
        "Your goal is to infer the most accurate **emotion label** for a given utterance.\n"
        "### Given the following conversation as a context\n"
        + "\n".join(context_lines(dataset, conversation, conversation_id, index, window))
        + "\n"
    )

    additions: list[str] = []
    if method in ("speaker", "full"):
        additions.append(
            f"### Given the characteristic of this speaker:\n### Speaker: {target_name}\n"
            f"- {speaker_info(speaker_data or {}, conversation_id, index)}\n"
        )
    if method in ("emotion", "full"):
        surface, implicit = emotion_info(emotion_data or {}, conversation_id, index)
        additions.append(
            f"### Given the speaker's Explicit Emotion Interpretation and Implicit Emotion Interpretation "
            f"in the utterance \"{target}\":\n### Speaker: {target_name}\n"
            f"- Explicit Emotion Interpretation: {surface}\n"
            f"- Implicit Emotion Interpretation: {implicit}\n"
        )
    if method in ("retrieval", "full"):
        references = retriever.search(target, conversation_id, index, split) if retriever else []
        block = ["### Reference Similar Emotional Expressions:"]
        for number, reference in enumerate(references, 1):
            block.append(
                f'{number}. "{reference["text"]}" → {reference["label"]} '
                f'(from {reference["dataset"]}, distance: {reference["distance"]:.2f})'
            )
        additions.append("\n".join(block) + "\n")

    labels = ", ".join(LABELS[dataset])
    if additions:
        system += "\n" + "\n".join(additions)
    system += f"\n### Available emotion labels: {labels}\n"
    question = (
        f'Based on the above conversation and the provided information, which emotional label of '
        f'{target_name} is expressed in the utterance "{target}".'
    )
    label = LABELS[dataset][conversation["labels"][index]]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": label},
    ]


def generate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_folder)
    speaker_dir = Path(args.speaker_feature_dir or args.data_folder)
    emotion_suffix = args.emotion_feature_suffix
    speaker_suffix = args.speaker_feature_suffix
    retriever = None
    if args.method in ("retrieval", "full"):
        retriever = Retriever(Path(args.retrieval_path), args.embedding_model_path)

    for split in ("train", "valid", "test"):
        raw_path = data_dir / f"{args.dataset}.{split}.json"
        conversations = json.loads(raw_path.read_text(encoding="utf-8"))
        emotion_data = None
        speaker_data = None
        if args.method in ("emotion", "full"):
            emotion_path = data_dir / f"{args.dataset}.{split}_{emotion_suffix}.json"
            emotion_data = json.loads(emotion_path.read_text(encoding="utf-8"))
        if args.method in ("speaker", "full"):
            speaker_path = speaker_dir / f"{args.dataset}.{split}_{speaker_suffix}.json"
            speaker_data = json.loads(speaker_path.read_text(encoding="utf-8"))

        difficulty = {}
        if split == "train":
            difficulty = {
                conversation_id: calculate_difficulty(args.dataset, conversation)
                for conversation_id, conversation in conversations.items()
            }

        output_path = data_dir / f"{args.dataset}.{split}.0shot_w{args.window}_experiment_{args.method}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            count = 0
            for conversation_id, conversation in conversations.items():
                for index in range(len(conversation["sentences"])):
                    record = {
                        "messages": build_messages(
                            args.dataset,
                            conversation,
                            conversation_id,
                            index,
                            args.window,
                            args.method,
                            emotion_data,
                            speaker_data,
                            retriever,
                            split,
                        ),
                        "difficulty": round(difficulty.get(conversation_id, 0.0), 4),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        print(f"Generated {count} samples: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iemocap", "meld"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data_folder", default=str(ROOT / "data"))
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--emotion_feature_suffix", default="ImplicitEmotion_V3_Qwen2.5-7B-Instruct")
    parser.add_argument("--speaker_feature_dir", default=None)
    parser.add_argument("--speaker_feature_suffix", default="spdescV6_trimmed_Qwen2.5-7B-Instruct")
    parser.add_argument("--retrieval_path", default=str(ROOT / "data" / "Emotion_Retrieval_Library_3.json"))
    parser.add_argument(
        "--embedding_model_path",
        default=os.environ.get("PRC_EMO_EMBEDDING_MODEL_PATH", str(ROOT / "models" / "all-MiniLM-L6-v2")),
    )
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
