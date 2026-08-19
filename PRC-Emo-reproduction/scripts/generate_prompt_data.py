"""Generate the final retrieval-augmented JSONL prompts without training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iemocap", "meld"], required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--prompting_type", default="ImplicitEmotion_V3")
    parser.add_argument("--extract_prompting_llm_id", default="Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--speaker_feature_dir",
        default=None,
        help="Directory containing speaker feature JSON files; defaults to data.",
    )
    parser.add_argument(
        "--speaker_feature_suffix",
        default="spdescV6_Qwen2.5-7B-Instruct",
        help="Suffix after <dataset>.<split>_ for speaker feature files.",
    )
    args = parser.parse_args()

    # Import after changing to the project root because the author's module
    # loads ./data/Emotion_Retrieval_Library_3.json during import.
    from reformat_data_ft_llm_combine import process

    data_folder = str(ROOT / "data")
    outputs = [
        str(ROOT / "data" / f"{args.dataset}.{split}.0shot_w{args.window}_{args.prompting_type}.jsonl")
        for split in ("train", "valid", "test")
    ]
    process(
        outputs,
        SimpleNamespace(
            data_folder=data_folder,
            data_name=args.dataset,
            window=args.window,
            prompting_type=args.prompting_type,
            extract_prompting_llm_id=args.extract_prompting_llm_id,
            speaker_feature_dir=args.speaker_feature_dir,
            speaker_feature_suffix=args.speaker_feature_suffix,
        ),
    )
    print("Generated:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
