#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-/home/pc/jcy/models/Qwen2.5-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-../persona_features/meld_qwen25_persona.json}"

python generate_persona_features.py \
    --project_root "${PROJECT_ROOT}" \
    --dataset meld \
    --model_name_or_path "${MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --max_input_tokens "${MAX_INPUT_TOKENS:-3500}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-96}" \
    --device_map "${DEVICE_MAP:-cuda}" \
    --torch_dtype "${TORCH_DTYPE:-float16}" \
    --min_utterances "${MIN_UTTERANCES:-2}" \
    --save_every "${SAVE_EVERY:-5}" \
    "$@"
