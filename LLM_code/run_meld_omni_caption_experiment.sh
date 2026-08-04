#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL_NAME="${MODEL_NAME:-qwen2.5}"
MODELS_ROOT="${MODELS_ROOT:-/home/pc/jcy/models}"
OMNI_MODEL_PATH="${OMNI_MODEL_PATH:-${MODELS_ROOT}/Qwen2.5-Omni-7B}"
RAW_BASE="${MELD_RAW_BASE:-/home/pc/jcy/MELD_raw}"
RAW_ROOT="${MELD_RAW_ROOT:-${RAW_BASE}/MELD.Raw}"
CAPTION_ROOT="${CAPTION_ROOT:-../PROCESSED_DATASET/meld/omni_caption}"
MANIFEST_PATH="${MANIFEST_PATH:-${CAPTION_ROOT}/meld_raw_manifest.jsonl}"
CAPTION_PATH="${CAPTION_PATH:-${CAPTION_ROOT}/meld_omni_captions.jsonl}"

HISTORICAL_WINDOW="${HISTORICAL_WINDOW:-12}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_chat}"
USE_AUDIO_CAPTION="${USE_AUDIO_CAPTION:-True}"
USE_VIDEO_CAPTION="${USE_VIDEO_CAPTION:-True}"
CAPTION_DATA_DIR="${CAPTION_DATA_DIR:-../PROCESSED_DATASET/meld/omni_caption_prompt/window_${HISTORICAL_WINDOW}_${PROMPT_STYLE}_av}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LORA_LR="${LORA_LR:-3e-4}"
LORA_DIM="${LORA_DIM:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_MODULE_NAME="${LORA_MODULE_NAME:-q_proj,k_proj,v_proj,query_key_value}"
MAX_LENGTH="${MAX_LENGTH:-1500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"

CAPTION_LIMIT="${CAPTION_LIMIT:-}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-64}"
VIDEO_FPS="${VIDEO_FPS:-0.5}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-100352}"
OMNI_CAPTION_DEVICE="${OMNI_CAPTION_DEVICE:-0,1,2}"
TRAIN_DEVICE="${TRAIN_DEVICE:-0}"
AUDIO_WORKERS="${AUDIO_WORKERS:-4}"
SKIP_CAPTION_GENERATION="${SKIP_CAPTION_GENERATION:-False}"

case "${MODEL_NAME}" in
    qwen2.5|qwen25|Qwen2.5|Qwen25)
        MODEL_LABEL="Qwen2.5"
        MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/Qwen2.5-7B-Instruct}"
        ;;
    llama2|LLaMA2|llama-2)
        MODEL_LABEL="LLaMA2"
        MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/Llama-2-7b-hf}"
        ;;
    *)
        echo "Invalid MODEL_NAME: ${MODEL_NAME}. Expected qwen2.5 or llama2."
        exit 1
        ;;
esac

limit_args=()
if [ -n "${CAPTION_LIMIT}" ]; then
    limit_args=(--limit "${CAPTION_LIMIT}")
fi

echo "******************************************************************************************"
echo "MELD Omni caption experiment"
echo "Raw root: ${RAW_ROOT}"
echo "Omni model: ${OMNI_MODEL_PATH}"
echo "Train model: ${MODEL_PATH}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Captions: ${CAPTION_PATH}"
echo "Caption data dir: ${CAPTION_DATA_DIR}"
echo "Caption limit: ${CAPTION_LIMIT:-full}"
echo "Omni devices: ${OMNI_CAPTION_DEVICE}"
echo "Train device: ${TRAIN_DEVICE}"
echo "******************************************************************************************"

bash prepare_meld_raw_data.sh "${RAW_BASE}"

python build_meld_caption_manifest.py \
    --raw_root "${RAW_ROOT}" \
    --out "${MANIFEST_PATH}"

python extract_meld_audio.py \
    --manifest "${MANIFEST_PATH}" \
    --num_workers "${AUDIO_WORKERS}" \
    "${limit_args[@]}"

if [ "${SKIP_CAPTION_GENERATION}" != "True" ]; then
    CUDA_VISIBLE_DEVICES="${OMNI_CAPTION_DEVICE}" python generate_meld_omni_captions.py \
        --manifest "${MANIFEST_PATH}" \
        --out "${CAPTION_PATH}" \
        --model_path "${OMNI_MODEL_PATH}" \
        --caption_type audio \
        --max_new_tokens "${CAPTION_MAX_NEW_TOKENS}" \
        "${limit_args[@]}"

    CUDA_VISIBLE_DEVICES="${OMNI_CAPTION_DEVICE}" python generate_meld_omni_captions.py \
        --manifest "${MANIFEST_PATH}" \
        --out "${CAPTION_PATH}" \
        --model_path "${OMNI_MODEL_PATH}" \
        --caption_type video \
        --max_new_tokens "${CAPTION_MAX_NEW_TOKENS}" \
        --video_fps "${VIDEO_FPS}" \
        --video_max_pixels "${VIDEO_MAX_PIXELS}" \
        "${limit_args[@]}"
fi

python validate_meld_omni_captions.py \
    --manifest "${MANIFEST_PATH}" \
    --captions "${CAPTION_PATH}"

python data_process_meld_omni_caption.py \
    --manifest "${MANIFEST_PATH}" \
    --captions "${CAPTION_PATH}" \
    --out_dir "${CAPTION_DATA_DIR}" \
    --historical_window "${HISTORICAL_WINDOW}" \
    --prompt_style "${PROMPT_STYLE}" \
    --use_audio_caption "${USE_AUDIO_CAPTION}" \
    --use_video_caption "${USE_VIDEO_CAPTION}"

OUTPUT_DIR="../experiments/${MODEL_LABEL}/lora/meld/window_${HISTORICAL_WINDOW}/LR_${LORA_LR}_BS_${BATCH_SIZE}_omni_caption_av_${SEED}_run_${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "dataset": "meld",
  "model_name": "${MODEL_NAME}",
  "model_label": "${MODEL_LABEL}",
  "model_path": "${MODEL_PATH}",
  "data_source": "omni_caption",
  "data_path": "${CAPTION_DATA_DIR}",
  "caption_path": "${CAPTION_PATH}",
  "manifest_path": "${MANIFEST_PATH}",
  "use_audio_caption": "${USE_AUDIO_CAPTION}",
  "use_video_caption": "${USE_VIDEO_CAPTION}",
  "historical_window": "${HISTORICAL_WINDOW}",
  "prompt_style": "${PROMPT_STYLE}",
  "video_fps": "${VIDEO_FPS}",
  "video_max_pixels": "${VIDEO_MAX_PIXELS}",
  "caption_limit": "${CAPTION_LIMIT}",
  "seed": "${SEED}",
  "output_dir": "${OUTPUT_DIR}"
}
EOF

CUDA_VISIBLE_DEVICES="${TRAIN_DEVICE}" python main.py \
    --dataset meld \
    --model_name_or_path "${MODEL_PATH}" \
    --data_dir "${CAPTION_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_batch_size 1 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --save_steps 100000 \
    --lora True \
    --lora_dim "${LORA_DIM}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_module_name "${LORA_MODULE_NAME}" \
    --learning_rate "${LORA_LR}" \
    --do_eval True \
    --do_train True \
    --statistic_mode True \
    --data_percent 1.0 \
    --seed "${SEED}" \
    --use_mm_prefix False

echo "Finished MELD Omni caption experiment: ${OUTPUT_DIR}"
