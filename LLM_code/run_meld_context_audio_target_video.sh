#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL_NAME="${MODEL_NAME:-qwen2.5}"
MODELS_ROOT="${MODELS_ROOT:-/home/pc/jcy/models}"
RAW_BASE="${MELD_RAW_BASE:-/home/pc/jcy/MELD_raw}"
RAW_ROOT="${MELD_RAW_ROOT:-${RAW_BASE}/MELD.Raw}"
CAPTION_ROOT="${CAPTION_ROOT:-../PROCESSED_DATASET/meld/omni_caption}"
MANIFEST_PATH="${MANIFEST_PATH:-${CAPTION_ROOT}/meld_raw_manifest.jsonl}"
RAW_CAPTION_PATH="${RAW_CAPTION_PATH:-${CAPTION_ROOT}/meld_omni_captions.jsonl}"
CLEAN_CAPTION_PATH="${CLEAN_CAPTION_PATH:-${CAPTION_ROOT}/meld_omni_captions.clean.context_audio_target_video.jsonl}"
RERUN_LIST_PATH="${RERUN_LIST_PATH:-${CAPTION_ROOT}/meld_omni_captions.rerun_needed.jsonl}"

HISTORICAL_WINDOW="${HISTORICAL_WINDOW:-12}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_chat}"
CAPTION_PROMPT_MODE="${CAPTION_PROMPT_MODE:-context_audio_target_video}"
CAPTION_DATA_DIR="${CAPTION_DATA_DIR:-../PROCESSED_DATASET/meld/omni_caption_prompt/window_${HISTORICAL_WINDOW}_${PROMPT_STYLE}_context_audio_target_video}"
SKIP_MISSING_CAPTIONS="${SKIP_MISSING_CAPTIONS:-True}"

RERUN_BAD_CAPTIONS="${RERUN_BAD_CAPTIONS:-False}"
OMNI_MODEL_PATH="${OMNI_MODEL_PATH:-${MODELS_ROOT}/Qwen2.5-Omni-7B}"
OMNI_MODEL_CLASS="${OMNI_MODEL_CLASS:-thinker}"
OMNI_VIDEO_FPS="${OMNI_VIDEO_FPS:-1.0}"
OMNI_VIDEO_MAX_PIXELS="${OMNI_VIDEO_MAX_PIXELS:-200704}"
OMNI_MAX_NEW_TOKENS="${OMNI_MAX_NEW_TOKENS:-96}"

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
MAX_LENGTH="${MAX_LENGTH:-1800}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"
TRAIN_DEVICE="${TRAIN_DEVICE:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29800}"
SKIP_PREPARE="${SKIP_PREPARE:-False}"
SKIP_TRAINING="${SKIP_TRAINING:-False}"
RUN_ANALYSIS="${RUN_ANALYSIS:-True}"

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

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path not found: ${MODEL_PATH}"
    exit 1
fi
if [ "${RERUN_BAD_CAPTIONS}" = "True" ] && [ ! -d "${OMNI_MODEL_PATH}" ]; then
    echo "Omni model path not found: ${OMNI_MODEL_PATH}"
    exit 1
fi

echo "******************************************************************************************"
echo "MELD context-audio + target-video caption experiment"
echo "Run id: ${RUN_ID}"
echo "Seed: ${SEED}"
echo "Train device: ${TRAIN_DEVICE}"
echo "Train port: ${MASTER_PORT}"
echo "Rerun bad captions: ${RERUN_BAD_CAPTIONS}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Raw captions: ${RAW_CAPTION_PATH}"
echo "Clean captions: ${CLEAN_CAPTION_PATH}"
echo "Caption data dir: ${CAPTION_DATA_DIR}"
echo "Train model: ${MODEL_PATH}"
echo "******************************************************************************************"

if [ "${SKIP_PREPARE}" != "True" ]; then
    bash prepare_meld_raw_data.sh "${RAW_BASE}"

    python build_meld_caption_manifest.py \
        --raw_root "${RAW_ROOT}" \
        --out "${MANIFEST_PATH}"

    python clean_meld_omni_captions.py \
        --captions "${RAW_CAPTION_PATH}" \
        --manifest "${MANIFEST_PATH}" \
        --out "${CLEAN_CAPTION_PATH}" \
        --rerun_out "${RERUN_LIST_PATH}" \
        --report_out "${CAPTION_ROOT}/meld_omni_captions.clean.context_audio_target_video_report.json"

    if [ "${RERUN_BAD_CAPTIONS}" = "True" ]; then
        python generate_meld_omni_captions.py \
            --manifest "${MANIFEST_PATH}" \
            --out "${RAW_CAPTION_PATH}" \
            --model_path "${OMNI_MODEL_PATH}" \
            --caption_type both \
            --prompt_version grpo_style_context_v1 \
            --model_class "${OMNI_MODEL_CLASS}" \
            --video_fps "${OMNI_VIDEO_FPS}" \
            --video_max_pixels "${OMNI_VIDEO_MAX_PIXELS}" \
            --max_new_tokens "${OMNI_MAX_NEW_TOKENS}" \
            --names_file "${RERUN_LIST_PATH}" \
            --overwrite

        python clean_meld_omni_captions.py \
            --captions "${RAW_CAPTION_PATH}" \
            --manifest "${MANIFEST_PATH}" \
            --out "${CLEAN_CAPTION_PATH}" \
            --rerun_out "${RERUN_LIST_PATH}" \
            --report_out "${CAPTION_ROOT}/meld_omni_captions.clean.context_audio_target_video_report.json"
    fi

    python validate_meld_omni_captions.py \
        --manifest "${MANIFEST_PATH}" \
        --captions "${CLEAN_CAPTION_PATH}"

    python data_process_meld_omni_caption.py \
        --manifest "${MANIFEST_PATH}" \
        --captions "${CLEAN_CAPTION_PATH}" \
        --out_dir "${CAPTION_DATA_DIR}" \
        --historical_window "${HISTORICAL_WINDOW}" \
        --prompt_style "${PROMPT_STYLE}" \
        --caption_prompt_mode "${CAPTION_PROMPT_MODE}" \
        --use_audio_caption True \
        --use_video_caption True \
        --skip_missing "${SKIP_MISSING_CAPTIONS}"
else
    echo "SKIP_PREPARE=True, reusing existing manifest, clean captions, and processed data."
fi

if [ "${SKIP_TRAINING}" = "True" ]; then
    echo "SKIP_TRAINING=True, preparation finished."
    exit 0
fi

OUTPUT_BASE="../experiments/${MODEL_LABEL}/lora/meld/window_${HISTORICAL_WINDOW}"
RUN_NAME="omni_context_audio_target_video_seed${SEED}"
OUTPUT_DIR="${OUTPUT_BASE}/LR_${LORA_LR}_BS_${BATCH_SIZE}_${RUN_NAME}_run_${RUN_ID}"
LOG_DIR="${OUTPUT_BASE}/logs_context_audio_target_video_${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cat > "${OUTPUT_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "dataset": "meld",
  "model_name": "${MODEL_NAME}",
  "model_label": "${MODEL_LABEL}",
  "model_path": "${MODEL_PATH}",
  "data_source": "omni_caption",
  "data_path": "${CAPTION_DATA_DIR}",
  "caption_path": "${CLEAN_CAPTION_PATH}",
  "caption_format": "context_audio_target_video",
  "caption_prompt_mode": "${CAPTION_PROMPT_MODE}",
  "manifest_path": "${MANIFEST_PATH}",
  "use_audio_caption": "True",
  "use_video_caption": "True",
  "historical_window": "${HISTORICAL_WINDOW}",
  "prompt_style": "${PROMPT_STYLE}",
  "seed": "${SEED}",
  "train_device": "${TRAIN_DEVICE}",
  "master_addr": "${MASTER_ADDR}",
  "master_port": "${MASTER_PORT}",
  "rerun_bad_captions": "${RERUN_BAD_CAPTIONS}",
  "output_dir": "${OUTPUT_DIR}"
}
EOF

echo "Training ${RUN_NAME}; log=${LOG_PATH}"
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
RANK=0 \
LOCAL_RANK=0 \
WORLD_SIZE=1 \
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
    --use_mm_prefix False \
    > "${LOG_PATH}" 2>&1

if [ "${RUN_ANALYSIS}" = "True" ]; then
    python analyze_experiment_results.py \
        --experiments_root ../experiments \
        --out_dir ../experiments/analysis \
        --top_k 30 || true
fi

echo "Finished ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_PATH}"
