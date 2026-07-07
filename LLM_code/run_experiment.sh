#!/usr/bin/env bash
set -e

# Unified Our-MSER runner.
# This keeps the original server baseline training protocol:
# BS=8, gradient accumulation=8, AdamW in main.py, and python main.py launch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

FLAG=1

MODEL_NAME="${MODEL_NAME:-qwen2.5}"
DATASET="${DATASET:-iemocap}"
Experiments_setting="${EXPERIMENTS_SETTING:-lora}"

audio_description="${AUDIO_DESCRIPTION:-True}"
audio_impression="${AUDIO_IMPRESSION:-False}"
audio_context="${AUDIO_CONTEXT:-False}"
audio_only="${AUDIO_ONLY:-False}"
prompt_style="${PROMPT_STYLE:-qwen_chat}"
data_format="${DATA_FORMAT:-prc}"
PRC_DATA_DIR="${PRC_DATA_DIR:-}"

SEED="${SEED:-1}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-15}"
LORA_LR="${LORA_LR:-3e-4}"
use_encoder="${USE_ENCODER:-False}"
LORA_DIM="${LORA_DIM:-32}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_MODULE_NAME="${LORA_MODULE_NAME:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

BS="${BATCH_SIZE:-8}"
accumulations="${GRADIENT_ACCUMULATION_STEPS:-8}"
historical_window="${HISTORICAL_WINDOW:-12}"
data_percent="${DATA_PERCENT:-1.0}"
REPROCESS_DATA="${REPROCESS_DATA:-False}"
PREPROCESS_ONLY="${PREPROCESS_ONLY:-False}"
USE_FEATURE_TEXT="${USE_FEATURE_TEXT:-True}"
USE_MM_PREFIX="${USE_MM_PREFIX:-False}"
MM_AUDIO_FEATURE_DIR="${MM_AUDIO_FEATURE_DIR:-chinese-hubert-large-UTT}"
MM_VIDEO_FEATURE_DIR="${MM_VIDEO_FEATURE_DIR:-clip-vit-large-patch14-UTT}"
MM_AUDIO_TOKENS="${MM_AUDIO_TOKENS:-4}"
MM_VIDEO_TOKENS="${MM_VIDEO_TOKENS:-4}"
MM_PROJECTOR_DROPOUT="${MM_PROJECTOR_DROPOUT:-0.05}"

MODELS_ROOT="${MODELS_ROOT:-/home/pc/jcy/models}"
DATASETS_ROOT="${DATASETS_ROOT:-/home/pc/jcy/datasets}"

case "${MODEL_NAME}" in
    llama2|LLaMA2|llama-2)
        MODEL_LABEL="LLaMA2"
        MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/Llama-2-7b-hf}"
        ;;
    qwen2.5|qwen25|Qwen2.5|Qwen25)
        MODEL_LABEL="Qwen2.5"
        MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/Qwen2.5-7B-Instruct}"
        ;;
    *)
        echo "Invalid MODEL_NAME: ${MODEL_NAME}. Expected llama2 or qwen2.5."
        FLAG=0
        ;;
esac

case "${DATASET}" in
    iemocap|meld) ;;
    *)
        echo "Invalid DATASET: ${DATASET}. Expected iemocap or meld."
        FLAG=0
        ;;
esac

case "${Experiments_setting}" in
    zero_shot|few_shot|lora|all_parameters) ;;
    *)
        echo "Invalid EXPERIMENTS_SETTING: ${Experiments_setting}."
        FLAG=0
        ;;
esac

case "${data_format}" in
    prc|speechcue) ;;
    *)
        echo "Invalid DATA_FORMAT: ${data_format}. Expected prc or speechcue."
        FLAG=0
        ;;
esac

if [ "${FLAG}" = 0 ]; then
    exit 1
fi

if [ -z "${USE_PERSONA+x}" ]; then
    if [ "${DATASET}" = "iemocap" ]; then
        include_persona="True"
    else
        include_persona="False"
    fi
else
    include_persona="${USE_PERSONA}"
fi

if [ -z "${PERSONA_PATH+x}" ]; then
    if [ "${DATASET}" = "meld" ]; then
        persona_path="../persona_features/meld_qwen25_persona.json"
    else
        persona_path="../persona_features/${DATASET}_llama2_persona.json"
    fi
else
    persona_path="${PERSONA_PATH}"
fi
if [ "${include_persona}" = "False" ]; then
    persona_path=""
fi

if [ "${include_persona}" = "True" ] && [ ! -f "${persona_path}" ]; then
    echo "Persona feature file not found: ${persona_path}"
    echo "Set USE_PERSONA=False or provide PERSONA_PATH."
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path not found: ${MODEL_PATH}"
    echo "Create it first. See ../MODEL_SETUP.md."
    exit 1
fi

if [ "${DATASET}" = "iemocap" ]; then
    MAX_LENGTH="${MAX_LENGTH:-2600}"
    MULTIMODAL_MANIFEST_DIR="${MULTIMODAL_MANIFEST_DIR:-${DATASETS_ROOT}/iemocap/features/splits_6way}"
elif [ "${DATASET}" = "meld" ]; then
    MAX_LENGTH="${MAX_LENGTH:-1500}"
    MULTIMODAL_MANIFEST_DIR="${MULTIMODAL_MANIFEST_DIR:-${DATASETS_ROOT}/meld/features/splits_7way}"
fi

task="text"
if [ "${audio_description}" = "True" ]; then
    task="${task}_speech"
fi
if [ "${audio_impression}" = "True" ]; then
    task="${task}_impression"
fi
if [ "${include_persona}" = "True" ]; then
    task="${task}_persona"
fi
if [ "${prompt_style}" != "legacy" ]; then
    task="${task}_${prompt_style}"
fi
if [ -n "${data_format}" ]; then
    task="${task}_${data_format}"
fi
if [ "${USE_MM_PREFIX}" = "True" ]; then
    task="${task}_avprefix_a${MM_AUDIO_TOKENS}_v${MM_VIDEO_TOKENS}"
fi
if [ "${USE_FEATURE_TEXT}" = "True" ]; then
    task="${task}_featuretext"
fi

echo "******************************************************************************************"
echo "Our-MSER experiment"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_LABEL}"
echo "Model path: ${MODEL_PATH}"
echo "SFT method: ${Experiments_setting}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "Batch size: ${BS}"
echo "Gradient accumulation steps: ${accumulations}"
echo "Epochs: ${num_train_epochs}"
echo "Historical window: ${historical_window}"
echo "Max context length: ${MAX_LENGTH}"
echo "Audio description: ${audio_description}"
echo "Audio impression: ${audio_impression}"
echo "Audio context: ${audio_context}"
echo "Prompt style: ${prompt_style}"
echo "Data format: ${data_format}"
echo "PRC data dir: ${PRC_DATA_DIR}"
echo "Include persona: ${include_persona}"
echo "Persona path: ${persona_path}"
echo "LoRA dim: ${LORA_DIM}"
echo "LoRA alpha: ${LORA_ALPHA}"
echo "LoRA dropout: ${LORA_DROPOUT}"
echo "LoRA modules: ${LORA_MODULE_NAME}"
echo "Data percent: ${data_percent}"
echo "Reprocess data: ${REPROCESS_DATA}"
echo "Preprocess only: ${PREPROCESS_ONLY}"
echo "Use feature text: ${USE_FEATURE_TEXT}"
echo "Use multimodal prefix: ${USE_MM_PREFIX}"
echo "Multimodal manifest dir: ${MULTIMODAL_MANIFEST_DIR}"
echo "MM audio feature dir: ${MM_AUDIO_FEATURE_DIR}"
echo "MM video feature dir: ${MM_VIDEO_FEATURE_DIR}"
echo "MM audio tokens: ${MM_AUDIO_TOKENS}"
echo "MM video tokens: ${MM_VIDEO_TOKENS}"
echo "******************************************************************************************"

persona_tag=""
if [ "${include_persona}" = "True" ]; then
    persona_tag="_persona"
fi
prompt_tag=""
if [ -n "${prompt_style}" ]; then
    prompt_tag="_${prompt_style}"
fi
format_tag=""
if [ -n "${data_format}" ]; then
    format_tag="_${data_format}"
fi
feature_text_tag=""
if [ "${USE_FEATURE_TEXT}" = "True" ]; then
    feature_text_tag="_featuretext"
fi
DATA_PATH="../PROCESSED_DATASET/${DATASET}/window/${audio_description}_${audio_impression}${persona_tag}${prompt_tag}${format_tag}${feature_text_tag}"

if [ "${REPROCESS_DATA}" = "True" ] || [ ! -f "${DATA_PATH}/train.json" ] || [ ! -f "${DATA_PATH}/test.json" ] || [ ! -f "${DATA_PATH}/valid.json" ]; then
    DATA_PATH=$(python data_process.py \
        --dataset "${DATASET}" \
        --historical_window "${historical_window}" \
        --audio_description "${audio_description}" \
        --audio_impression "${audio_impression}" \
        --audio_only "${audio_only}" \
        --audio_context "${audio_context}" \
        --experiments_setting "${Experiments_setting}" \
        --include_persona "${include_persona}" \
        --persona_path "${persona_path}" \
        --prompt_style "${prompt_style}" \
        --data_format "${data_format}" \
        --prc_data_dir "${PRC_DATA_DIR}" \
        --use_feature_text "${USE_FEATURE_TEXT}" \
        --text_manifest_dir "${MULTIMODAL_MANIFEST_DIR}")
    DATA_ACTION="generated"
else
    DATA_ACTION="reused"
fi

echo "******************************************************************************************"
echo "Processed data ${DATA_ACTION} successfully!"
echo "Processed Data_Path: ${DATA_PATH}"
echo "******************************************************************************************"

if [ "${PREPROCESS_ONLY}" = "True" ]; then
    exit 0
fi

if [ "${Experiments_setting}" = "zero_shot" ]; then
    DO_EVAL=True
    DO_TRAIN=False
    LORA=False
    LR=0
elif [ "${Experiments_setting}" = "few_shot" ]; then
    DO_EVAL=True
    DO_TRAIN=False
    LORA=False
    LR=0
elif [ "${Experiments_setting}" = "lora" ]; then
    DO_EVAL=True
    DO_TRAIN=True
    LORA=True
    LR=${LORA_LR}
elif [ "${Experiments_setting}" = "all_parameters" ]; then
    DO_EVAL=True
    DO_TRAIN=True
    LORA=False
    LR=2e-5
fi

OUTPUT_DIR="../experiments/${MODEL_LABEL}/${Experiments_setting}/${DATASET}/window_${historical_window}/LR_${LR}_BS_${BS}_per_${data_percent}_${task}_class5_${SEED}_single_gpu_e15"

echo "******************************************************************************************"
echo "Start running main.py without DeepSpeed launcher"
echo "Output dir: ${OUTPUT_DIR}"
echo "******************************************************************************************"

python main.py \
    --dataset "${DATASET}" \
    --model_name_or_path "${MODEL_PATH}" \
    --data_dir "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BS}" \
    --gradient_accumulation_steps "${accumulations}" \
    --eval_batch_size 1 \
    --num_train_epochs "${num_train_epochs}" \
    --save_steps 100000 \
    --lora "${LORA}" \
    --lora_dim "${LORA_DIM}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_module_name "${LORA_MODULE_NAME}" \
    --learning_rate "${LR}" \
    --do_eval "${DO_EVAL}" \
    --do_train "${DO_TRAIN}" \
    --statistic_mode True \
    --data_percent "${data_percent}" \
    --seed "${SEED}" \
    --use_mm_prefix "${USE_MM_PREFIX}" \
    --multimodal_manifest_dir "${MULTIMODAL_MANIFEST_DIR}" \
    --mm_audio_feature_dir "${MM_AUDIO_FEATURE_DIR}" \
    --mm_video_feature_dir "${MM_VIDEO_FEATURE_DIR}" \
    --mm_audio_tokens "${MM_AUDIO_TOKENS}" \
    --mm_video_tokens "${MM_VIDEO_TOKENS}" \
    --mm_projector_dropout "${MM_PROJECTOR_DROPOUT}"
