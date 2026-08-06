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
CLEAN_CAPTION_PATH="${CLEAN_CAPTION_PATH:-${CAPTION_ROOT}/meld_omni_captions.clean.jsonl}"
CUELIST_CAPTION_PATH="${CUELIST_CAPTION_PATH:-${CAPTION_ROOT}/meld_omni_captions.cuelist.jsonl}"

HISTORICAL_WINDOW="${HISTORICAL_WINDOW:-12}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_chat}"
SKIP_MISSING_CAPTIONS="${SKIP_MISSING_CAPTIONS:-True}"
USE_AUDIO_CAPTION="${USE_AUDIO_CAPTION:-True}"
USE_VIDEO_CAPTION="${USE_VIDEO_CAPTION:-True}"

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
TRAIN_DEVICES="${TRAIN_DEVICES:-0 1 2}"
TRAIN_PORTS="${TRAIN_PORTS:-29600 29601 29602}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
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

read -r -a DEVICE_ARRAY <<< "${TRAIN_DEVICES}"
read -r -a PORT_ARRAY <<< "${TRAIN_PORTS}"
if [ "${#DEVICE_ARRAY[@]}" -lt 3 ]; then
    echo "TRAIN_DEVICES must contain at least three devices, for example: TRAIN_DEVICES=\"0 1 2\""
    exit 1
fi
if [ "${#PORT_ARRAY[@]}" -lt 3 ]; then
    echo "TRAIN_PORTS must contain at least three ports, for example: TRAIN_PORTS=\"29600 29601 29602\""
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${RAW_CAPTION_PATH}" ]; then
    echo "Raw caption file not found: ${RAW_CAPTION_PATH}"
    echo "Run generate_meld_omni_captions.py first, or set RAW_CAPTION_PATH to an existing caption JSONL."
    exit 1
fi

echo "******************************************************************************************"
echo "MELD caption ablation 3-GPU run"
echo "Run id: ${RUN_ID}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Raw captions: ${RAW_CAPTION_PATH}"
echo "Clean captions: ${CLEAN_CAPTION_PATH}"
echo "Cue-list captions: ${CUELIST_CAPTION_PATH}"
echo "Train devices: ${TRAIN_DEVICES}"
echo "Train ports: ${TRAIN_PORTS}"
echo "Train model: ${MODEL_PATH}"
echo "******************************************************************************************"

bash prepare_meld_raw_data.sh "${RAW_BASE}"

python build_meld_caption_manifest.py \
    --raw_root "${RAW_ROOT}" \
    --out "${MANIFEST_PATH}"

python clean_meld_omni_captions.py \
    --captions "${RAW_CAPTION_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --out "${CLEAN_CAPTION_PATH}" \
    --rerun_out "${CAPTION_ROOT}/meld_omni_captions.rerun_needed.jsonl" \
    --report_out "${CAPTION_ROOT}/meld_omni_captions.clean_report.json"

python validate_meld_omni_captions.py \
    --manifest "${MANIFEST_PATH}" \
    --captions "${CLEAN_CAPTION_PATH}"

python convert_meld_captions_to_cuelist.py \
    --captions "${CLEAN_CAPTION_PATH}" \
    --out "${CUELIST_CAPTION_PATH}" \
    --report_out "${CAPTION_ROOT}/meld_omni_captions.cuelist_report.json"

python validate_meld_omni_captions.py \
    --manifest "${MANIFEST_PATH}" \
    --captions "${CUELIST_CAPTION_PATH}"

DATA_ROOT="../PROCESSED_DATASET/meld/omni_caption_prompt"
CLEAN_SEPARATE_DIR="${DATA_ROOT}/window_${HISTORICAL_WINDOW}_${PROMPT_STYLE}_clean_separate"
CUELIST_SEPARATE_DIR="${DATA_ROOT}/window_${HISTORICAL_WINDOW}_${PROMPT_STYLE}_cuelist_separate"
CUELIST_UNIFIED_DIR="${DATA_ROOT}/window_${HISTORICAL_WINDOW}_${PROMPT_STYLE}_cuelist_unified"

process_data() {
    local caption_path="$1"
    local out_dir="$2"
    local caption_prompt_mode="$3"
    python data_process_meld_omni_caption.py \
        --manifest "${MANIFEST_PATH}" \
        --captions "${caption_path}" \
        --out_dir "${out_dir}" \
        --historical_window "${HISTORICAL_WINDOW}" \
        --prompt_style "${PROMPT_STYLE}" \
        --caption_prompt_mode "${caption_prompt_mode}" \
        --use_audio_caption "${USE_AUDIO_CAPTION}" \
        --use_video_caption "${USE_VIDEO_CAPTION}" \
        --skip_missing "${SKIP_MISSING_CAPTIONS}"
}

process_data "${CLEAN_CAPTION_PATH}" "${CLEAN_SEPARATE_DIR}" "separate_fields"
process_data "${CUELIST_CAPTION_PATH}" "${CUELIST_SEPARATE_DIR}" "separate_fields"
process_data "${CUELIST_CAPTION_PATH}" "${CUELIST_UNIFIED_DIR}" "unified_cues"

if [ "${SKIP_TRAINING}" = "True" ]; then
    echo "SKIP_TRAINING=True, data preparation finished."
    exit 0
fi

OUTPUT_BASE="../experiments/${MODEL_LABEL}/lora/meld/window_${HISTORICAL_WINDOW}"
LOG_DIR="${OUTPUT_BASE}/parallel_logs_${RUN_ID}"
mkdir -p "${LOG_DIR}"

pids=()
run_names=()

cleanup() {
    if [ "${#pids[@]}" -gt 0 ]; then
        for pid in "${pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" 2>/dev/null || true
            fi
        done
    fi
}
trap cleanup INT TERM

launch_train() {
    local device="$1"
    local port="$2"
    local run_name="$3"
    local data_dir="$4"
    local caption_path="$5"
    local caption_format="$6"
    local caption_prompt_mode="$7"
    local output_dir="${OUTPUT_BASE}/LR_${LORA_LR}_BS_${BATCH_SIZE}_${run_name}_${SEED}_run_${RUN_ID}"
    local log_path="${LOG_DIR}/${run_name}.log"

    mkdir -p "${output_dir}"
    cat > "${output_dir}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "dataset": "meld",
  "model_name": "${MODEL_NAME}",
  "model_label": "${MODEL_LABEL}",
  "model_path": "${MODEL_PATH}",
  "data_source": "omni_caption",
  "data_path": "${data_dir}",
  "caption_path": "${caption_path}",
  "caption_format": "${caption_format}",
  "caption_prompt_mode": "${caption_prompt_mode}",
  "manifest_path": "${MANIFEST_PATH}",
  "use_audio_caption": "${USE_AUDIO_CAPTION}",
  "use_video_caption": "${USE_VIDEO_CAPTION}",
  "historical_window": "${HISTORICAL_WINDOW}",
  "prompt_style": "${PROMPT_STYLE}",
  "seed": "${SEED}",
  "train_device": "${device}",
  "master_addr": "${MASTER_ADDR}",
  "master_port": "${port}",
  "output_dir": "${output_dir}"
}
EOF

    echo "Launching ${run_name} on GPU ${device}, MASTER_PORT=${port}"
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT="${port}" \
    RANK=0 \
    LOCAL_RANK=0 \
    WORLD_SIZE=1 \
    CUDA_VISIBLE_DEVICES="${device}" python main.py \
        --dataset meld \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
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
        > "${log_path}" 2>&1 &

    local pid="$!"
    pids+=("${pid}")
    run_names+=("${run_name}")
    echo "${run_name}: pid=${pid} log=${log_path} output=${output_dir}"
}

launch_train "${DEVICE_ARRAY[0]}" "${PORT_ARRAY[0]}" "omni_clean_separate" "${CLEAN_SEPARATE_DIR}" "${CLEAN_CAPTION_PATH}" "clean_natural" "separate_fields"
launch_train "${DEVICE_ARRAY[1]}" "${PORT_ARRAY[1]}" "omni_cuelist_separate" "${CUELIST_SEPARATE_DIR}" "${CUELIST_CAPTION_PATH}" "cue_list" "separate_fields"
launch_train "${DEVICE_ARRAY[2]}" "${PORT_ARRAY[2]}" "omni_cuelist_unified" "${CUELIST_UNIFIED_DIR}" "${CUELIST_CAPTION_PATH}" "cue_list" "unified_cues"

status=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    run_name="${run_names[$idx]}"
    if wait "${pid}"; then
        echo "Finished ${run_name}"
    else
        echo "Failed ${run_name}; check ${LOG_DIR}/${run_name}.log"
        status=1
    fi
done

if [ "${RUN_ANALYSIS}" = "True" ]; then
    python analyze_experiment_results.py \
        --experiments_root ../experiments \
        --out_dir ../experiments/analysis \
        --top_k 30 || true
fi

exit "${status}"
