#!/usr/bin/env bash
set -u -o pipefail

# Regenerate emotion features on GPU0/GPU1 in parallel, rebuild retrieval
# prompts, then train IEMOCAP and MELD sequentially with three GPUs each.
# This launcher writes to a separate run directory and does not overwrite
# previous training outputs.

PROJECT_ROOT="${PROJECT_ROOT:-/home/pc/jcy/Our-MSER/PRC-Emo-reproduction}"
MODEL_PATH="${MODEL_PATH:-/home/pc/jcy/models/Qwen2.5-7B-Instruct}"
GPU_IEMOCAP="${GPU_IEMOCAP:-0}"
GPU_MELD="${GPU_MELD:-1}"
TRAIN_GPUS=("${GPU0:-0}" "${GPU1:-1}" "${GPU2:-2}")
PROMPT_GPU="${PROMPT_GPU:-0}"

WINDOW="${WINDOW:-5}"
PROMPT_TYPE="${PROMPT_TYPE:-ImplicitEmotion_V3}"
EXTRACT_LLM_ID="${EXTRACT_LLM_ID:-Qwen2.5-7B-Instruct}"
SPEAKER_FEATURE_TYPE="${SPEAKER_FEATURE_TYPE:-spdescV6}"
SPEAKER_FEATURE_SUFFIX="${SPEAKER_FEATURE_SUFFIX:-${SPEAKER_FEATURE_TYPE}_${EXTRACT_LLM_ID}}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-3e-4}"
LORA_R="${LORA_R:-32}"
BUCKETS="${BUCKETS:-2}"
CURRICULUM_UPDATE_EPOCHS="${CURRICULUM_UPDATE_EPOCHS:-1}"
SEEDS=(42 43 44)

# The validated smoke test used deterministic generation. Override these
# variables if sampling is required for a separate comparison.
EMOTION_FORCE_REGEN="${PRC_EMO_FORCE_REGEN:-1}"
EMOTION_DO_SAMPLE="${PRC_EMO_DO_SAMPLE:-0}"
EMOTION_INPUT_MAX_LENGTH="${PRC_EMO_INPUT_MAX_LENGTH:-2048}"
EMOTION_MAX_NEW_TOKENS="${PRC_EMO_MAX_NEW_TOKENS:-256}"
EMOTION_BATCH_SIZE="${PRC_EMO_BATCH_SIZE:-2}"

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/features_then_train}"
OUTPUT_FOLDER="$RUN_ROOT/finetuned_llm"
LOG_ROOT="$PROJECT_ROOT/logs/features_then_train"
STATUS_FILE="$RUN_ROOT/run_status.tsv"
PIPELINE_STATUS_FILE="$RUN_ROOT/pipeline_status.tsv"
REPORT_FILE="$RUN_ROOT/RESULTS.md"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$LOG_ROOT" "$OUTPUT_FOLDER"
printf 'phase\tdataset\tseed\tgpu\texit_code\tlog\toutput_dir\n' > "$STATUS_FILE"
printf 'phase\tdataset\tgpu\texit_code\tlog\n' > "$PIPELINE_STATUS_FILE"

run_logged() {
    local label="$1"
    shift
    local log_file="$LOG_ROOT/${label}.log"
    echo "[$(date '+%F %T')] START $label" | tee "$log_file"
    "$@" >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" | tee -a "$log_file"
    return "$code"
}

record_pipeline_status() {
    local phase="$1"
    local dataset="$2"
    local gpu="$3"
    local code="$4"
    local log_file="$5"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$dataset" "$gpu" "$code" "$log_file" >> "$PIPELINE_STATUS_FILE"
}

all_files_ready() {
    local path
    for path in "$@"; do
        if [[ ! -s "$path" ]]; then
            return 1
        fi
    done
    return 0
}

emotion_files() {
    local dataset="$1"
    local model_basename="${MODEL_PATH##*/}"
    printf '%s\n' \
        "$PROJECT_ROOT/data/${dataset}.train_ImplicitEmotion_V3_${model_basename}.json" \
        "$PROJECT_ROOT/data/${dataset}.valid_ImplicitEmotion_V3_${model_basename}.json" \
        "$PROJECT_ROOT/data/${dataset}.test_ImplicitEmotion_V3_${model_basename}.json"
}

speaker_files() {
    local dataset="$1"
    local model_basename="${MODEL_PATH##*/}"
    printf '%s\n' \
        "$PROJECT_ROOT/data/${dataset}.train_${SPEAKER_FEATURE_TYPE}_${model_basename}.json" \
        "$PROJECT_ROOT/data/${dataset}.valid_${SPEAKER_FEATURE_TYPE}_${model_basename}.json" \
        "$PROJECT_ROOT/data/${dataset}.test_${SPEAKER_FEATURE_TYPE}_${model_basename}.json"
}

prompt_files() {
    local dataset="$1"
    printf '%s\n' \
        "$PROJECT_ROOT/data/${dataset}.train.0shot_w${WINDOW}_${PROMPT_TYPE}.jsonl" \
        "$PROJECT_ROOT/data/${dataset}.valid.0shot_w${WINDOW}_${PROMPT_TYPE}.jsonl" \
        "$PROJECT_ROOT/data/${dataset}.test.0shot_w${WINDOW}_${PROMPT_TYPE}.jsonl"
}

backup_existing_emotion_files() {
    local dataset="$1"
    local path
    while IFS= read -r path; do
        if [[ -f "$path" ]]; then
            cp -f "$path" "${path}.bak_before_features_then_train"
        fi
    done < <(emotion_files "$dataset")
}

generate_emotion_features() {
    local dataset="$1"
    local gpu="$2"
    local label="${dataset}_emotion_features_gpu${gpu}"
    local log_file="$LOG_ROOT/${label}.log"

    backup_existing_emotion_files "$dataset"
    if ! run_logged "$label" env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PRC_EMO_DATASET="$dataset" \
        PRC_EMO_DATA_FOLDER="$PROJECT_ROOT/data" \
        PRC_EMO_MODEL_PATH="$MODEL_PATH" \
        PRC_EMO_SPLITS="valid,test,train" \
        PRC_EMO_MAX_UTTERANCES=0 \
        PRC_EMO_FORCE_REGEN="$EMOTION_FORCE_REGEN" \
        PRC_EMO_OUTPUT_SUFFIX="" \
        PRC_EMO_DO_SAMPLE="$EMOTION_DO_SAMPLE" \
        PRC_EMO_INPUT_MAX_LENGTH="$EMOTION_INPUT_MAX_LENGTH" \
        PRC_EMO_MAX_NEW_TOKENS="$EMOTION_MAX_NEW_TOKENS" \
        PRC_EMO_BATCH_SIZE="$EMOTION_BATCH_SIZE" \
        python src/llm_emotion_extract_v2.py; then
        record_pipeline_status emotion_features "$dataset" "$gpu" 1 "$log_file"
        return 1
    fi

    local generated_files=()
    mapfile -t generated_files < <(emotion_files "$dataset")
    if ! all_files_ready "${generated_files[@]}"; then
        echo "Emotion feature files are missing or empty for $dataset" | tee -a "$log_file"
        record_pipeline_status emotion_features "$dataset" "$gpu" 1 "$log_file"
        return 1
    fi
    record_pipeline_status emotion_features "$dataset" "$gpu" 0 "$log_file"
    return 0
}

prepare_prompts() {
    local dataset="$1"
    local label="${dataset}_prompt_preprocessing"
    local log_file="$LOG_ROOT/${label}.log"
    if ! run_logged "$label" env \
        CUDA_VISIBLE_DEVICES="$PROMPT_GPU" \
        PRC_EMO_EMBEDDING_MODEL_PATH="${PRC_EMO_EMBEDDING_MODEL_PATH:-$PROJECT_ROOT/models/all-MiniLM-L6-v2}" \
        python scripts/generate_prompt_data.py \
        --dataset "$dataset" \
        --window "$WINDOW" \
        --prompting_type "$PROMPT_TYPE" \
        --extract_prompting_llm_id "$EXTRACT_LLM_ID" \
        --speaker_feature_suffix "$SPEAKER_FEATURE_SUFFIX"; then
        record_pipeline_status prompt_preprocessing "$dataset" "$PROMPT_GPU" 1 "$log_file"
        return 1
    fi

    local generated_files=()
    mapfile -t generated_files < <(prompt_files "$dataset")
    if ! all_files_ready "${generated_files[@]}"; then
        echo "Prompt files are missing or empty for $dataset" | tee -a "$log_file"
        record_pipeline_status prompt_preprocessing "$dataset" "$PROMPT_GPU" 1 "$log_file"
        return 1
    fi
    record_pipeline_status prompt_preprocessing "$dataset" "$PROMPT_GPU" 0 "$log_file"
    return 0
}

run_train() {
    local dataset="$1"
    local seed="$2"
    local gpu="$3"
    local output_id="${dataset}_qwen25_${PROMPT_TYPE}_features_then_train_seed${seed}"
    local output_dir="$OUTPUT_FOLDER/$output_id"
    local log_file="$LOG_ROOT/${dataset}_seed${seed}_gpu${gpu}.log"

    echo "[$(date '+%F %T')] START train dataset=$dataset seed=$seed gpu=$gpu" > "$log_file"
    env CUDA_VISIBLE_DEVICES="$gpu" \
        PRC_EMO_ATTN_IMPL="${PRC_EMO_ATTN_IMPL:-sdpa}" \
        python src/ft_llm_cl.py \
        --do_train --do_eval_dev --do_eval_test --curriculum \
        --bucket_number "$BUCKETS" \
        --curriculum_update_epochs "$CURRICULUM_UPDATE_EPOCHS" \
        --base_model_id "$MODEL_PATH" \
        --ft_model_id "$output_id" \
        --output_folder "$OUTPUT_FOLDER" \
        --lr_scheduler linear \
        --lr "$LR" \
        --lora_r "$LORA_R" \
        --max_steps -1 \
        --epoch "$EPOCHS" \
        --kshot 0 \
        --window "$WINDOW" \
        --data_name "$dataset" \
        --prompting_type "$PROMPT_TYPE" \
        --extract_prompting_llm_id "$EXTRACT_LLM_ID" \
        --seed "$seed" \
        --max_seq_len 2048 \
        --eval_delay 100000 \
        --data_folder "$PROJECT_ROOT/data" \
        >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" >> "$log_file"
    printf 'train\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$seed" "$gpu" "$code" "$log_file" "$output_dir" >> "$STATUS_FILE"
    return "$code"
}

run_dataset() {
    local dataset="$1"
    local pids=()
    echo "===== Starting parallel training: $dataset ====="
    for i in 0 1 2; do
        run_train "$dataset" "${SEEDS[$i]}" "${TRAIN_GPUS[$i]}" &
        pids+=("$!")
    done
    local overall=0
    for pid in "${pids[@]}"; do
        wait "$pid" || overall=1
    done
    if [[ "$overall" -ne 0 ]]; then
        echo "At least one $dataset training process failed; continuing to the next dataset." >&2
    fi
    return "$overall"
}

echo "===== Parallel emotion feature generation ====="
echo "IEMOCAP -> GPU $GPU_IEMOCAP; MELD -> GPU $GPU_MELD"

generate_emotion_features iemocap "$GPU_IEMOCAP" &
iemocap_feature_pid=$!
generate_emotion_features meld "$GPU_MELD" &
meld_feature_pid=$!

feature_failure=0
wait "$iemocap_feature_pid" || feature_failure=1
wait "$meld_feature_pid" || feature_failure=1
if [[ "$feature_failure" -ne 0 ]]; then
    echo "Emotion feature generation failed; training will not start." >&2
    exit 1
fi

for dataset in iemocap meld; do
    speaker_ready=()
    mapfile -t speaker_ready < <(speaker_files "$dataset")
    if ! all_files_ready "${speaker_ready[@]}"; then
        echo "Missing speaker feature files for $dataset. Existing spdescV6 files are required." >&2
        exit 1
    fi
done

echo "===== Rebuilding retrieval-augmented prompts ====="
if ! prepare_prompts iemocap; then
    echo "IEMOCAP prompt preprocessing failed; training will not start." >&2
    exit 1
fi
if ! prepare_prompts meld; then
    echo "MELD prompt preprocessing failed; training will not start." >&2
    exit 1
fi

echo "===== Training IEMOCAP first, then MELD; three GPUs per dataset ====="
run_dataset iemocap || true
run_dataset meld || true

python scripts/summarize_results.py \
    --run_root "$RUN_ROOT" \
    --status "$STATUS_FILE" \
    --output "$REPORT_FILE"

echo "Finished. Report: $REPORT_FILE"
echo "Pipeline status: $PIPELINE_STATUS_FILE"
