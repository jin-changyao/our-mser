#!/usr/bin/env bash
set -u -o pipefail

# Server entry point for the Qwen2.5 PRC-Emo reproduction.
# Default paths follow our-mser/MODEL_SETUP.md. Override with environment
# variables when the server mount point is different.
PROJECT_ROOT="${PROJECT_ROOT:-/home/pc/jcy/Our-MSER/PRC-Emo-reproduction}"
MODEL_PATH="${MODEL_PATH:-/home/pc/jcy/models/Qwen2.5-7B-Instruct}"
GPUS=("${GPU0:-0}" "${GPU1:-1}" "${GPU2:-2}")
GEN_GPU="${GEN_GPU:-0}"
WINDOW="${WINDOW:-5}"
PROMPT_TYPE="${PROMPT_TYPE:-ImplicitEmotion_V3}"
EXTRACT_LLM_ID="${EXTRACT_LLM_ID:-Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-3e-4}"
LORA_R="${LORA_R:-32}"
BUCKETS="${BUCKETS:-2}"
CURRICULUM_UPDATE_EPOCHS="${CURRICULUM_UPDATE_EPOCHS:-1}"
SEEDS=(42 43 44)

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs runs/finetuned_llm
STATUS_FILE="runs/run_status.tsv"
REPORT_FILE="runs/RESULTS.md"
printf 'phase\tdataset\tseed\tgpu\texit_code\tlog\toutput_dir\n' > "$STATUS_FILE"

run_logged() {
    local label="$1"
    shift
    local log_file="logs/${label}.log"
    echo "[$(date '+%F %T')] START $label" | tee "$log_file"
    "$@" >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" | tee -a "$log_file"
    return "$code"
}

generate_features() {
    local dataset="$1"
    echo "===== Generating Qwen2.5 emotion descriptions: $dataset on GPU $GEN_GPU ====="
    if ! run_logged "${dataset}_emotion_features" env \
        CUDA_VISIBLE_DEVICES="$GEN_GPU" \
        PRC_EMO_DATASET="$dataset" \
        PRC_EMO_DATA_FOLDER="$PROJECT_ROOT/data" \
        PRC_EMO_MODEL_PATH="$MODEL_PATH" \
        python src/llm_emotion_extract_v2.py; then
        echo "Feature generation failed for $dataset; see logs/${dataset}_emotion_features.log" >&2
        exit 1
    fi

    echo "===== Generating Qwen2.5 speaker profiles: $dataset on GPU $GEN_GPU ====="
    if ! run_logged "${dataset}_speaker_features" env \
        CUDA_VISIBLE_DEVICES="$GEN_GPU" \
        PRC_EMO_DATASET="$dataset" \
        PRC_EMO_DATA_FOLDER="$PROJECT_ROOT/data" \
        PRC_EMO_MODEL_PATH="$MODEL_PATH" \
        python src/llm_bio_extract_v2.py; then
        echo "Speaker feature generation failed for $dataset; see logs/${dataset}_speaker_features.log" >&2
        exit 1
    fi
}

prepare_prompts() {
    local dataset="$1"
    echo "===== Building retrieval-augmented prompts: $dataset on GPU $GEN_GPU ====="
    if ! run_logged "${dataset}_prompt_preprocessing" env \
        CUDA_VISIBLE_DEVICES="$GEN_GPU" \
        python scripts/generate_prompt_data.py \
        --dataset "$dataset" \
        --window "$WINDOW" \
        --prompting_type "$PROMPT_TYPE" \
        --extract_prompting_llm_id "$EXTRACT_LLM_ID"; then
        echo "Prompt preprocessing failed for $dataset; see logs/${dataset}_prompt_preprocessing.log" >&2
        exit 1
    fi
}

run_train() {
    local dataset="$1"
    local seed="$2"
    local gpu="$3"
    local output_id="${dataset}_qwen25_${PROMPT_TYPE}_seed${seed}"
    local output_dir="$PROJECT_ROOT/runs/finetuned_llm/$output_id"
    local log_file="logs/${dataset}_seed${seed}_gpu${gpu}.log"

    echo "[$(date '+%F %T')] START train dataset=$dataset seed=$seed gpu=$gpu" > "$log_file"
    env CUDA_VISIBLE_DEVICES="$gpu" python src/ft_llm_cl.py \
        --do_train --do_eval_dev --do_eval_test --curriculum \
        --bucket_number "$BUCKETS" \
        --curriculum_update_epochs "$CURRICULUM_UPDATE_EPOCHS" \
        --base_model_id "$MODEL_PATH" \
        --ft_model_id "$output_id" \
        --output_folder "$PROJECT_ROOT/runs/finetuned_llm" \
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
        run_train "$dataset" "${SEEDS[$i]}" "${GPUS[$i]}" &
        pids+=("$!")
    done
    local overall=0
    for pid in "${pids[@]}"; do
        wait "$pid" || overall=1
    done
    if [[ "$overall" -ne 0 ]]; then
        echo "At least one $dataset training process failed; continuing to the next dataset." >&2
    fi
}

# GPU0 is used serially for all non-training generation and prompt assembly.
generate_features iemocap
generate_features meld
prepare_prompts iemocap
prepare_prompts meld

# Run IEMOCAP first, then MELD. Each dataset uses three cards concurrently.
run_dataset iemocap
run_dataset meld

python scripts/summarize_results.py \
    --run_root "$PROJECT_ROOT/runs" \
    --status "$STATUS_FILE" \
    --output "$REPORT_FILE"

echo "Finished. Report: $REPORT_FILE"
