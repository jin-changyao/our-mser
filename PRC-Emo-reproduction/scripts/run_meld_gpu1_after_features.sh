#!/usr/bin/env bash
set -u -o pipefail

# MELD-only continuation after emotion features have been generated/repaired.
# GPU1 rebuilds the retrieval prompts and sequentially runs the requested
# training seeds. No emotion feature generation is performed here.

PROJECT_ROOT="${PROJECT_ROOT:-/home/pc/jcy/Our-MSER/PRC-Emo-reproduction}"
MODEL_PATH="${MODEL_PATH:-/home/pc/jcy/models/Qwen2.5-7B-Instruct}"
GPU="${GPU:-1}"
PROMPT_GPU="${PROMPT_GPU:-$GPU}"
WINDOW="${WINDOW:-5}"
PROMPT_TYPE="${PROMPT_TYPE:-ImplicitEmotion_V3}"
EXTRACT_LLM_ID="${EXTRACT_LLM_ID:-Qwen2.5-7B-Instruct}"
SPEAKER_FEATURE_TYPE="${SPEAKER_FEATURE_TYPE:-spdescV6}"
SEEDS="${SEEDS:-42,43,44}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-3e-4}"
LORA_R="${LORA_R:-32}"
BUCKETS="${BUCKETS:-2}"
CURRICULUM_UPDATE_EPOCHS="${CURRICULUM_UPDATE_EPOCHS:-1}"

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/meld_gpu1_after_features}"
OUTPUT_FOLDER="$RUN_ROOT/finetuned_llm"
LOG_ROOT="$PROJECT_ROOT/logs/meld_gpu1_after_features"
STATUS_FILE="$RUN_ROOT/run_status.tsv"
REPORT_FILE="$RUN_ROOT/RESULTS.md"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$LOG_ROOT" "$OUTPUT_FOLDER"
printf 'phase\tdataset\tseed\tgpu\texit_code\tlog\toutput_dir\n' > "$STATUS_FILE"

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

all_files_ready() {
    local path
    for path in "$@"; do
        [[ -s "$path" ]] || return 1
    done
    return 0
}

emotion_files=(
    "$PROJECT_ROOT/data/meld.train_ImplicitEmotion_V3_${MODEL_PATH##*/}.json"
    "$PROJECT_ROOT/data/meld.valid_ImplicitEmotion_V3_${MODEL_PATH##*/}.json"
    "$PROJECT_ROOT/data/meld.test_ImplicitEmotion_V3_${MODEL_PATH##*/}.json"
)
speaker_files=(
    "$PROJECT_ROOT/data/meld.train_${SPEAKER_FEATURE_TYPE}_${MODEL_PATH##*/}.json"
    "$PROJECT_ROOT/data/meld.valid_${SPEAKER_FEATURE_TYPE}_${MODEL_PATH##*/}.json"
    "$PROJECT_ROOT/data/meld.test_${SPEAKER_FEATURE_TYPE}_${MODEL_PATH##*/}.json"
)

if ! all_files_ready "${emotion_files[@]}"; then
    echo "MELD emotion feature files are missing or empty." >&2
    exit 1
fi
if ! all_files_ready "${speaker_files[@]}"; then
    echo "MELD speaker feature files are missing or empty." >&2
    exit 1
fi

prompt_label="meld_prompt_preprocessing_gpu${PROMPT_GPU}"
if ! run_logged "$prompt_label" env \
    CUDA_VISIBLE_DEVICES="$PROMPT_GPU" \
    PRC_EMO_EMBEDDING_MODEL_PATH="${PRC_EMO_EMBEDDING_MODEL_PATH:-$PROJECT_ROOT/models/all-MiniLM-L6-v2}" \
    python scripts/generate_prompt_data.py \
    --dataset meld \
    --window "$WINDOW" \
    --prompting_type "$PROMPT_TYPE" \
    --extract_prompting_llm_id "$EXTRACT_LLM_ID"; then
    echo "MELD prompt preprocessing failed; training will not start." >&2
    exit 1
fi

prompt_files=(
    "$PROJECT_ROOT/data/meld.train.0shot_w${WINDOW}_${PROMPT_TYPE}.jsonl"
    "$PROJECT_ROOT/data/meld.valid.0shot_w${WINDOW}_${PROMPT_TYPE}.jsonl"
    "$PROJECT_ROOT/data/meld.test.0shot_w${WINDOW}_${PROMPT_TYPE}.jsonl"
)
if ! all_files_ready "${prompt_files[@]}"; then
    echo "MELD prompt files are missing or empty." >&2
    exit 1
fi

overall=0
for seed in ${SEEDS//,/ }; do
    output_id="meld_qwen25_${PROMPT_TYPE}_gpu${GPU}_after_feature_repair_seed${seed}"
    output_dir="$OUTPUT_FOLDER/$output_id"
    log_file="$LOG_ROOT/meld_seed${seed}_gpu${GPU}.log"
    echo "[$(date '+%F %T')] START train dataset=meld seed=$seed gpu=$GPU" > "$log_file"

    env CUDA_VISIBLE_DEVICES="$GPU" \
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
        --data_name meld \
        --prompting_type "$PROMPT_TYPE" \
        --extract_prompting_llm_id "$EXTRACT_LLM_ID" \
        --seed "$seed" \
        --max_seq_len 2048 \
        --eval_delay 100000 \
        --data_folder "$PROJECT_ROOT/data" \
        >> "$log_file" 2>&1
    code=$?
    echo "[$(date '+%F %T')] EXIT $code" >> "$log_file"
    printf 'train\tmeld\t%s\t%s\t%s\t%s\t%s\n' \
        "$seed" "$GPU" "$code" "$log_file" "$output_dir" >> "$STATUS_FILE"
    if [[ "$code" -ne 0 ]]; then
        overall=1
        echo "MELD seed $seed failed; continuing with the next seed." >&2
    fi
done

python scripts/summarize_results.py \
    --run_root "$RUN_ROOT" \
    --status "$STATUS_FILE" \
    --output "$REPORT_FILE"

echo "Finished. Report: $REPORT_FILE"
echo "Overall exit code: $overall"
exit "$overall"
