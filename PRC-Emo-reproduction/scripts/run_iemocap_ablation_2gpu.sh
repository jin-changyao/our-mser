#!/usr/bin/env bash
set -u -o pipefail

# IEMOCAP ablation experiment:
#   GPU0: no optional feature, speaker feature only
#   GPU2: explicit emotion only, retrieval only
# Each GPU runs its two variants sequentially to avoid VRAM contention.

PROJECT_ROOT="${PROJECT_ROOT:-/home/pc/jcy/Our-MSER/PRC-Emo-reproduction}"
MODEL_PATH="${MODEL_PATH:-/home/pc/jcy/models/Qwen2.5-7B-Instruct}"
GPU0="${GPU0:-0}"
GPU2="${GPU2:-2}"
WINDOW="${WINDOW:-5}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-3e-4}"
LORA_R="${LORA_R:-32}"
BUCKETS="${BUCKETS:-2}"
CURRICULUM_UPDATE_EPOCHS="${CURRICULUM_UPDATE_EPOCHS:-1}"
FORCE_PROMPT_REGEN="${FORCE_PROMPT_REGEN:-0}"

RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/iemocap_ablation_2gpu}"
OUTPUT_FOLDER="$RUN_ROOT/finetuned_llm"
LOG_ROOT="$PROJECT_ROOT/logs/iemocap_ablation_2gpu"
STATUS_FILE="$RUN_ROOT/run_status.tsv"
REPORT_FILE="$RUN_ROOT/RESULTS.md"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$LOG_ROOT" "$OUTPUT_FOLDER"
printf 'phase\tdataset\tseed\tgpu\texit_code\tlog\toutput_dir\n' > "$STATUS_FILE"

all_files_ready() {
    local path
    for path in "$@"; do
        [[ -s "$path" ]] || return 1
    done
}

prompt_files() {
    local variant="$1"
    printf '%s\n' \
        "$PROJECT_ROOT/data/iemocap.train.0shot_w${WINDOW}_ablation_${variant}.jsonl" \
        "$PROJECT_ROOT/data/iemocap.valid.0shot_w${WINDOW}_ablation_${variant}.jsonl" \
        "$PROJECT_ROOT/data/iemocap.test.0shot_w${WINDOW}_ablation_${variant}.jsonl"
}

generate_variant_data() {
    local variant="$1"
    local gpu="$2"
    local log_file="$LOG_ROOT/prompt_${variant}_gpu${gpu}.log"
    local files=()
    mapfile -t files < <(prompt_files "$variant")

    if [[ "$FORCE_PROMPT_REGEN" != "1" ]] && all_files_ready "${files[@]}"; then
        echo "[$(date '+%F %T')] Reusing prompt files for $variant" | tee "$log_file"
        return 0
    fi

    echo "[$(date '+%F %T')] START prompt variant=$variant gpu=$gpu" | tee "$log_file"
    env CUDA_VISIBLE_DEVICES="$gpu" \
        PRC_EMO_EMBEDDING_MODEL_PATH="${PRC_EMO_EMBEDDING_MODEL_PATH:-$PROJECT_ROOT/models/all-MiniLM-L6-v2}" \
        python scripts/generate_iemocap_ablation_data.py \
        --variant "$variant" \
        --data_folder "$PROJECT_ROOT/data" \
        --window "$WINDOW" \
        --emotion_model_id "$MODEL_PATH" \
        >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" | tee -a "$log_file"
    return "$code"
}

run_variant() {
    local variant="$1"
    local gpu="$2"
    local output_id="iemocap_qwen25_ablation_${variant}_gpu${gpu}_seed${SEED}"
    local output_dir="$OUTPUT_FOLDER/$output_id"
    local log_file="$LOG_ROOT/train_${variant}_gpu${gpu}_seed${SEED}.log"

    if ! generate_variant_data "$variant" "$gpu"; then
        echo "Prompt generation failed for $variant; skipping training." >&2
        printf 'prompt\tiemocap:%s\t%s\t%s\t1\t%s\t%s\n' \
            "$variant" "$SEED" "$gpu" "$LOG_ROOT/prompt_${variant}_gpu${gpu}.log" "$output_dir" >> "$STATUS_FILE"
        return 1
    fi

    echo "[$(date '+%F %T')] START train variant=$variant seed=$SEED gpu=$gpu" > "$log_file"
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
        --data_name iemocap \
        --prompting_type "ablation_${variant}" \
        --extract_prompting_llm_id Qwen2.5-7B-Instruct \
        --seed "$SEED" \
        --max_seq_len 2048 \
        --eval_delay 100000 \
        --data_folder "$PROJECT_ROOT/data" \
        >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" >> "$log_file"
    printf 'train\tiemocap:%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$variant" "$SEED" "$gpu" "$code" "$log_file" "$output_dir" >> "$STATUS_FILE"
    return "$code"
}

run_gpu0() {
    local status=0
    run_variant none "$GPU0" || status=1
    run_variant speaker "$GPU0" || status=1
    return "$status"
}

run_gpu2() {
    local status=0
    run_variant explicit "$GPU2" || status=1
    run_variant retrieval "$GPU2" || status=1
    return "$status"
}

run_gpu0 & pid_gpu0=$!
run_gpu2 & pid_gpu2=$!
overall=0
wait "$pid_gpu0" || overall=1
wait "$pid_gpu2" || overall=1

python scripts/summarize_results.py \
    --run_root "$RUN_ROOT" \
    --status "$STATUS_FILE" \
    --output "$REPORT_FILE"

echo "Finished. Report: $REPORT_FILE"
echo "Overall exit code: $overall"
exit "$overall"
