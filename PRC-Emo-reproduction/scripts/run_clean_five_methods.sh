#!/usr/bin/env bash
set -u -o pipefail

# Run five single-seed methods on IEMOCAP and MELD without GPU conflicts.
# Prompt generation is serialized on PROMPT_GPU. Training is scheduled in
# waves of three jobs, then two jobs, and only moves on after the whole wave
# has finished.

PROJECT_ROOT="${PROJECT_ROOT:-/home/pc/jcy/Our-MSER/PRC-Emo-reproduction}"
MODEL_PATH="${MODEL_PATH:-/home/pc/jcy/models/Qwen2.5-7B-Instruct}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
PROMPT_GPU="${PROMPT_GPU:-$GPU0}"
SEED="${SEED:-42}"
WINDOW="${WINDOW:-5}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-3e-4}"
LORA_R="${LORA_R:-32}"
BUCKETS="${BUCKETS:-2}"
CURRICULUM_UPDATE_EPOCHS="${CURRICULUM_UPDATE_EPOCHS:-1}"
EMOTION_FEATURE_SUFFIX="${EMOTION_FEATURE_SUFFIX:-ImplicitEmotion_V3_${MODEL_PATH##*/}}"
SPEAKER_FEATURE_SUFFIX="${SPEAKER_FEATURE_SUFFIX:-spdescV6_trimmed_${MODEL_PATH##*/}}"
SPEAKER_FEATURE_DIR="${SPEAKER_FEATURE_DIR:-$PROJECT_ROOT/data}"
EMBEDDING_MODEL_PATH="${PRC_EMO_EMBEDDING_MODEL_PATH:-$PROJECT_ROOT/models/all-MiniLM-L6-v2}"
FORCE_PROMPT_REGEN="${FORCE_PROMPT_REGEN:-1}"
RUN_NAME="${RUN_NAME:-clean_five_methods_seed${SEED}_$(date '+%Y%m%d_%H%M%S')}"

METHODS=(none emotion speaker retrieval full)
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_NAME}"
OUTPUT_FOLDER="$RUN_ROOT/finetuned_llm"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_NAME"
STATUS_FILE="$RUN_ROOT/run_status.tsv"
REPORT_FILE="$RUN_ROOT/RESULTS.md"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$RUN_ROOT" "$OUTPUT_FOLDER" "$LOG_ROOT"
printf 'phase\tdataset\tseed\tgpu\texit_code\tlog\toutput_dir\n' > "$STATUS_FILE"

all_files_ready() {
    local path
    for path in "$@"; do
        [[ -s "$path" ]] || return 1
    done
    return 0
}

emotion_file() {
    local dataset="$1" split="$2"
    echo "$PROJECT_ROOT/data/${dataset}.${split}_${EMOTION_FEATURE_SUFFIX}.json"
}

speaker_file() {
    local dataset="$1" split="$2"
    echo "$SPEAKER_FEATURE_DIR/${dataset}.${split}_${SPEAKER_FEATURE_SUFFIX}.json"
}

prompt_file() {
    local dataset="$1" method="$2" split="$3"
    echo "$PROJECT_ROOT/data/${dataset}.${split}.0shot_w${WINDOW}_experiment_${method}.jsonl"
}

prompt_ready_flag() {
    local dataset="$1" method="$2"
    echo "$RUN_ROOT/prompt_${dataset}_${method}.ready"
}

record_status() {
    local phase="$1" dataset="$2" method="$3" seed="$4" gpu="$5" code="$6" log="$7" output="$8"
    printf '%s\t%s:%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$dataset" "$method" "$seed" "$gpu" "$code" "$log" "$output" >> "$STATUS_FILE"
}

prepare_method_prompt() {
    local dataset="$1" method="$2"
    local log_file="$LOG_ROOT/prompt_${dataset}_${method}.log"
    local output_flag
    output_flag="$(prompt_ready_flag "$dataset" "$method")"
    local files=("$(prompt_file "$dataset" "$method" train)" "$(prompt_file "$dataset" "$method" valid)" "$(prompt_file "$dataset" "$method" test)")

    if [[ "$FORCE_PROMPT_REGEN" != "1" ]] && all_files_ready "${files[@]}"; then
        echo "[$(date '+%F %T')] REUSE prompt dataset=$dataset method=$method" | tee "$log_file"
        touch "$output_flag"
        record_status prompt "$dataset" "$method" "$SEED" "$PROMPT_GPU" 0 "$log_file" "${files[0]}"
        return 0
    fi

    echo "[$(date '+%F %T')] START prompt dataset=$dataset method=$method gpu=$PROMPT_GPU" | tee "$log_file"
    env CUDA_VISIBLE_DEVICES="$PROMPT_GPU" \
        PRC_EMO_EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH" \
        python scripts/generate_experiment_prompts.py \
        --dataset "$dataset" \
        --method "$method" \
        --data_folder "$PROJECT_ROOT/data" \
        --window "$WINDOW" \
        --emotion_feature_suffix "$EMOTION_FEATURE_SUFFIX" \
        --speaker_feature_dir "$SPEAKER_FEATURE_DIR" \
        --speaker_feature_suffix "$SPEAKER_FEATURE_SUFFIX" \
        --embedding_model_path "$EMBEDDING_MODEL_PATH" \
        >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" | tee -a "$log_file"
    if [[ "$code" -eq 0 ]] && all_files_ready "${files[@]}"; then
        touch "$output_flag"
    else
        rm -f "$output_flag"
        echo "Prompt generation failed or output files are missing: $dataset/$method" >&2
    fi
    record_status prompt "$dataset" "$method" "$SEED" "$PROMPT_GPU" "$code" "$log_file" "${files[0]}"
    return "$code"
}

prepare_dataset() {
    local dataset="$1"
    local overall=0
    echo "===== Preparing $dataset prompts ====="

    for split in train valid test; do
        local raw="$PROJECT_ROOT/data/${dataset}.${split}.json"
        if ! all_files_ready "$raw"; then
            echo "Missing raw data: $raw" >&2
            overall=1
        fi
    done

    for method in "${METHODS[@]}"; do
        case "$method" in
            emotion|full)
                for split in train valid test; do
                    if ! all_files_ready "$(emotion_file "$dataset" "$split")"; then
                        echo "Missing emotion feature: $(emotion_file "$dataset" "$split")" >&2
                        overall=1
                    fi
                done
                ;;
            speaker|full)
                for split in train valid test; do
                    if ! all_files_ready "$(speaker_file "$dataset" "$split")"; then
                        echo "Missing speaker feature: $(speaker_file "$dataset" "$split")" >&2
                        overall=1
                    fi
                done
                ;;
        esac
        if [[ "$method" == "retrieval" || "$method" == "full" ]]; then
            if ! all_files_ready "$PROJECT_ROOT/data/Emotion_Retrieval_Library_3.json" || [[ ! -e "$EMBEDDING_MODEL_PATH" ]]; then
                echo "Missing retrieval library or embedding model." >&2
                overall=1
            fi
        fi
    done

    [[ "$overall" -eq 0 ]] || return 1
    for method in "${METHODS[@]}"; do
        prepare_method_prompt "$dataset" "$method" || overall=1
    done
    return "$overall"
}

train_method() {
    local dataset="$1" method="$2" gpu="$3"
    local method_root="$OUTPUT_FOLDER/$dataset/$method"
    local output_id="${dataset}_experiment_${method}_seed${SEED}"
    local output_dir="$method_root/$output_id"
    local log_file="$LOG_ROOT/train_${dataset}_${method}_seed${SEED}_gpu${gpu}.log"
    mkdir -p "$method_root"

    echo "[$(date '+%F %T')] START train dataset=$dataset method=$method seed=$SEED gpu=$gpu" > "$log_file"
    env CUDA_VISIBLE_DEVICES="$gpu" \
        PRC_EMO_ATTN_IMPL="${PRC_EMO_ATTN_IMPL:-sdpa}" \
        python src/ft_llm_cl.py \
        --do_train --do_eval_dev --do_eval_test --curriculum \
        --bucket_number "$BUCKETS" \
        --curriculum_update_epochs "$CURRICULUM_UPDATE_EPOCHS" \
        --base_model_id "$MODEL_PATH" \
        --ft_model_id "$output_id" \
        --output_folder "$method_root" \
        --lr_scheduler linear \
        --lr "$LR" \
        --lora_r "$LORA_R" \
        --max_steps -1 \
        --epoch "$EPOCHS" \
        --kshot 0 \
        --window "$WINDOW" \
        --data_name "$dataset" \
        --prompting_type "experiment_${method}" \
        --extract_prompting_llm_id "Qwen2.5-7B-Instruct" \
        --seed "$SEED" \
        --max_seq_len 2048 \
        --eval_delay 100000 \
        --data_folder "$PROJECT_ROOT/data" \
        >> "$log_file" 2>&1
    local code=$?
    echo "[$(date '+%F %T')] EXIT $code" >> "$log_file"
    record_status train "$dataset" "$method" "$SEED" "$gpu" "$code" "$log_file" "$output_dir"
    return "$code"
}

run_wave() {
    local dataset="$1"
    shift
    local methods=("$@")
    local gpus=("$GPU0" "$GPU1" "$GPU2")
    local pids=()
    local i method
    echo "===== START training wave: $dataset / ${methods[*]} ====="
    for i in "${!methods[@]}"; do
        method="${methods[$i]}"
        if [[ ! -f "$(prompt_ready_flag "$dataset" "$method")" ]]; then
            echo "Skipping $dataset/$method because its Prompt was not prepared." >&2
            continue
        fi
        train_method "$dataset" "$method" "${gpus[$i]}" &
        pids+=("$!")
    done

    local overall=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || overall=1
    done
    echo "===== END training wave: $dataset / ${methods[*]} (status=$overall) ====="
    return "$overall"
}

run_dataset() {
    local dataset="$1"
    local overall=0
    prepare_dataset "$dataset" || overall=1
    run_wave "$dataset" none emotion retrieval || overall=1
    run_wave "$dataset" speaker full || overall=1
    return "$overall"
}

overall=0
run_dataset iemocap || overall=1
run_dataset meld || overall=1

python scripts/summarize_results.py \
    --run_root "$RUN_ROOT" \
    --status "$STATUS_FILE" \
    --output "$REPORT_FILE" || overall=1

echo "Finished. Run root: $RUN_ROOT"
echo "Report: $REPORT_FILE"
echo "Overall exit code: $overall"
exit "$overall"
