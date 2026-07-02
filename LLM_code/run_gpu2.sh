#!/usr/bin/env bash
set -e

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_experiment.sh" "$@"
