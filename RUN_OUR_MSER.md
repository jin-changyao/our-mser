# Our-MSER Running Guide

Use only this project for formal experiments:

```bash
/home/pc/jcy/Our-MSER
```

The runner keeps the original server baseline training protocol:

```text
python main.py
BS=8
gradient accumulation=8
epochs=15
learning rate=3e-4
historical window=12
LoRA dim=16
LoRA alpha=16
LoRA modules=q_proj,k_proj,v_proj,query_key_value
```

## Model Paths

The scripts expect models under:

```bash
/home/pc/jcy/models
```

Expected paths:

```bash
/home/pc/jcy/models/Llama-2-7b-hf
/home/pc/jcy/models/Qwen2.5-7B-Instruct
```

## GPU Scripts

GPU 0:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
bash run_gpu0.sh
```

GPU 2:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
bash run_gpu2.sh
```

Both scripts set the full single-process distributed environment:

```bash
MASTER_ADDR=127.0.0.1
RANK=0
LOCAL_RANK=0
WORLD_SIZE=1
```

GPU 0 uses `MASTER_PORT=29500`; GPU 2 uses `MASTER_PORT=29502`.

## Diagnose Prompt Truncation

This does not train the model. It loads the processed JSON files and the
tokenizer, then compares right truncation and left truncation.

IEMOCAP:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python diagnose_truncation.py \
  --data_dir ../PROCESSED_DATASET/iemocap/manifest/window_12_qwen_chat \
  --model_path /home/pc/jcy/models/Qwen2.5-7B-Instruct \
  --max_length 2600 \
  --out ../experiments/diagnosis_truncation_iemocap_2600.json
```

MELD:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python diagnose_truncation.py \
  --data_dir ../PROCESSED_DATASET/meld/manifest/window_12_qwen_chat \
  --model_path /home/pc/jcy/models/Qwen2.5-7B-Instruct \
  --max_length 1500 \
  --out ../experiments/diagnosis_truncation_meld_1500.json
```

## Diagnose Legacy + Manifest Alignment

Use this before running a mixed setting such as legacy SpeechCue prompts plus C
audio/video prefix features. It checks whether every legacy sample can find the
matching C-manifest row, whether labels match, whether audio/video feature files
exist, whether speech-text features can be found, and whether the prompt really
contains speech information.

IEMOCAP legacy speech-text + C features:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python diagnose_legacy_manifest_alignment.py \
  --dataset iemocap \
  --legacy_data_dir ../PROCESSED_DATASET/iemocap/window/True_False_qwen_chat_prc \
  --manifest_dir /home/pc/jcy/datasets/iemocap/features/splits_6way \
  --speech_feature_dir ../speech_features \
  --out ../experiments/diagnosis_align_iemocap_legacy_speech_manifest.json
```

MELD legacy speech-text + C features:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python diagnose_legacy_manifest_alignment.py \
  --dataset meld \
  --legacy_data_dir ../PROCESSED_DATASET/meld/window/True_False_qwen_chat_prc \
  --manifest_dir /home/pc/jcy/datasets/meld/features/splits_7way \
  --speech_feature_dir ../speech_features \
  --out ../experiments/diagnosis_align_meld_legacy_speech_manifest.json
```

## Summarize Experiment Results

Run this after experiments finish. It scans all `preds_for_eval*.text` files
under `../experiments`, reads `run_config.json` and `model_args.json` when
available, and writes reusable summary files.

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python summarize_experiments.py --print_best
```

Default outputs:

```text
/home/pc/jcy/Our-MSER/experiments/summary/all_results.csv
/home/pc/jcy/Our-MSER/experiments/summary/best_results.csv
/home/pc/jcy/Our-MSER/experiments/summary/experiment_summary.md
```

Use `best_results.csv` or `experiment_summary.md` to quickly check which
configuration is best for each dataset. Use `all_results.csv` when you want to
compare every epoch of every run.

## Common Runs

Default model is Qwen2.5. The default method is C-manifest + Qwen2.5 + audio/video feature prefix, without speech-description text and without persona:

```bash
bash run_gpu0.sh
```

Default method flags:

```text
DATA_SOURCE=manifest
AUDIO_DESCRIPTION=False
USE_PERSONA=False
USE_MM_PREFIX=True
PROMPT_STYLE=qwen_chat
```

Each run writes to a timestamped output directory ending in `run_YYYYmmdd_HHMMSS`.
The directory contains `run_config.json` from the shell runner, `model_args.json`
from `main.py`, `model_params.json`, `eval_results_*.txt`, and
`preds_for_eval_*.text`.

The default prompt style is `qwen_chat`, which renders each sample with the
model tokenizer's chat template. To run the old prompt format for comparison:

```bash
PROMPT_STYLE=legacy bash run_gpu0.sh
```

Processed data is reused by default once the matching
`PROCESSED_DATASET/.../train.json`, `test.json`, and `valid.json` files exist.
To generate the default IEMOCAP + Qwen-chat data once before training:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
PREPROCESS_ONLY=True REPROCESS_DATA=True bash run_gpu0.sh
```

After that, normal runs reuse the processed files. To force regeneration:

```bash
REPROCESS_DATA=True bash run_gpu0.sh
```

IEMOCAP + LLaMA2 with the same C-manifest + AV-prefix route:

```bash
MODEL_NAME=llama2 bash run_gpu0.sh
```

MELD + Qwen2.5 + C-manifest + AV-prefix:

```bash
DATASET=meld bash run_gpu0.sh
```

MELD + LLaMA2 + C-manifest + AV-prefix:

```bash
DATASET=meld MODEL_NAME=llama2 bash run_gpu0.sh
```

Run on GPU 2 by replacing `run_gpu0.sh` with `run_gpu2.sh`.

## Generate MELD Persona

MELD persona is disabled by default. Generate it first if you want to run the persona ablation:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
bash generate_meld_persona_features.sh
```

The default generator model is:

```bash
/home/pc/jcy/models/Qwen2.5-7B-Instruct
```

The default output is:

```bash
/home/pc/jcy/Our-MSER/persona_features/meld_qwen25_persona.json
```

The MELD generator uses dialogue-local profiles and defaults to `MIN_UTTERANCES=2`. Speakers with fewer utterances get a limited-evidence profile instead of a confident persona.

After generation, run MELD with persona:

```bash
DATASET=meld USE_PERSONA=True bash run_gpu0.sh
```

or:

```bash
DATASET=meld MODEL_NAME=qwen2.5 USE_PERSONA=True bash run_gpu2.sh
```

## Persona Defaults

Defaults:

```text
IEMOCAP: USE_PERSONA=True
MELD: USE_PERSONA=False
```

Override manually:

```bash
USE_PERSONA=False bash run_gpu0.sh
USE_PERSONA=True PERSONA_PATH=../persona_features/meld_qwen25_persona.json DATASET=meld bash run_gpu0.sh
```

MELD persona is an ablation setting, not the default baseline. First run MELD without persona, then compare with `USE_PERSONA=True`.
