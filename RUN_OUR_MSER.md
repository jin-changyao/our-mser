# Our-MSER Running Guide

Use only this project for formal experiments:

```bash
/home/pc/jcy/Our-MSER
```

The runner keeps the original server baseline protocol:

```text
python main.py
BS=8
gradient accumulation=8
epochs=15
learning rate=3e-4
historical window=12
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

## Common Runs

IEMOCAP + LLaMA2 + speech description + persona:

```bash
bash run_gpu0.sh
```

IEMOCAP + Qwen2.5 + speech description + persona:

```bash
MODEL_NAME=qwen2.5 bash run_gpu0.sh
```

MELD + LLaMA2 + speech description, no persona:

```bash
DATASET=meld bash run_gpu0.sh
```

MELD + Qwen2.5 + speech description, no persona:

```bash
DATASET=meld MODEL_NAME=qwen2.5 bash run_gpu0.sh
```

Run on GPU 2 by replacing `run_gpu0.sh` with `run_gpu2.sh`.

## Generate MELD Persona

MELD persona is disabled by default. Generate it first if you want to run the persona ablation:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
bash generate_meld_persona_features.sh
```

The default output is:

```bash
/home/pc/jcy/Our-MSER/persona_features/meld_llama2_persona.json
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
USE_PERSONA=True PERSONA_PATH=../persona_features/meld_llama2_persona.json DATASET=meld bash run_gpu0.sh
```

MELD persona is an ablation setting, not the default baseline. First run MELD without persona, then compare with `USE_PERSONA=True`.
