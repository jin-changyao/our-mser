# Experiment Log

Last updated: 2026-07-15

Use this file for interpreted results. Raw results remain in `experiments/`.

## Known Baselines

### IEMOCAP Qwen2.5 speech text baseline

Configuration:

```text
DATA_SOURCE=legacy
AUDIO_DESCRIPTION=True
USE_MM_PREFIX=False
USE_PERSONA=False
PROMPT_STYLE=qwen_chat
LORA_DIM=16
LORA_ALPHA=16
LORA_MODULE_NAME=q_proj,k_proj,v_proj,query_key_value
```

Observed:

```text
Around 71-72 weighted F1.
```

Interpretation:

```text
This is the current strong baseline.
```

### IEMOCAP speech text + direct AV prefix

Configuration:

```text
DATA_SOURCE=legacy
AUDIO_DESCRIPTION=True
USE_MM_PREFIX=True
MM_AUDIO_TOKENS=1
MM_VIDEO_TOKENS=1
GRADIENT_CHECKPOINTING=True
```

Observed:

```text
GPU0 about 70.8
GPU2 about 70.3
```

Interpretation:

```text
Direct AV prefix hurts or fails to improve the strong speech-text baseline.
```

### Manifest direct AV prefix

Observed:

```text
IEMOCAP manifest text only: about 67.6
IEMOCAP manifest + AV prefix: about 69.7-69.9
MELD manifest + AV prefix: below speech-text baseline
```

Interpretation:

```text
AV prefix contains useful signal in weaker manifest setup, but is not good enough to beat speech text.
```

## Experiment Standards

- Compare methods under the same LoRA and data settings when possible.
- Repeat important comparisons on at least two runs or two GPUs.
- Treat differences below 1 F1 as inconclusive unless repeated.
- Use `summarize_experiments.py --print_best` after runs.

