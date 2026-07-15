# Project Status

Last updated: 2026-07-15

## Goal

Build an improved multimodal emotion recognition method for IEMOCAP and MELD, using SpeechCueLLM as the current code base and MSE-Adapter/PRC-Emo as method references.

## Active Baseline

Current strongest practical baseline:

```text
Qwen2.5 + legacy data process + speech text + qwen_chat prompt
```

Typical IEMOCAP result:

```text
71-72 weighted F1
```

Observed repeated-run behavior:

```text
Different GPU/runs can vary by roughly 1-2 F1.
Small gains below 1 F1 should not be trusted without repeated runs.
```

## Current Code Capabilities

- Qwen2.5 and LLaMA2 model selection through `MODEL_NAME`.
- `DATA_SOURCE=legacy` for SpeechCue-style processed prompts.
- `DATA_SOURCE=manifest` for C-manifest prompts and features.
- `USE_MM_PREFIX=True` injects C audio/video features as prefix embeddings.
- Legacy processed JSON now includes `utterance_id` for safer C-feature alignment.
- `GRADIENT_CHECKPOINTING=True` is exposed in `run_experiment.sh`.
- `summarize_experiments.py` produces reusable experiment summaries.
- `diagnose_legacy_manifest_alignment.py` checks legacy prompt + C feature alignment.
- `diagnose_truncation.py` checks prompt truncation.

## Current Findings

Useful:

- Speech text prompt is consistently useful.
- Qwen2.5 is worth keeping as primary LLM.
- Gradient checkpointing reduces memory enough to make Qwen2.5 mixed runs feasible.

Weak or unstable:

- Persona is not stable; sometimes helps, often hurts.
- Direct AV prefix does not currently improve the strong speech-text baseline.
- Simply increasing prefix tokens or LoRA capacity is not a promising next step.

## Current Main Risks

- Result variance is large enough to confuse small improvements.
- Direct multimodal prefix may inject noise when speech text is already strong.
- MELD and IEMOCAP have different data/feature quality patterns.
- 24GB GPUs are close to the limit without gradient checkpointing.

## Recommended Next Milestone

Design and implement a text-guided multimodal learning variant:

```text
audio feature -> audio prefix
speech description -> teacher semantic representation
loss = LM loss + lambda * audio-speech alignment loss
```

Then compare:

```text
speech text baseline
vs
speech text + direct AV prefix
vs
speech text + text-guided audio prefix
```

