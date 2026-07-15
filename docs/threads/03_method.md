# Method Thread

This thread owns research design and method planning.

## Responsibilities

- Decide what the next method should be.
- Compare ideas from MSE-Adapter, SpeechCueLLM, PRC-Emo, and related papers.
- Define ablations.
- Produce implementation requirements for the Code thread.

## Not Responsibilities

- Do not debug server commands.
- Do not patch code directly unless the user explicitly wants implementation in the same thread.

## Current Method Understanding

Direct prefix:

```text
audio/video feature -> projector -> prefix tokens -> Qwen
```

Observed issue:

```text
It does not improve the strong speech-text baseline.
```

Hypothesis:

```text
Continuous features are hard for the LLM to interpret without semantic guidance.
```

## Primary Next Method

Text-guided multimodal learning.

Core idea:

```text
Use speech text as a teacher for audio features.
```

Possible loss:

```text
L = L_lm + lambda * L_align
```

Where:

```text
L_lm: normal label-generation loss
L_align: cosine/MSE contrast between audio prefix and speech-text representation
```

Why:

```text
Speech text is already useful and semantically understandable by Qwen.
Audio features should first learn to express that semantic information.
```

## Future Method Point

Conflict-aware fusion.

Problem:

```text
Text, audio, and video can imply different emotions.
```

Potential solution:

```text
dynamic gates / reliability scores / modality dropout / conflict detector
```

Keep this as a future innovation point after text-guided learning is tested.

## Ablation Plan For Any New Method

Required:

```text
speech text baseline
direct AV prefix
text-guided audio only
text-guided audio + video if video teacher exists
alignment loss weight sweep
```

Optional:

```text
audio only
video only
modality dropout
gated fusion
```

