# Decision Log

Last updated: 2026-07-15

## Decisions

### Use Our-MSER as the formal working repository

Decision:

```text
Use /home/pc/jcy/Our-MSER and GitHub jin-changyao/our-mser for active experiments.
```

Reason:

```text
Keeps SpeechCueLLM and MSE-Adapter references intact while giving us a clean place for our method.
```

### Use Qwen2.5 as primary LLM

Decision:

```text
Default active model is Qwen2.5-7B-Instruct.
```

Reason:

```text
It performed competitively or better than LLaMA2 in current runs.
```

### Treat speech text as the strong baseline

Decision:

```text
The main baseline is Qwen2.5 + legacy SpeechCue data + speech text prompt.
```

Reason:

```text
It is currently the strongest and most stable direction.
```

### Do not treat direct AV prefix as final method

Decision:

```text
Direct audio/video prefix injection is an ablation baseline, not the final method.
```

Reason:

```text
It lowered scores when added to the strong speech-text baseline.
```

### Prioritize text-guided multimodal learning

Decision:

```text
Next method direction should use speech text or emotion text to guide continuous audio/video features.
```

Reason:

```text
Direct continuous feature learning is hard; speech text is strong and can act as a semantic teacher.
```

### Keep conflict-aware fusion as a future method point

Decision:

```text
Modality conflict handling is a future design axis.
```

Reason:

```text
Text, audio, and video can express different emotions. The model needs dynamic reliability/gating instead of naive fusion.
```

