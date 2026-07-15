# Data And Alignment Thread

This thread owns dataset, manifest, feature, label, and split alignment.

## Responsibilities

- Check whether datasets match across SpeechCue, PRC-Emo, and C manifests.
- Validate labels, splits, speaker names, utterance IDs, and feature paths.
- Maintain data diagnosis scripts.
- Explain whether a mixed data setting is valid.

## Not Responsibilities

- Do not decide model architecture.
- Do not run long training jobs except for tiny checks.

## Current Data Sources

SpeechCue legacy:

```text
original_data/<dataset>/<dataset>.pkl
speech_features/*.csv
data_process.py
```

C manifest:

```text
/home/pc/jcy/datasets/iemocap/features/splits_6way
/home/pc/jcy/datasets/meld/features/splits_7way
```

Local Windows copies:

```text
D:\datasets\iemocap\features\splits_6way
D:\datasets\meld\features\splits_7way
```

## Current Alignment Status

IEMOCAP:

```text
New/old text is effectively identical.
Legacy processed JSON now includes utterance_id.
USE_MM_PREFIX=True uses utterance_id to find C features.
Repeated short utterances like "Yeah." and "No." were the main prior mismatch source.
```

MELD:

```text
Manifest has speaker names from PRC data.
MELD text has more differences than IEMOCAP.
Label mismatch was previously reported as 0 against old SpeechCue labels.
```

## Diagnostic Scripts

Check legacy prompt + C feature alignment:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python diagnose_legacy_manifest_alignment.py \
  --dataset iemocap \
  --legacy_data_dir ../PROCESSED_DATASET/iemocap/window/True_False_qwen_chat_prc \
  --manifest_dir /home/pc/jcy/datasets/iemocap/features/splits_6way \
  --speech_feature_dir ../speech_features \
  --out ../experiments/diagnosis_align_iemocap_legacy_speech_manifest.json
```

Check truncation:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python diagnose_truncation.py \
  --data_dir ../PROCESSED_DATASET/iemocap/manifest/window_12_qwen_chat \
  --model_path /home/pc/jcy/models/Qwen2.5-7B-Instruct \
  --max_length 2600
```

## Alignment Acceptance Criteria

For a mixed legacy + C feature experiment:

```text
missing_feature = 0
missing_speech_feature = 0
prompt_without_speech_marker = 0
label_mismatch should be near 0 or understood
missing_manifest should be near 0 or fallback feature files must exist
```

