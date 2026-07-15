# Code Thread

This thread owns code changes, scripts, diagnostics, commits, and GitHub pushes.

## Responsibilities

- Modify code in `Our-MSER-code`.
- Keep changes scoped and documented.
- Add or update runner flags.
- Add diagnostic scripts.
- Commit and push to GitHub.
- Update relevant docs when behavior changes.

## Not Responsibilities

- Do not decide long-term research direction alone.
- Do not interpret large experiment tables unless needed for a code fix.

## Current Important Files

```text
LLM_code/run_experiment.sh
LLM_code/main.py
LLM_code/data_process.py
LLM_code/data_utils/data_utils.py
LLM_code/model_utils/model.py
LLM_code/summarize_experiments.py
LLM_code/diagnose_legacy_manifest_alignment.py
LLM_code/diagnose_truncation.py
```

## Current Implementation Notes

- `run_experiment.sh` controls environment-variable based experiments.
- `GRADIENT_CHECKPOINTING=True` passes `--gradient_checkpointing`.
- `data_process.py` emits `utterance_id` for legacy processed samples.
- `read_data()` prefers `utterance_id` for C manifest feature lookup.
- `AVPrefixLLM` currently does simple independent audio/video projector prefix injection.

## Code Change Protocol

Before changes:

```bash
git status --short --branch
```

After changes:

```bash
python -m py_compile <changed python files>
git diff --check
git status --short --branch
```

Then commit and push:

```bash
git add <files>
git commit -m "<message>"
git push origin HEAD:main
```

## Next Likely Code Task

Implement text-guided audio prefix learning:

```text
speech description -> teacher representation
audio feature -> prefix representation
alignment loss added to LM loss
```

Keep it behind flags, for example:

```text
USE_TEXT_GUIDED_AUDIO=True
ALIGN_LOSS_WEIGHT=...
```

