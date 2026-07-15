# Experiment Thread

This thread owns experiment planning, execution commands, result summaries, and run interpretation.

## Responsibilities

- Decide the next runs.
- Provide exact server commands.
- Read `experiment_summary.md`, `best_results.csv`, and `run_config.json`.
- Update `docs/status/EXPERIMENT_LOG.md` with interpreted results.
- Mark which comparisons are fair or unfair.

## Not Responsibilities

- Do not design large new methods here.
- Do not refactor code unless it is a small runner/logging fix.
- Do not deep-dive dataset alignment; send that to the Data thread.

## Standard Workflow

1. Pull latest code on server.
2. Run one or two controlled configurations.
3. Run:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
python summarize_experiments.py --print_best
```

4. Compare against the correct baseline.
5. Update `docs/status/EXPERIMENT_LOG.md`.

## Fair Comparison Rules

Use same:

```text
dataset
model
prompt/data source
LoRA config
seed when possible
gradient checkpointing if memory requires it
```

Only change the variable under test.

## Current Priority Experiments

1. Confirm direct AV prefix result is consistently below speech text baseline.
2. If needed, run no-prefix baseline with `GRADIENT_CHECKPOINTING=True` to confirm checkpointing does not change results meaningfully.
3. After new method implementation, compare:

```text
speech text baseline
speech text + direct AV prefix
speech text + text-guided audio prefix
```

