# Replay & reproducibility

`lib.replay.replay_project(...)` reads a project's `experiment_log.jsonl`
plus the framework version pinned in `PROJECT.json` and reproduces all
artifacts deterministically.

## What is reproduced

- The joined parquet (`sketch/raw_joined.parquet`).
- The full sketch (L1..L7) at the build seed.
- One `ExperimentResult` per row in the original log, executed via
  `lib.run.execute_plan` with the recorded seeds.

## Drift report

The replayer returns a `drift` list, one entry per replayed iteration:

```json
{
  "id": "P-7-abc",
  "primary_metric": "roc_auc",
  "original": 0.812,
  "replayed": 0.812,
  "abs_delta": 0.0
}
```

A clean replay has `abs_delta < 1e-6` for every iteration. Any delta is
a determinism bug.

## CLI

```bash
eda replay <project_name>
eda replay <project_name> --up-to-iteration 12
```

## Why this matters

- **Audit trail.** Anyone with the project repo can reproduce every
  recommended decision.
- **Bug finding.** A regression in a sketch layer or a metric immediately
  surfaces as drift.
- **Migration testing.** When schemas bump, replay confirms the
  migration is value-preserving.
