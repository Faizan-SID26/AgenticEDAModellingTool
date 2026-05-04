# Quickstart

End-to-end run on a synthetic manufacturing-defect dataset.

## 1. Create a project

```bash
eda new-project my_first --domain manufacturing \
    --recipe manufacturing_defect_classification --budget 30
```

This creates `projects/my_first/` from the template, with
`PROJECT.json` stamped with the framework version.

## 2. Drop your data

Place your raw files (csv / parquet / jsonl) under
`projects/my_first/data/`. The framework will auto-detect schemas.

For this walkthrough you can generate a synthetic dataset:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 5000
process = pd.DataFrame({
    "batch_id": np.arange(n),
    "batch_time": pd.date_range("2024-01-01", periods=n, freq="h"),
    "reactor_temp": rng.normal(100, 5, n),
    "reactor_pressure": rng.normal(1.0, 0.05, n),
    "raw_grade": rng.choice(["A","B","C"], n),
})
qa = pd.DataFrame({
    "batch_id": np.arange(n),
    "downstream_qc_score": rng.normal(0, 1, n),
    "defect": rng.binomial(1, 0.1, n),
})
process.to_parquet("projects/my_first/data/process.parquet")
qa.to_parquet("projects/my_first/data/qa.parquet")
```

## 3. Inspect

In Claude Code, change to the project directory:

```bash
cd projects/my_first
```

Then run the inspection slash command:

```
/init
```

This profiles every file and writes `memory/INIT_PROFILE.json` and
`results/init_report.md`.

## 4. Plan

```
/plan
```

The planner asks a small batch of questions (mostly confirm-the-inference)
and locks the MISSION. After lock, `MISSION.json`, `memory/COLUMNS.json`,
`memory/JOIN_PLAN.json`, and `memory/HYPOTHESES.jsonl` are written.

## 5. Run

```
/run
```

The orchestrator bootstraps the sketch, then iterates autonomously:
researcher → runner → record, with hypothesis-generation every 5
iterations and synthesis + vision checkpoint every 10. Termination is
automatic on goal-met / budget-exhausted / stagnation / iteration-cap.

After termination, the analyst writes `results/FINAL.md`.

## 6. Contribute

```
/contribute
```

This stages `CONTRIBUTION.md` and tells you the exact git commands to
push the branch + open a PR. After merge, CI extracts knowledge into
`knowledge/`.

## 7. Inspect what was learned

```bash
eda library
eda status my_first
```
