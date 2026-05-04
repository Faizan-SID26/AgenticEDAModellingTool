# Walk-through: manufacturing defect classification

End-to-end on a synthetic process line. ~10 minutes wall-time on a laptop.

## 1. Project + data

```bash
eda new-project demo_defect --domain manufacturing \
    --recipe manufacturing_defect_classification --budget 30
```

Generate a synthetic dataset with 5000 batches and a defect indicator
that depends on `reactor_temp`:

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
    "raw_grade": rng.choice(["A", "B", "C"], n),
})
qa = pd.DataFrame({
    "batch_id": np.arange(n),
    "downstream_qc_score": rng.normal(0, 1, n),
    "defect": ((process["reactor_temp"] - 100) > 4 + rng.normal(0, 0.5, n)).astype(int),
})
process.to_parquet("projects/demo_defect/data/process.parquet")
qa.to_parquet("projects/demo_defect/data/qa.parquet")
```

## 2. /init

```
cd projects/demo_defect
/init
```

Output: `memory/INIT_PROFILE.json` and `results/init_report.md`. The
report identifies `defect` as the likely target and `batch_time` as the
likely time column.

## 3. /plan

```
/plan
```

The planner asks ~3 questions, mostly confirm-the-inference:
- "Is `defect` the target?" → yes.
- "Is `batch_time` the time-ordering column?" → yes.
- "Treat `downstream_qc_score` as forbidden?" → yes (it's downstream of
  the defect).

After lock, `MISSION.json` reflects:
- capability: regime_based / stage_frontier / binary / time_split / decision.
- target_column: `defect`, time_column: `batch_time`.
- forbidden_columns: `["downstream_qc_score"]`.
- success_criterion: `roc_auc >= 0.78` on validation.

## 4. /run

```
/run
```

Phase A bootstraps the sketch in <30 seconds.

Phase B iterates. Expected behavior:
- Iter 1 (H-seed-1, naive baseline) → roc_auc ≈ 0.95 on validation.
  Skeptic flags `too_good_to_be_true_likely_leakage` because we used few
  features. Reviewer notes the model is over-relying on `reactor_temp`.
- Iter 2-3 (interactions, regimes) → small gains.
- Goal-met termination at iter ≈ 4-5.

Phase C: analyst writes `results/FINAL.md`:
- decision: "Adopt logreg with features [reactor_temp, reactor_pressure, ...]"
- counterfactual estimate (linear regression fallback) for the effect
  of `reactor_temp` on `defect`.
- ruled-out failure modes from L7.

## 5. /contribute

```
/contribute
```

Stages `CONTRIBUTION.md`. Commit + push + PR. Post-merge, the extractor
adds two rows to `knowledge/hypothesis_library.jsonl` (if info-gain
threshold met) and updates the sketch index.
