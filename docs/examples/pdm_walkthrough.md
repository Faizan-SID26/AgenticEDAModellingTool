# Walk-through: predictive maintenance

```bash
eda new-project demo_pdm --domain manufacturing \
    --recipe equipment_pdm --budget 30
```

Synthetic example:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)
n_assets = 80
rows = []
for asset in range(n_assets):
    age = rng.integers(100, 1000)
    vibration = rng.normal(0.1 + age / 5000, 0.02)
    rows.append({
        "asset_id": asset,
        "obs_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 200))),
        "age_days": age,
        "vibration": vibration,
        "time_to_failure_days": max(1, int(np.clip(np.random.exponential(scale=300 - 0.2 * age), 1, 1000))),
    })
pd.DataFrame(rows).to_parquet("projects/demo_pdm/data/asset_obs.parquet")
```

## /plan

The planner locks `time_to_event` target with group_kfold validation
(grouping by `asset_id`). `target_column = time_to_failure_days`.

## /run

- Iter 1: `cox_ph` baseline. Concordance ≈ 0.6.
- Iter 2-3: feature engineering on `age_days * vibration`.
- Goal-met around iter 4 if threshold = 0.65.

`/finalize` returns an alert-policy recommendation: when to flag an
asset for inspection given its current vibration + age.
