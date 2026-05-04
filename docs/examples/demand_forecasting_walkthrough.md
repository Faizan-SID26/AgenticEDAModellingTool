# Walk-through: demand forecasting

```bash
eda new-project demo_demand --domain forecasting_demand \
    --recipe demand_forecasting --budget 30
```

Generate a synthetic series with weekly seasonality + a price effect:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
days = pd.date_range("2023-01-01", periods=600, freq="D")
dow = np.array([d.dayofweek for d in days])
season = 100 + 20 * np.sin(2 * np.pi * dow / 7)
price = 10 + rng.normal(0, 1, len(days))
demand = season - 3 * (price - 10) + rng.normal(0, 4, len(days))
df = pd.DataFrame({"date": days, "dow": dow, "list_price": price, "demand": demand})
df.to_parquet("projects/demo_demand/data/series.parquet")
```

## /init → /plan → /run

Expected:
- /plan locks capability=`seasonal/forecast_horizon/multi_horizon/rolling_origin/forecast`.
- /run baseline naive_seasonal MAPE ≈ 5–10%.
- Lagged ridge improves to MAPE ≈ 4–6%.
- Goal-met at MAPE ≤ 0.15 (very early — threshold is loose for the
  synthetic data).
- /finalize produces a forecast-shaped recommendation.
