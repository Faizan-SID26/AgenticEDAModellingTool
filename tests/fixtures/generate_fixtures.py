"""Generate synthetic test fixtures (deterministic).

Outputs:
    tests/fixtures/synthetic_classification.parquet
    tests/fixtures/synthetic_regression.parquet
    tests/fixtures/synthetic_timeseries.parquet
    tests/fixtures/synthetic_pdm.parquet

Run:
    python tests/fixtures/generate_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent


def _synthetic_classification(seed: int = 0, n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.choice(["A", "B", "C"], n)
    logits = 1.5 * x1 - 0.5 * x2 + (np.where(x3 == "A", 1.0, 0.0)) + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "downstream_qc": rng.normal(0, 1, n),
            "y": (logits > 0).astype(int),
        }
    )


def _synthetic_regression(seed: int = 1, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(2, 0.5, n)
    y = 1.5 * x1 + 0.3 * x2 ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _synthetic_timeseries(seed: int = 2, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    dow = np.array([d.dayofweek for d in t])
    season = 100 + 20 * np.sin(2 * np.pi * dow / 7)
    price = 10 + rng.normal(0, 1, n)
    demand = season - 3 * (price - 10) + rng.normal(0, 4, n)
    return pd.DataFrame({"t": t, "dow": dow, "list_price": price, "demand": demand})


def _synthetic_pdm(seed: int = 3, n_assets: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for asset in range(n_assets):
        age = int(rng.integers(100, 1000))
        vibration = float(rng.normal(0.1 + age / 5000, 0.02))
        ttf = max(1, int(np.clip(rng.exponential(scale=300 - 0.2 * age), 1, 1000)))
        rows.append(
            {
                "asset_id": asset,
                "obs_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 200))),
                "age_days": age,
                "vibration": vibration,
                "time_to_failure_days": ttf,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _synthetic_classification().to_parquet(OUT_DIR / "synthetic_classification.parquet")
    _synthetic_regression().to_parquet(OUT_DIR / "synthetic_regression.parquet")
    _synthetic_timeseries().to_parquet(OUT_DIR / "synthetic_timeseries.parquet")
    _synthetic_pdm().to_parquet(OUT_DIR / "synthetic_pdm.parquet")
    print(f"wrote 4 fixtures under {OUT_DIR}")


if __name__ == "__main__":
    main()
