"""Univariate / multivariate forecasting (multi_horizon target)."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="forecasting",
    description="Forecasting numeric targets at horizons h>=1, using rolling-origin validation.",
    composition=CapabilityComposition(
        temporal_structure="seasonal",
        leakage_model="forecast_horizon",
        target_type="multi_horizon",
        validation_strategy="rolling_origin",
        recommendation_type="forecast",
    ),
    required_mission_fields=("target_column", "time_column", "success_criterion"),
    default_models=("naive_seasonal", "ridge_lagged", "lgbm_regressor"),
    default_metrics=("mape", "rmse", "smape"),
    primary_metric="mape",
    primary_metric_direction="<=",
    sketch_extras_needed=("L1_distributions", "L3_regimes", "L5_timeseries"),
    seed_hypothesis_recipe_keys=(
        "naive_seasonal_baseline",
        "lagged_ridge",
        "stl_decompose_then_residual_model",
    ),
)


def make_splitter():
    """Return a rolling-origin splitter."""

    def split(
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
        n_splits: int = 5,
        horizon: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        order = (
            np.argsort(np.asarray(list(time)), kind="stable")
            if time is not None
            else np.arange(n_rows)
        )
        chunk = max(1, n_rows // (n_splits + 2))
        out = []
        for i in range(1, n_splits + 1):
            train_end = i * chunk
            val_start = train_end + horizon - 1
            val_end = min(val_start + chunk, n_rows)
            if val_start >= n_rows or val_end <= val_start:
                continue
            out.append((order[:train_end], order[val_start:val_end], None))
        if not out:
            mid = max(1, n_rows // 2)
            out.append((order[:mid], order[mid:], None))
        return out

    return split
