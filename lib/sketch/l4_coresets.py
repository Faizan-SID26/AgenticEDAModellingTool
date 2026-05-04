"""L4: per-capability importance-weighted coresets.

Strategy:
- For classification: stratified-by-target weighted sample, with extra
  weight on minority class (so a cap of, say, 5000 rows is enough to fit
  fast baselines without losing minority signal).
- For regression: weight by absolute deviation from the target's median
  (so tail observations are over-represented).
- For survival / temporal: weight by recency.
- Default: simple uniform random sample.

Each coreset is saved as a parquet file with the rows + a `weight` column.
The L4 summary lives in `manifest.l4_paths`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib.schemas.sketch import L4CoresetSummary

_log = logging.getLogger("eda.sketch.l4")

_DEFAULT_CORESET_SIZE = 5000


def _classification_weights(y: pd.Series) -> np.ndarray:
    """Inverse-class-frequency weights, normalized to mean 1."""
    counts = y.value_counts(dropna=False)
    inv = (1.0 / counts).to_dict()
    w = y.map(inv).astype(float).fillna(1.0).values
    return w / w.mean()


def _regression_weights(y: pd.Series) -> np.ndarray:
    """Tail-emphasizing weights = 1 + |y - median(y)| / mad."""
    med = float(np.nanmedian(y))
    mad = float(np.nanmedian(np.abs(y - med))) or 1.0
    w = 1.0 + np.abs(y - med).values / mad
    return w / w.mean()


def _recency_weights(time: pd.Series) -> np.ndarray:
    """Linearly increasing in rank-of-time."""
    if time is None or time.isna().all():
        return np.ones(len(time))
    rank = time.rank(method="dense").values.astype(float)
    rank = (rank - rank.min()) / max(1.0, (rank.max() - rank.min()))
    return 0.1 + 1.9 * rank


def build_coreset(
    df: pd.DataFrame,
    capability_key: str,
    *,
    target: Optional[str] = None,
    time_column: Optional[str] = None,
    n_rows: int = _DEFAULT_CORESET_SIZE,
    seed: int = 0,
) -> tuple[pd.DataFrame, L4CoresetSummary]:
    """Build a coreset DataFrame + a summary; *summary.path is set by caller*."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return df.assign(weight=[]).copy(), L4CoresetSummary(
            capability_key=capability_key, n_rows=1, weight_l2_norm=0.0, path=""
        )

    if capability_key in ("tabular_classification", "temporal_classification", "anomaly_detection", "root_cause_attribution") and target and target in df.columns:
        weights = _classification_weights(df[target])
    elif capability_key in ("tabular_regression", "forecasting") and target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        weights = _regression_weights(df[target])
    elif capability_key == "predictive_maintenance" and time_column and time_column in df.columns:
        weights = _recency_weights(df[time_column])
    else:
        weights = np.ones(n)

    n_take = min(n_rows, n)
    probs = weights / weights.sum() if weights.sum() > 0 else None
    idx = rng.choice(n, size=n_take, replace=False, p=probs) if probs is not None else rng.choice(n, size=n_take, replace=False)
    cs = df.iloc[idx].copy()
    cs_weights = weights[idx]
    # Renormalize selected weights to sum to n_take so downstream learners
    # can treat them as sample weights without scale drift.
    cs_weights = cs_weights * (n_take / cs_weights.sum() if cs_weights.sum() > 0 else 1.0)
    cs["weight"] = cs_weights

    summary = L4CoresetSummary(
        capability_key=capability_key,
        n_rows=int(n_take),
        weight_l2_norm=float(np.linalg.norm(cs_weights)),
        path="",  # filled by caller
    )
    return cs.reset_index(drop=True), summary


def save_coreset(coreset: pd.DataFrame, path: Path) -> None:
    """Save coreset parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coreset.to_parquet(path)


def load_coreset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
