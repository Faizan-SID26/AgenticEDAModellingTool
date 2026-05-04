"""L1: per-column distribution summary.

Implementation notes:
- Quantiles: exact via numpy on the build-time sample (the source dataset
  is loaded once at /bootstrap; nothing about quantile recomputation runs
  per-iteration). The schema field `quantiles` is the percentile→value
  dump; `t-digest accuracy` semantics are not visible to the agent.
- Cardinality: HyperLogLog via `datasketch` if available; otherwise an
  exact unique count is recorded with a flag.
- Top categories: pandas `value_counts` head-K.

Output is a list[L1ColumnSummary] persisted as a JSON file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.schemas.sketch import L1ColumnSummary

_log = logging.getLogger("eda.sketch.l1")

_DEFAULT_PERCENTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
_TOP_CATEGORIES = 8


def _hll_estimate(values: pd.Series) -> int:
    """Return a HyperLogLog cardinality estimate (or exact if HLL missing)."""
    try:
        from datasketch import HyperLogLog  # type: ignore

        h = HyperLogLog(p=12)
        for v in values.dropna().astype(str):
            h.update(v.encode("utf-8"))
        return int(h.count())
    except ImportError:
        _log.debug("datasketch not available; using exact unique count")
        return int(values.nunique(dropna=True))


def _classify_dtype(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        n = len(s)
        nunq = s.nunique(dropna=True)
        return "categorical" if nunq <= max(50, int(n**0.5)) else "text"
    return "categorical"


def summarize_column(s: pd.Series) -> L1ColumnSummary:
    """Build an L1ColumnSummary for one pandas Series."""
    dtype = _classify_dtype(s)
    n_total = int(len(s))
    n_missing = int(s.isna().sum())
    n_unique = _hll_estimate(s)

    quantiles: dict[str, float] = {}
    top_cats: list[tuple[str, int]] = []
    mean = stdev = None

    if dtype == "numeric" and n_total - n_missing > 0:
        valid = s.dropna().astype(float).values
        for p in _DEFAULT_PERCENTILES:
            quantiles[f"{p:.2f}"] = float(np.quantile(valid, p))
        mean = float(np.mean(valid))
        stdev = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    elif dtype == "categorical":
        vc = s.value_counts(dropna=True).head(_TOP_CATEGORIES)
        top_cats = [(str(k), int(v)) for k, v in vc.items()]

    return L1ColumnSummary(
        column=str(s.name),
        dtype=dtype,  # type: ignore[arg-type]
        n_total=n_total,
        n_missing=n_missing,
        n_unique_estimate=n_unique,
        quantiles=quantiles,
        top_categories=top_cats,
        mean=mean,
        stdev=stdev,
    )


def build_l1(df: pd.DataFrame) -> list[L1ColumnSummary]:
    """Build per-column summaries for every column in `df`."""
    return [summarize_column(df[c]) for c in df.columns]


def save_l1(summaries: list[L1ColumnSummary], path: Path) -> None:
    """Write L1 summaries as a JSON list (one entry per column)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.model_dump() for s in summaries]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_l1(path: Path) -> list[L1ColumnSummary]:
    """Load L1 summaries from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [L1ColumnSummary.model_validate(r) for r in raw]
