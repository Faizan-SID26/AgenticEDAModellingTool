"""L3: regime / change-point structure.

Tries `ruptures` PELT first; falls back to a CUSUM-on-mean detector if
ruptures isn't installed. Per-regime mini-summaries are kept slim
(target rate / mean per regime).

Caps:
- Max regimes: 8 (more is rarely actionable; we'd rather signal "no
  regime structure" than overfit).
- Min regime size: max(50, n // 20).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib.schemas.sketch import L3RegimeSummary

_log = logging.getLogger("eda.sketch.l3")

_MAX_REGIMES = 8
_MIN_REGIME_FRACTION = 0.05


def _detect_changepoints_pelt(signal: np.ndarray, *, pen: float = 10.0) -> list[int]:
    """Try ruptures.PELT for change-point detection."""
    try:
        import ruptures as rpt  # type: ignore
    except ImportError:
        return []
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    bkps = algo.predict(pen=pen)
    return [int(b) for b in bkps[:-1]]  # drop trailing N


def _detect_changepoints_cusum(signal: np.ndarray, *, threshold: float = 4.0) -> list[int]:
    """CUSUM fallback. Sliding mean shift detector with magnitude `threshold` * sigma."""
    if signal.size < 20:
        return []
    sig = pd.Series(signal).ffill().bfill().values
    mu = float(np.mean(sig))
    sd = float(np.std(sig, ddof=1)) or 1.0
    z = (sig - mu) / sd
    pos = np.zeros_like(z)
    neg = np.zeros_like(z)
    points: list[int] = []
    for i in range(1, len(z)):
        pos[i] = max(0.0, pos[i - 1] + z[i] - 0.5)
        neg[i] = min(0.0, neg[i - 1] + z[i] + 0.5)
        if pos[i] > threshold or neg[i] < -threshold:
            points.append(i)
            pos[i] = 0.0
            neg[i] = 0.0
    return points


def _enforce_min_size(boundaries: list[int], n: int) -> list[int]:
    """Drop boundaries that produce regimes shorter than the min size."""
    if not boundaries:
        return []
    min_sz = max(50, int(n * _MIN_REGIME_FRACTION))
    kept: list[int] = []
    prev = 0
    for b in sorted(set(boundaries)):
        if b - prev < min_sz:
            continue
        kept.append(b)
        prev = b
    if n - prev < min_sz and kept:
        kept.pop()  # last regime too small
    return kept[: _MAX_REGIMES - 1]


def build_l3(
    df: pd.DataFrame,
    *,
    time_column: Optional[str] = None,
    target: Optional[str] = None,
    primary_columns: Optional[list[str]] = None,
) -> L3RegimeSummary:
    """Build L3 regime summary on a numeric signal (target if given, else first numeric)."""
    if df.empty:
        return L3RegimeSummary(n_regimes=1, boundary_indices=[], regime_sizes=[len(df)])

    df_sorted = df
    if time_column and time_column in df.columns:
        df_sorted = df.sort_values(time_column).reset_index(drop=True)

    if target and target in df_sorted.columns and pd.api.types.is_numeric_dtype(df_sorted[target]):
        signal_col = target
    else:
        num = df_sorted.select_dtypes(include=[np.number])
        if num.empty:
            return L3RegimeSummary(n_regimes=1, boundary_indices=[], regime_sizes=[len(df_sorted)])
        signal_col = num.columns[0]

    signal = df_sorted[signal_col].astype(float).ffill().bfill().values
    boundaries = _detect_changepoints_pelt(signal) or _detect_changepoints_cusum(signal)
    boundaries = _enforce_min_size(boundaries, len(signal))

    # Build regime indices.
    edges = [0, *boundaries, len(signal)]
    sizes: list[int] = []
    target_dist: list[dict[str, float]] = []
    means: dict[str, list[float]] = {}
    primary_columns = primary_columns or [signal_col]
    if target and target in df_sorted.columns and target not in primary_columns:
        primary_columns.append(target)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        seg = df_sorted.iloc[lo:hi]
        sizes.append(int(hi - lo))
        for col in primary_columns:
            if col in seg.columns and pd.api.types.is_numeric_dtype(seg[col]):
                means.setdefault(col, []).append(float(seg[col].mean()))
        if target and target in seg.columns:
            t = seg[target]
            if pd.api.types.is_numeric_dtype(t):
                target_dist.append(
                    {
                        "mean": float(t.mean()),
                        "stdev": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
                        "rate": float((t > 0).mean()),  # binary-friendly
                    }
                )
            else:
                vc = t.value_counts(normalize=True)
                top = vc.head(1)
                target_dist.append(
                    {
                        "top_class_rate": float(top.iloc[0]) if len(top) else 0.0,
                        "n_unique": float(len(vc)),
                    }
                )

    return L3RegimeSummary(
        n_regimes=len(sizes),
        boundary_indices=list(map(int, boundaries)),
        regime_means=means,
        regime_sizes=sizes,
        regime_target_distribution=target_dist,
    )


def save_l3(summary: L3RegimeSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def load_l3(path: Path) -> L3RegimeSummary:
    return L3RegimeSummary.model_validate_json(Path(path).read_text(encoding="utf-8"))
