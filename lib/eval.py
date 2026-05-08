"""Capability-dispatched metrics.

`dispatch_metrics(capability_key, y_true, y_pred)` returns a dict of
metric_name → value computed for that capability. The set of metrics
matches `CapabilitySpec.default_metrics`.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger("eda.eval")


# --- Metric primitives --------------------------------------------------


def _rmse(y, yp) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def _mae(y, yp) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(yp, dtype=float))))


def _mape(y, yp) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    denom = np.where(np.abs(y) > 1e-9, y, np.nan)
    res = np.abs((y - yp) / denom)
    return float(np.nanmean(res))


def _smape(y, yp) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    denom = (np.abs(y) + np.abs(yp)) / 2.0
    denom = np.where(denom > 1e-9, denom, np.nan)
    return float(np.nanmean(np.abs(y - yp) / denom))


def _r2(y, yp) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return float(1.0 - ss_res / ss_tot)


def _roc_auc(y, yp) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, yp))


def _average_precision(y, yp) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, yp))


def _log_loss(y, yp) -> float:
    from sklearn.metrics import log_loss

    y = np.asarray(y).astype(int)
    yp = np.clip(np.asarray(yp, dtype=float), 1e-7, 1 - 1e-7)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(log_loss(y, yp))


def _brier(y, yp) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    return float(np.mean((yp - y) ** 2))


def _precision_at_k(y, yp, k: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size == 0:
        return 0.0
    k_eff = min(k, y.size)
    idx = np.argsort(-yp)[:k_eff]
    return float(y[idx].sum() / k_eff)


def _top_k_size(n_rows: int, k_frac: float) -> int:
    """Number of rows in the top `k_frac` slice. At least 1, at most n_rows."""
    return max(1, min(n_rows, int(round(k_frac * n_rows))))


def _recall_at_top_k_pct(y, yp, k_frac: float = 0.10) -> float:
    """Recall on the top `k_frac` of scores: of all positives in the data,
    what fraction land in the top `k_frac` by predicted score? Ties broken
    by argsort order."""
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size == 0:
        return 0.0
    total_pos = float(y.sum())
    if total_pos <= 0:
        return float("nan")
    n_top = _top_k_size(y.size, k_frac)
    idx = np.argsort(-yp)[:n_top]
    return float(y[idx].sum() / total_pos)


def _precision_at_top_k_pct(y, yp, k_frac: float = 0.10) -> float:
    """Precision on the top `k_frac` of scores: of the top `k_frac` rows by
    score, what fraction are positives?"""
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size == 0:
        return 0.0
    n_top = _top_k_size(y.size, k_frac)
    idx = np.argsort(-yp)[:n_top]
    return float(y[idx].sum() / n_top)


def _lift_at_top_k_pct(y, yp, k_frac: float = 0.10) -> float:
    """Precision on the top `k_frac` divided by the base positive rate. >1
    means the model is enriching positives in the top slice."""
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size == 0:
        return 0.0
    base = float(y.mean())
    if base <= 0:
        return float("nan")
    return float(_precision_at_top_k_pct(y, yp, k_frac) / base)


def _top_k_capture_rate(y, yp, k_frac: float = 0.10) -> float:
    """Alias for recall at top k (semantic name used by deployment teams)."""
    return _recall_at_top_k_pct(y, yp, k_frac)


def _expected_calibration_error(y, yp, n_bins: int = 15) -> float:
    """Standard ECE: weighted average over equal-width bins of |mean(y) -
    mean(yp)|. Lower is better. Returns NaN if `yp` does not look like a
    probability (max > 1 or min < 0)."""
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size == 0:
        return float("nan")
    if float(np.nanmin(yp)) < -1e-9 or float(np.nanmax(yp)) > 1 + 1e-9:
        return float("nan")
    yp_clipped = np.clip(yp, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(y.size)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (yp_clipped >= lo) & (yp_clipped <= hi)
        else:
            mask = (yp_clipped >= lo) & (yp_clipped < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        confidence = float(yp_clipped[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (m / n) * abs(accuracy - confidence)
    return float(ece)


def _ndcg_at_k(y, yp, k: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size == 0:
        return 0.0
    k_eff = min(k, y.size)
    order = np.argsort(-yp)[:k_eff]
    gains = (2 ** y[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, 2 + k_eff))
    dcg = float(np.sum(gains * discounts))
    ideal_order = np.argsort(-y)[:k_eff]
    igains = (2 ** y[ideal_order] - 1)
    idcg = float(np.sum(igains * discounts)) or 1.0
    return float(dcg / idcg)


def _spearman_rank_corr(y, yp) -> float:
    from scipy import stats

    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if y.size < 2:
        return 0.0
    c, _ = stats.spearmanr(y, yp)
    return float(0.0 if np.isnan(c) else c)


def _concordance_index(y, yp) -> float:
    """C-index for survival. y is event_time (continuous), yp is risk.

    For the v1 fallback (event indicator unknown), assume all events
    observed. Pairs are concordant if (yp_i > yp_j) when (y_i < y_j).
    """
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    n = len(y)
    pairs = 0
    concordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            pairs += 1
            if (y[i] < y[j] and yp[i] > yp[j]) or (y[i] > y[j] and yp[i] < yp[j]):
                concordant += 1
    if pairs == 0:
        return 0.5
    return float(concordant / pairs)


# --- Dispatch -----------------------------------------------------------


_TOP_K_PCTS: tuple[int, ...] = (1, 2, 5, 10, 20)
"""Standard top-k slices for deployment-style metrics. Each is registered as
`recall_at_top_pct_<N>`, `precision_at_top_pct_<N>`, `lift_at_top_pct_<N>`."""

_DEFAULT_TOP_K_PCT = 10
"""When the metric name is `recall_at_top_k` (no _pct_<N> suffix), use this
fraction. Lets the user write a short success_criterion.metric while still
getting a defined percent."""


def _top_k_metric_factory(fn, k_pct: int):
    """Bind k_frac=k_pct/100 onto a top-k primitive so it has the standard
    (y, yp) signature the dispatcher expects."""
    k_frac = float(k_pct) / 100.0
    return lambda y, yp: float(fn(y, yp, k_frac))


_TOP_K_METRICS: dict[str, Any] = {}
for _N in _TOP_K_PCTS:
    _TOP_K_METRICS[f"recall_at_top_pct_{_N}"] = _top_k_metric_factory(_recall_at_top_k_pct, _N)
    _TOP_K_METRICS[f"precision_at_top_pct_{_N}"] = _top_k_metric_factory(_precision_at_top_k_pct, _N)
    _TOP_K_METRICS[f"lift_at_top_pct_{_N}"] = _top_k_metric_factory(_lift_at_top_k_pct, _N)


_BY_CAP = {
    "tabular_classification": (
        "roc_auc", "average_precision", "log_loss", "brier",
        "expected_calibration_error",
        "recall_at_top_pct_10", "precision_at_top_pct_10", "lift_at_top_pct_10",
    ),
    "temporal_classification": (
        "roc_auc", "average_precision", "log_loss", "brier",
        "expected_calibration_error",
        "recall_at_top_pct_10", "precision_at_top_pct_10", "lift_at_top_pct_10",
    ),
    "tabular_regression": ("rmse", "mae", "r2", "mape"),
    "forecasting": ("mape", "rmse", "smape"),
    "anomaly_detection": (
        "roc_auc", "average_precision", "precision_at_k",
        "recall_at_top_pct_5", "precision_at_top_pct_5", "lift_at_top_pct_5",
    ),
    "predictive_maintenance": ("concordance_index",),
    "root_cause_attribution": ("ndcg_at_10", "spearman_rank_corr"),
}

_FN: dict[str, Any] = {
    "rmse": _rmse,
    "mae": _mae,
    "mape": _mape,
    "smape": _smape,
    "r2": _r2,
    "roc_auc": _roc_auc,
    "average_precision": _average_precision,
    "log_loss": _log_loss,
    "brier": _brier,
    "precision_at_k": _precision_at_k,
    "ndcg_at_10": lambda y, yp: _ndcg_at_k(y, yp, 10),
    "spearman_rank_corr": _spearman_rank_corr,
    "concordance_index": _concordance_index,
    "ibs": lambda y, yp: float("nan"),  # placeholder
    "cumulative_dynamic_auc": lambda y, yp: float("nan"),  # placeholder
    # Top-k deployment metrics. Short aliases default to 10% so users can
    # write `metric: "recall_at_top_k"` in MISSION.success_criterion without
    # picking a specific percent.
    "recall_at_top_k": _top_k_metric_factory(_recall_at_top_k_pct, _DEFAULT_TOP_K_PCT),
    "precision_at_top_k": _top_k_metric_factory(_precision_at_top_k_pct, _DEFAULT_TOP_K_PCT),
    "lift_at_top_k": _top_k_metric_factory(_lift_at_top_k_pct, _DEFAULT_TOP_K_PCT),
    "top_k_capture_rate": _top_k_metric_factory(_top_k_capture_rate, _DEFAULT_TOP_K_PCT),
    # Calibration.
    "expected_calibration_error": _expected_calibration_error,
    "ece": _expected_calibration_error,
    **_TOP_K_METRICS,
}


def dispatch_metrics(capability_key: str, y_true, y_pred) -> dict[str, float]:
    """Compute the default metric set for a capability."""
    keys = _BY_CAP.get(capability_key)
    if keys is None:
        _log.warning("no metric set for capability %s", capability_key)
        return {}
    out: dict[str, float] = {}
    for k in keys:
        fn = _FN.get(k)
        if fn is None:
            continue
        try:
            out[k] = float(fn(y_true, y_pred))
        except Exception as e:  # noqa: BLE001
            _log.debug("metric %s failed: %s", k, e)
            out[k] = float("nan")
    return out
