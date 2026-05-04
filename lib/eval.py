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


_BY_CAP = {
    "tabular_classification": ("roc_auc", "average_precision", "log_loss", "brier"),
    "temporal_classification": ("roc_auc", "average_precision", "log_loss", "brier"),
    "tabular_regression": ("rmse", "mae", "r2", "mape"),
    "forecasting": ("mape", "rmse", "smape"),
    "anomaly_detection": ("roc_auc", "average_precision", "precision_at_k"),
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
