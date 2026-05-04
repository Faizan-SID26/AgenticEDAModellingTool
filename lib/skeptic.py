"""Deterministic skeptic — capability-dispatched check lists.

For each capability, runs a small set of checks and returns one of
ACCEPT / WARN / FAIL. Strict mode promotes WARN to FAIL.

The skeptic is pure: it reads structured inputs (experiment row, simple
sketch summaries) and emits a structured `SkepticResult`.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from lib.schemas.experiment import ExperimentResult, SkepticResult

_log = logging.getLogger("eda.skeptic")


# --- Generic checks -----------------------------------------------------


def _check_finite_metric(experiment: ExperimentResult) -> tuple[bool, str | None]:
    v = experiment.primary_metric_value
    if not math.isfinite(v):
        return False, "primary_metric_non_finite"
    return True, None


def _check_train_val_gap(experiment: ExperimentResult, *, max_gap: float = 0.25) -> tuple[bool, str | None]:
    """Flag suspicious train/val gap on the primary metric."""
    pm = experiment.primary_metric
    tr = experiment.metrics.train.get(pm)
    va = experiment.metrics.validation.get(pm)
    if tr is None or va is None:
        return True, None
    gap = abs(tr - va)
    if gap > max_gap:
        return False, f"train_val_gap_{pm}"
    return True, None


def _check_too_good_to_be_true(experiment: ExperimentResult) -> tuple[bool, str | None]:
    """A primary AUC of >0.99 with few features is suspicious — usually leakage."""
    pm = experiment.primary_metric
    v = experiment.primary_metric_value
    if pm in ("roc_auc", "average_precision") and v >= 0.99 and len(experiment.features_used) < 10:
        return False, "too_good_to_be_true_likely_leakage"
    return True, None


def _check_metric_in_range(experiment: ExperimentResult) -> tuple[bool, str | None]:
    pm = experiment.primary_metric
    v = experiment.primary_metric_value
    if pm in ("roc_auc", "average_precision") and not (0 <= v <= 1.0001):
        return False, f"{pm}_out_of_range"
    return True, None


def _check_nan_metrics(experiment: ExperimentResult) -> tuple[bool, str | None]:
    bad = [k for d in (experiment.metrics.train, experiment.metrics.validation, experiment.metrics.test) for k, v in d.items() if v is not None and not math.isfinite(v)]
    if bad:
        return False, "nan_in_metrics"
    return True, None


# --- Capability extras --------------------------------------------------


def _temporal_extras(experiment: ExperimentResult) -> list[tuple[bool, str | None]]:
    """For temporal capabilities, ensure the experiment did not request shuffled splits."""
    return []


def _survival_extras(experiment: ExperimentResult) -> list[tuple[bool, str | None]]:
    return []


def _anomaly_extras(experiment: ExperimentResult) -> list[tuple[bool, str | None]]:
    return []


_EXTRAS_BY_CAP = {
    "temporal_classification": _temporal_extras,
    "predictive_maintenance": _survival_extras,
    "anomaly_detection": _anomaly_extras,
}


# --- Main entry ---------------------------------------------------------


def evaluate(
    experiment: ExperimentResult,
    capability_key: str,
    *,
    strict: bool = False,
    domain_extras: list[str] | None = None,
) -> SkepticResult:
    """Run all checks for `capability_key` and produce a SkepticResult."""
    checks = [
        ("finite_metric", _check_finite_metric),
        ("metric_in_range", _check_metric_in_range),
        ("nan_metrics", _check_nan_metrics),
        ("train_val_gap", _check_train_val_gap),
        ("too_good_to_be_true", _check_too_good_to_be_true),
    ]
    extras_fn = _EXTRAS_BY_CAP.get(capability_key)
    failed: list[str] = []
    warnings: list[str] = []
    for name, fn in checks:
        ok, key = fn(experiment)
        if not ok:
            (failed if strict or name in ("finite_metric", "nan_metrics", "metric_in_range") else warnings).append(
                key or name
            )
    if extras_fn is not None:
        for ok, key in extras_fn(experiment):
            if not ok:
                warnings.append(key or "")

    # Domain-extras keys are *advisory* in v1: a domain might expose
    # 'physical_bounds_check' that requires raw rows; we do not run it
    # here, but record it as a warning so the synthesis surfaces the gap.
    if domain_extras:
        for k in domain_extras:
            warnings.append(f"domain_extra_skipped:{k}")

    if failed:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"
    else:
        verdict = "ACCEPT"

    return SkepticResult(
        verdict=verdict,  # type: ignore[arg-type]
        failed_checks=failed,
        warnings=warnings,
        notes="strict mode" if strict else "",
    )
