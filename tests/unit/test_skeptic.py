"""Capability-dispatched skeptic tests."""
from __future__ import annotations

import math

from lib.schemas.experiment import ExperimentResult, FitMetrics, SkepticResult
from lib.skeptic import evaluate


def _exp(metric_value: float, *, train: dict | None = None, val: dict | None = None,
         model: str = "logreg", n_features: int = 5) -> ExperimentResult:
    return ExperimentResult(
        id="P-1-x",
        iteration=1,
        hypothesis_id="H-seed-1",
        model=model,
        features_used=[f"f{i}" for i in range(n_features)],
        params={},
        calibrated=False,
        technique_family="linear",
        area="baseline",
        metrics=FitMetrics(train=train or {}, validation=val or {}),
        primary_metric="roc_auc",
        primary_metric_value=metric_value,
        skeptic=SkepticResult(verdict="ACCEPT"),
    )


def test_accept_clean():
    e = _exp(0.78, train={"roc_auc": 0.80}, val={"roc_auc": 0.78})
    res = evaluate(e, "tabular_classification")
    assert res.verdict == "ACCEPT"


def test_warn_on_train_val_gap():
    e = _exp(0.6, train={"roc_auc": 0.95}, val={"roc_auc": 0.6})
    res = evaluate(e, "tabular_classification")
    assert res.verdict == "WARN"
    assert "train_val_gap_roc_auc" in res.warnings


def test_fail_on_too_good_to_be_true():
    e = _exp(0.999, train={"roc_auc": 0.999}, val={"roc_auc": 0.999}, n_features=2)
    res = evaluate(e, "tabular_classification")
    # Currently a warning unless strict; promote on strict.
    res2 = evaluate(e, "tabular_classification", strict=True)
    assert res2.verdict == "FAIL"
    assert "too_good_to_be_true_likely_leakage" in res2.failed_checks


def test_fail_on_nan_metrics():
    e = _exp(float("nan"))
    res = evaluate(e, "tabular_classification")
    assert res.verdict == "FAIL"
    assert "primary_metric_non_finite" in res.failed_checks
