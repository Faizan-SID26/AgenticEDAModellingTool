"""Capability-dispatched metric tests."""
from __future__ import annotations

import math

import numpy as np

from lib.eval import dispatch_metrics


def test_classification_metrics():
    y = np.array([0, 1, 0, 1, 1, 0])
    yp = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])
    m = dispatch_metrics("tabular_classification", y, yp)
    assert "roc_auc" in m and m["roc_auc"] > 0.9
    assert "average_precision" in m


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.1, 1.9, 2.9, 4.0])
    m = dispatch_metrics("tabular_regression", y, yp)
    assert "rmse" in m and 0 <= m["rmse"] < 0.2
    assert "r2" in m and m["r2"] > 0.9


def test_forecasting_metrics():
    y = np.array([10.0, 12.0, 14.0])
    yp = np.array([11.0, 12.0, 13.0])
    m = dispatch_metrics("forecasting", y, yp)
    assert "mape" in m and m["mape"] < 0.2


def test_anomaly_metrics():
    y = np.array([0, 0, 0, 1, 1])
    yp = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    m = dispatch_metrics("anomaly_detection", y, yp)
    assert "average_precision" in m


def test_ndcg_perfect():
    y = np.array([3, 2, 1, 0])
    yp = np.array([0.9, 0.8, 0.5, 0.1])
    m = dispatch_metrics("root_cause_attribution", y, yp)
    assert math.isclose(m["ndcg_at_10"], 1.0, abs_tol=1e-6)


def test_concordance_index():
    y = np.array([1.0, 2.0, 3.0])
    yp = np.array([0.9, 0.5, 0.1])  # higher risk → smaller event time
    m = dispatch_metrics("predictive_maintenance", y, yp)
    assert m["concordance_index"] >= 0.99
