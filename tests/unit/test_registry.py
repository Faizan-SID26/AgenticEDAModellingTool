"""Model factory tests."""
from __future__ import annotations

import numpy as np
import pytest

from lib.registry import ModelFactory, build, factory


def _xy_class(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _xy_reg(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    y = X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 0.1, n)
    return X, y


def test_logreg_factory():
    X, y = _xy_class()
    m = factory("tabular_classification").default(seed=0)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (200,)


def test_ridge_factory():
    X, y = _xy_reg()
    m = factory("tabular_regression").default(seed=0)
    m.fit(X, y)
    yhat = m.predict(X)
    assert yhat.shape == (200,)


def test_lgbm_or_fallback():
    X, y = _xy_class()
    m = build("lgbm_binary", seed=0)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.min() >= 0 and p.max() <= 1


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build("nonexistent_model")


def test_isolation_forest_predict_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    m = build("isolation_forest", seed=0)
    m.fit(X)
    p = m.predict_proba(X)
    assert p.shape == (200, 2)
    assert (p >= 0).all() and (p <= 1).all()
