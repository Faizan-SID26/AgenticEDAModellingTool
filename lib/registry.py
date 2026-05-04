"""Model factories per capability.

`factory(capability_key)` returns a `ModelFactory` whose `default(seed=...)`
gives a reasonable starter model. `build(model_key, **params)` constructs
a specific named model.

All models conform to a sklearn-like interface: `fit(X, y, sample_weight=None)`
and either `predict_proba` (for binary classifiers) or `predict`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_log = logging.getLogger("eda.registry")


# --- Lazy-imported model builders ---------------------------------------


def _logreg(seed: int = 0, **params) -> Any:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=seed, **params)


def _ridge(seed: int = 0, **params) -> Any:
    from sklearn.linear_model import Ridge
    return Ridge(random_state=seed, **params)


def _lgbm_binary(seed: int = 0, **params) -> Any:
    """LightGBM binary classifier with conservative defaults; falls back to GradientBoosting."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_estimators=200, num_leaves=31, verbose=-1, **params)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=seed, **params)


def _lgbm_regressor(seed: int = 0, **params) -> Any:
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_estimators=200, num_leaves=31, verbose=-1, **params)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=seed, **params)


def _isolation_forest(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import IsolationForest

    class _IFWrapper:
        def __init__(self, seed: int, **kw):
            self.m = IsolationForest(random_state=seed, **kw)

        def fit(self, X, y=None, sample_weight=None):
            self.m.fit(X)
            return self

        def predict_proba(self, X):
            scores = -self.m.score_samples(X)  # higher = more anomalous
            # Map to [0,1] via min-max for AP compatibility.
            mn = scores.min()
            mx = scores.max() or 1.0
            p = (scores - mn) / (mx - mn + 1e-9)
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return self.predict_proba(X)[:, 1]

    return _IFWrapper(seed, **params)


def _ocsvm(seed: int = 0, **params) -> Any:
    from sklearn.svm import OneClassSVM

    class _OCSVMWrapper:
        def __init__(self, **kw):
            self.m = OneClassSVM(**kw)

        def fit(self, X, y=None, sample_weight=None):
            self.m.fit(X)
            return self

        def predict(self, X):
            return -self.m.decision_function(X)

        def predict_proba(self, X):
            scores = self.predict(X)
            mn = scores.min()
            mx = scores.max() or 1.0
            p = (scores - mn) / (mx - mn + 1e-9)
            return np.column_stack([1 - p, p])

    return _OCSVMWrapper(**params)


def _lof(seed: int = 0, **params) -> Any:
    from sklearn.neighbors import LocalOutlierFactor

    class _LOFWrapper:
        def __init__(self, **kw):
            kw.setdefault("novelty", True)
            self.m = LocalOutlierFactor(**kw)

        def fit(self, X, y=None, sample_weight=None):
            self.m.fit(X)
            return self

        def predict(self, X):
            return -self.m.decision_function(X)

        def predict_proba(self, X):
            scores = self.predict(X)
            mn = scores.min()
            mx = scores.max() or 1.0
            p = (scores - mn) / (mx - mn + 1e-9)
            return np.column_stack([1 - p, p])

    return _LOFWrapper(**params)


def _cox_ph(seed: int = 0, **params) -> Any:
    """Cox proportional hazards via lifelines if available; fallback: ridge on time-to-event."""
    try:
        from lifelines import CoxPHFitter

        class _CoxWrap:
            def __init__(self):
                self.m = CoxPHFitter()
                self.cols: list[str] = []

            def fit(self, X, y, sample_weight=None):
                import pandas as pd

                df = pd.DataFrame(X)
                self.cols = list(df.columns)
                df["__T__"] = np.asarray(y, dtype=float)
                df["__E__"] = 1
                self.m.fit(df, duration_col="__T__", event_col="__E__")
                return self

            def predict(self, X):
                import pandas as pd

                df = pd.DataFrame(X, columns=self.cols)
                return self.m.predict_partial_hazard(df).values

        return _CoxWrap()
    except ImportError:
        return _ridge(seed=seed)


def _lgbm_survival(seed: int = 0, **params) -> Any:
    """No proper survival LGBM in v1: fall back to LGBM regressor predicting time-to-event."""
    return _lgbm_regressor(seed=seed)


def _naive_seasonal(seed: int = 0, **params) -> Any:
    """Naive forecaster: y_hat[t] = y[t - period]. Period inferred at fit time."""
    class _Naive:
        def __init__(self, period: int = 7):
            self.period = period
            self.history: np.ndarray | None = None

        def fit(self, X, y, sample_weight=None):
            self.history = np.asarray(y, dtype=float)
            return self

        def predict(self, X):
            n = len(X)
            if self.history is None or self.history.size == 0:
                return np.zeros(n)
            tail = self.history[-self.period :]
            reps = int(np.ceil(n / self.period))
            return np.tile(tail, reps)[:n]

    return _Naive(**(params or {}))


def _ridge_lagged(seed: int = 0, **params) -> Any:
    """Lagged ridge: assume X already includes lags; thin wrapper around Ridge."""
    return _ridge(seed=seed, **params)


def _permutation_importance_dummy(seed: int = 0, **params) -> Any:
    """Stand-in for the 'permutation_importance' default model — uses an LGBM regressor."""
    return _lgbm_regressor(seed=seed, **params)


_MODELS: dict[str, Callable[..., Any]] = {
    "logreg": _logreg,
    "ridge": _ridge,
    "lgbm_binary": _lgbm_binary,
    "lgbm_regressor": _lgbm_regressor,
    "isolation_forest": _isolation_forest,
    "ocsvm": _ocsvm,
    "lof": _lof,
    "cox_ph": _cox_ph,
    "lgbm_survival": _lgbm_survival,
    "naive_seasonal": _naive_seasonal,
    "ridge_lagged": _ridge_lagged,
    "permutation_importance": _permutation_importance_dummy,
}


@dataclass
class ModelFactory:
    """Wraps the per-capability default plus a builder for arbitrary keys."""

    capability_key: str
    default_key: str

    def default(self, seed: int = 0, **params: Any) -> Any:
        return self.build(self.default_key, seed=seed, **params)

    def build(self, model_key: str, seed: int = 0, **params: Any) -> Any:
        builder = _MODELS.get(model_key)
        if builder is None:
            raise KeyError(f"unknown model key: {model_key}")
        return builder(seed=seed, **params)


_DEFAULT_MODEL_BY_CAP = {
    "tabular_classification": "logreg",
    "temporal_classification": "lgbm_binary",
    "tabular_regression": "ridge",
    "forecasting": "naive_seasonal",
    "anomaly_detection": "isolation_forest",
    "predictive_maintenance": "cox_ph",
    "root_cause_attribution": "lgbm_binary",
}


def factory(capability_key: str) -> ModelFactory:
    default = _DEFAULT_MODEL_BY_CAP.get(capability_key, "logreg")
    return ModelFactory(capability_key=capability_key, default_key=default)


def build(model_key: str, seed: int = 0, **params: Any) -> Any:
    """Build any registered model directly by key."""
    builder = _MODELS.get(model_key)
    if builder is None:
        raise KeyError(f"unknown model key: {model_key}")
    return builder(seed=seed, **params)
