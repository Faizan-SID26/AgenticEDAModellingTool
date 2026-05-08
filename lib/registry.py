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


# --- Linear breadth -----------------------------------------------------


def _elasticnet(seed: int = 0, **params) -> Any:
    from sklearn.linear_model import ElasticNet
    return ElasticNet(random_state=seed, **params)


def _logreg_l1(seed: int = 0, **params) -> Any:
    from sklearn.linear_model import LogisticRegression
    params.setdefault("penalty", "l1")
    params.setdefault("solver", "liblinear")
    return LogisticRegression(max_iter=1000, random_state=seed, **params)


def _logreg_elastic(seed: int = 0, **params) -> Any:
    from sklearn.linear_model import LogisticRegression
    params.setdefault("penalty", "elasticnet")
    params.setdefault("solver", "saga")
    params.setdefault("l1_ratio", 0.5)
    return LogisticRegression(max_iter=1000, random_state=seed, **params)


def _ridge_classifier(seed: int = 0, **params) -> Any:
    from sklearn.linear_model import RidgeClassifier
    return RidgeClassifier(random_state=seed, **params)


# --- Tree breadth -------------------------------------------------------


def _decision_tree_classifier(seed: int = 0, **params) -> Any:
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=seed, **params)


def _extra_trees_classifier(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier
    params.setdefault("n_estimators", 200)
    return ExtraTreesClassifier(random_state=seed, **params)


def _extra_trees_regressor(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor
    params.setdefault("n_estimators", 200)
    return ExtraTreesRegressor(random_state=seed, **params)


def _random_forest_classifier(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    params.setdefault("n_estimators", 200)
    return RandomForestClassifier(random_state=seed, **params)


def _random_forest_regressor(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import RandomForestRegressor
    params.setdefault("n_estimators", 200)
    return RandomForestRegressor(random_state=seed, **params)


# --- Boosted-tree breadth (lazy XGBoost / CatBoost) ----------------------


def _xgboost_binary(seed: int = 0, **params) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise KeyError(f"xgboost_binary requires `xgboost`; install to use ({e})")
    params.setdefault("n_estimators", 300)
    params.setdefault("tree_method", "hist")
    params.setdefault("eval_metric", "logloss")
    return XGBClassifier(random_state=seed, **params)


def _xgboost_regressor(seed: int = 0, **params) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise KeyError(f"xgboost_regressor requires `xgboost`; install to use ({e})")
    params.setdefault("n_estimators", 300)
    params.setdefault("tree_method", "hist")
    return XGBRegressor(random_state=seed, **params)


def _catboost_binary(seed: int = 0, **params) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise KeyError(f"catboost_binary requires `catboost`; install to use ({e})")
    params.setdefault("iterations", 300)
    params.setdefault("verbose", False)
    return CatBoostClassifier(random_state=seed, **params)


def _catboost_regressor(seed: int = 0, **params) -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise KeyError(f"catboost_regressor requires `catboost`; install to use ({e})")
    params.setdefault("iterations", 300)
    params.setdefault("verbose", False)
    return CatBoostRegressor(random_state=seed, **params)


# --- Loss-function variants ---------------------------------------------


def _lgbm_focal(seed: int = 0, **params) -> Any:
    """LightGBM with focal-loss objective. `alpha` and `gamma` may be passed
    as params; defaults follow the focal-loss paper recommendations."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise KeyError(f"lgbm_focal requires `lightgbm`; install to use ({e})")
    from lib.objectives.focal import lgbm_focal_objective
    alpha = float(params.pop("alpha", 0.25))
    gamma = float(params.pop("gamma", 2.0))
    params.setdefault("n_estimators", 200)
    params.setdefault("verbose", -1)
    return LGBMClassifier(
        random_state=seed,
        objective=lgbm_focal_objective(alpha=alpha, gamma=gamma),
        **params,
    )


def _lgbm_weighted(seed: int = 0, **params) -> Any:
    """LightGBM tagged so the runner passes `sample_weight` to fit. Same
    underlying classifier as `lgbm_binary` — the difference is operational
    (`use_sample_weight=True` in plan.params)."""
    return _lgbm_binary(seed=seed, **params)


def _xgboost_focal(seed: int = 0, **params) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise KeyError(f"xgboost_focal requires `xgboost`; install to use ({e})")
    from lib.objectives.focal import lgbm_focal_objective
    alpha = float(params.pop("alpha", 0.25))
    gamma = float(params.pop("gamma", 2.0))
    params.setdefault("n_estimators", 300)
    params.setdefault("tree_method", "hist")
    return XGBClassifier(
        random_state=seed,
        objective=lgbm_focal_objective(alpha=alpha, gamma=gamma),
        **params,
    )


# --- Neural tabular -----------------------------------------------------


def _mlp_tabular(seed: int = 0, **params) -> Any:
    """sklearn MLP wrapped in a StandardScaler so unscaled tabular features
    don't blow up gradients. Defaults: 2 hidden layers, early stopping."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    params.setdefault("hidden_layer_sizes", (128, 64))
    params.setdefault("activation", "relu")
    params.setdefault("alpha", 1e-4)
    params.setdefault("learning_rate_init", 1e-3)
    params.setdefault("max_iter", 200)
    params.setdefault("early_stopping", True)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(random_state=seed, **params)),
        ]
    )


def _mlp_tabular_regressor(seed: int = 0, **params) -> Any:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    params.setdefault("hidden_layer_sizes", (128, 64))
    params.setdefault("activation", "relu")
    params.setdefault("alpha", 1e-4)
    params.setdefault("learning_rate_init", 1e-3)
    params.setdefault("max_iter", 200)
    params.setdefault("early_stopping", True)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(random_state=seed, **params)),
        ]
    )


def _ft_transformer(seed: int = 0, **params) -> Any:
    try:
        import pytorch_tabular  # type: ignore  # noqa: F401
    except ImportError as e:
        raise KeyError(f"ft_transformer requires `pytorch_tabular`; install to use ({e})")
    # The pytorch_tabular API is involved; a thin trainer wrapper is
    # acceptable here. We import lazily inside to avoid hard-pinning at
    # framework startup.
    from pytorch_tabular import TabularModel  # type: ignore
    from pytorch_tabular.config import (  # type: ignore
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
    )
    from pytorch_tabular.models import FTTransformerConfig  # type: ignore

    class _FTWrap:
        def __init__(self, **kw):
            self.kw = kw
            self._tm: Any = None

        def _build(self, X, y):
            import pandas as pd
            df = pd.DataFrame(X)
            df["__y__"] = y
            cont_cols = [c for c in df.columns if c != "__y__"]
            data = DataConfig(target=["__y__"], continuous_cols=cont_cols)
            model = FTTransformerConfig(task="classification", **self.kw)
            opt = OptimizerConfig()
            trainer = TrainerConfig(max_epochs=20, batch_size=512, auto_lr_find=False, accelerator="cpu")
            self._tm = TabularModel(data_config=data, model_config=model, optimizer_config=opt, trainer_config=trainer)
            self._tm.fit(train=df)
            return self

        def fit(self, X, y, sample_weight=None):
            return self._build(X, y)

        def predict_proba(self, X):
            import pandas as pd
            df = pd.DataFrame(X)
            preds = self._tm.predict(df)
            return preds[["prediction_0", "prediction_1"]].values  # type: ignore[no-any-return]

        def predict(self, X):
            return self.predict_proba(X)[:, 1]

    return _FTWrap(**params)


def _tabnet(seed: int = 0, **params) -> Any:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore
    except ImportError as e:
        raise KeyError(f"tabnet requires `pytorch-tabnet`; install to use ({e})")
    params.setdefault("seed", seed)
    return TabNetClassifier(**params)


# --- Ensembles ----------------------------------------------------------


def _voting_soft(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import VotingClassifier
    estimators = [
        ("lgbm", _lgbm_binary(seed=seed)),
        ("logreg", _logreg(seed=seed)),
        ("rf", _random_forest_classifier(seed=seed)),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", **params)


def _bagging(seed: int = 0, **params) -> Any:
    from sklearn.ensemble import BaggingClassifier
    params.setdefault("n_estimators", 20)
    return BaggingClassifier(random_state=seed, **params)


def _stacked_blend(seed: int = 0, **params) -> Any:
    """Stack lgbm + logreg (+ xgboost if available) under a logistic meta-learner."""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    base: list[tuple[str, Any]] = [
        ("lgbm", _lgbm_binary(seed=seed)),
        ("logreg", _logreg(seed=seed)),
    ]
    try:
        base.append(("xgb", _xgboost_binary(seed=seed)))
    except KeyError:
        pass  # xgboost optional
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False,
        **params,
    )


# --- Anomaly breadth ----------------------------------------------------


def _autoencoder_anomaly(seed: int = 0, **params) -> Any:
    """Reconstruction-error anomaly via a small MLP autoencoder. Uses
    sklearn's MLPRegressor as a stand-in; reconstruction error → score."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    class _AEWrap:
        def __init__(self, hidden_layer_sizes=(32, 8, 32), **kw):
            self.scaler = StandardScaler()
            self.m = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=seed,
                early_stopping=True,
                max_iter=200,
                **kw,
            )

        def fit(self, X, y=None, sample_weight=None):
            X = np.asarray(X, dtype=float)
            Xs = self.scaler.fit_transform(X)
            self.m.fit(Xs, Xs)
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            Xs = self.scaler.transform(X)
            recon = self.m.predict(Xs)
            err = np.mean((Xs - recon) ** 2, axis=1)
            return err

        def predict_proba(self, X):
            scores = self.predict(X)
            mn = scores.min()
            mx = scores.max() or 1.0
            p = (scores - mn) / (mx - mn + 1e-9)
            return np.column_stack([1 - p, p])

    return _AEWrap(**params)


# --- Survival breadth ---------------------------------------------------


def _random_survival_forest(seed: int = 0, **params) -> Any:
    try:
        from sksurv.ensemble import RandomSurvivalForest  # type: ignore
    except ImportError as e:
        raise KeyError(f"random_survival_forest requires `scikit-survival`; install to use ({e})")
    params.setdefault("n_estimators", 100)
    return RandomSurvivalForest(random_state=seed, **params)


def _lgbm_aft(seed: int = 0, **params) -> Any:
    """Accelerated-failure-time LGBM. Falls back to LGBM regressor on log-time."""
    return _lgbm_regressor(seed=seed, **params)


# --- Forecasting breadth ------------------------------------------------


def _prophet(seed: int = 0, **params) -> Any:
    try:
        from prophet import Prophet  # type: ignore
    except ImportError as e:
        raise KeyError(f"prophet requires `prophet`; install to use ({e})")

    class _ProphetWrap:
        def __init__(self, **kw):
            self.kw = kw
            self.m = Prophet(**kw)

        def fit(self, X, y, sample_weight=None):
            import pandas as pd
            df = pd.DataFrame({"ds": pd.to_datetime(np.asarray(X).ravel()), "y": np.asarray(y, dtype=float)})
            self.m.fit(df)
            return self

        def predict(self, X):
            import pandas as pd
            df = pd.DataFrame({"ds": pd.to_datetime(np.asarray(X).ravel())})
            f = self.m.predict(df)
            return f["yhat"].values

    return _ProphetWrap(**params)


def _theta(seed: int = 0, **params) -> Any:
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel  # type: ignore
    except ImportError as e:
        raise KeyError(f"theta requires `statsmodels`; install to use ({e})")

    class _ThetaWrap:
        def __init__(self, **kw):
            self.kw = kw
            self.fitted: Any = None
            self.history: np.ndarray | None = None

        def fit(self, X, y, sample_weight=None):
            self.history = np.asarray(y, dtype=float)
            self.fitted = ThetaModel(self.history, **self.kw).fit()
            return self

        def predict(self, X):
            n = len(X)
            return np.asarray(self.fitted.forecast(steps=n), dtype=float)

    return _ThetaWrap(**params)


def _ets(seed: int = 0, **params) -> Any:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except ImportError as e:
        raise KeyError(f"ets requires `statsmodels`; install to use ({e})")

    class _ETSWrap:
        def __init__(self, **kw):
            self.kw = kw
            self.fitted: Any = None

        def fit(self, X, y, sample_weight=None):
            self.fitted = ExponentialSmoothing(np.asarray(y, dtype=float), **self.kw).fit()
            return self

        def predict(self, X):
            n = len(X)
            return np.asarray(self.fitted.forecast(n), dtype=float)

    return _ETSWrap(**params)


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
    # Linear breadth.
    "elasticnet": _elasticnet,
    "logreg_l1": _logreg_l1,
    "logreg_elastic": _logreg_elastic,
    "ridge_classifier": _ridge_classifier,
    # Tree breadth.
    "decision_tree": _decision_tree_classifier,
    "extra_trees": _extra_trees_classifier,
    "extra_trees_regressor": _extra_trees_regressor,
    "random_forest": _random_forest_classifier,
    "random_forest_regressor": _random_forest_regressor,
    # Boosted-tree breadth.
    "xgboost_binary": _xgboost_binary,
    "xgboost_regressor": _xgboost_regressor,
    "catboost_binary": _catboost_binary,
    "catboost_regressor": _catboost_regressor,
    # Loss variants.
    "lgbm_focal": _lgbm_focal,
    "lgbm_weighted": _lgbm_weighted,
    "xgboost_focal": _xgboost_focal,
    # Neural tabular.
    "mlp_tabular": _mlp_tabular,
    "mlp_tabular_regressor": _mlp_tabular_regressor,
    "ft_transformer": _ft_transformer,
    "tabnet": _tabnet,
    # Ensembles.
    "voting_soft": _voting_soft,
    "bagging": _bagging,
    "stacked_blend": _stacked_blend,
    # Anomaly breadth.
    "autoencoder_anomaly": _autoencoder_anomaly,
    # Survival breadth.
    "random_survival_forest": _random_survival_forest,
    "lgbm_aft": _lgbm_aft,
    # Forecasting breadth.
    "prophet": _prophet,
    "theta": _theta,
    "ets": _ets,
}


# Per-capability registration. is_available() filters wildcards / cross-project
# hypotheses so the framework doesn't propose models that don't fit the
# capability (e.g. prophet for tabular_classification).
_CAPABILITY_REGISTRY: dict[str, frozenset[str]] = {
    "tabular_classification": frozenset({
        "logreg", "ridge_classifier",
        "elasticnet", "logreg_l1", "logreg_elastic",
        "lgbm_binary", "xgboost_binary", "catboost_binary",
        "lgbm_focal", "lgbm_weighted", "xgboost_focal",
        "decision_tree", "extra_trees", "random_forest",
        "voting_soft", "bagging", "stacked_blend",
        "mlp_tabular", "ft_transformer", "tabnet",
    }),
    "temporal_classification": frozenset({
        "logreg", "ridge_classifier", "logreg_l1",
        "lgbm_binary", "xgboost_binary", "catboost_binary",
        "lgbm_focal", "lgbm_weighted", "xgboost_focal",
        "extra_trees", "random_forest",
        "voting_soft", "bagging", "stacked_blend",
        "mlp_tabular",
    }),
    "tabular_regression": frozenset({
        "ridge", "elasticnet",
        "lgbm_regressor", "xgboost_regressor", "catboost_regressor",
        "extra_trees_regressor", "random_forest_regressor",
        "mlp_tabular_regressor",
    }),
    "forecasting": frozenset({
        "naive_seasonal", "ridge_lagged",
        "prophet", "theta", "ets",
        "lgbm_regressor", "xgboost_regressor",
    }),
    "anomaly_detection": frozenset({
        "isolation_forest", "ocsvm", "lof", "autoencoder_anomaly",
        "lgbm_binary", "xgboost_binary",  # supervised-anomaly mode
    }),
    "predictive_maintenance": frozenset({
        "cox_ph", "lgbm_survival", "lgbm_aft", "random_survival_forest",
    }),
    "root_cause_attribution": frozenset({
        "lgbm_binary", "xgboost_binary", "logreg", "permutation_importance",
    }),
}


def is_available(model_key: str, capability_key: str) -> bool:
    """Return True if `model_key` is registered for `capability_key` AND its
    optional dependency is importable. Used by the wildcard generator and
    cross-project hydration so the researcher is never offered a model that
    can't actually be built in the current environment."""
    allowed = _CAPABILITY_REGISTRY.get(capability_key)
    if allowed is None or model_key not in allowed:
        return False
    if model_key not in _MODELS:
        return False
    try:
        # Build the model with no params to surface lazy ImportError as KeyError.
        _MODELS[model_key](seed=0)
        return True
    except KeyError:
        return False
    except Exception:  # noqa: BLE001 — any other build failure also disqualifies
        return False


def models_available_for(capability_key: str) -> list[str]:
    """Convenience: return the available model keys for a given capability."""
    allowed = _CAPABILITY_REGISTRY.get(capability_key, frozenset())
    return sorted(k for k in allowed if is_available(k, capability_key))


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
