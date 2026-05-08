"""Population-prevalence evaluator.

The L4 coreset upweights minority classes (classification) or tail rows
(regression) so fast baselines don't lose signal at small sample sizes.
This shifts the empirical distribution. Metrics computed on the coreset
are therefore *not* directly comparable to metrics on the original
population — the AP / recall / etc. seen during /run can flatter what the
model would actually deliver in deployment.

`evaluate_at_population(...)` re-runs the same metric dispatch as
`lib.run.execute_plan`, but on the full `joined.parquet` at original
prevalence. It is invoked by the runner every 10 iterations and by
`finalize` on the chosen best experiment, with all metrics suffixed
`_pop` so coreset and population values can coexist on the same
ExperimentResult row.

The full-table load is cached per process via `functools.lru_cache` keyed
on the parquet path so repeated calls within one /run are cheap.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.eval import dispatch_metrics
from lib.features import expand_features
from lib.schemas.mission import Mission
from lib.schemas.plan import PlanDict
from lib.sketch.manifest import load_manifest

_log = logging.getLogger("eda.eval_population")


@lru_cache(maxsize=4)
def _load_joined_cached(joined_path: str) -> pd.DataFrame:
    """Process-local LRU. Joined parquet rarely changes within a /run, so
    a single cached load is adequate."""
    return pd.read_parquet(joined_path)


def _safe_predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Match `lib.run._safe_predict` semantics so coreset and population
    evaluation use the same prediction shape."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba, dtype=float).ravel()
        except Exception:  # noqa: BLE001
            pass
    return np.asarray(model.predict(X), dtype=float)


def joined_parquet_path(project_dir: Path) -> Optional[Path]:
    """Return the path to the joined parquet, or None if it isn't present."""
    project_dir = Path(project_dir)
    candidates = [
        project_dir / "data" / "_joined.parquet",
        project_dir / "_joined.parquet",
    ]
    try:
        manifest = load_manifest(project_dir)
        if getattr(manifest, "joined_path", None):
            candidates.insert(0, project_dir / manifest.joined_path)
    except Exception:  # noqa: BLE001
        pass
    for p in candidates:
        if p.exists():
            return p
    return None


def evaluate_at_population(
    model: Any,
    plan: PlanDict,
    mission: Mission,
    project_dir: Path,
    capability_key: str,
    *,
    sketch_top_interactions: Optional[list[dict]] = None,
) -> dict[str, float]:
    """Compute the capability metric set at original-population prevalence.

    Returns a flat dict like `{"recall_at_top_pct_10_pop": 0.32, ...}` so
    the values can be merged into `experiment.metrics.validation` without
    colliding with the coreset metrics. Returns {} when the joined parquet
    is absent (e.g. tests, or pre-bootstrap states).

    The evaluator never re-fits the model — it only scores. This keeps cost
    bounded to a single full-table prediction pass.
    """
    project_dir = Path(project_dir)
    joined_path = joined_parquet_path(project_dir)
    if joined_path is None:
        _log.debug("evaluate_at_population: no joined parquet under %s", project_dir)
        return {}
    df = _load_joined_cached(str(joined_path))
    if mission.target_column not in df.columns:
        _log.debug("evaluate_at_population: target %s missing from %s", mission.target_column, joined_path)
        return {}

    # Apply the same DSL expansion as lib.run.execute_plan so the model
    # gets the same column shape it was trained on.
    try:
        expanded, concrete = expand_features(
            df,
            list(plan.features or []),
            mission,
            sketch_top_interactions=sketch_top_interactions,
        )
    except Exception as e:  # noqa: BLE001 — never fail iteration on pop eval
        _log.debug("evaluate_at_population: feature expand failed: %s", e)
        return {}

    feats = [f for f in concrete if f != mission.target_column and f in expanded.columns]
    if not feats:
        return {}

    X = expanded[feats].select_dtypes(include="number").fillna(0).values
    y = expanded[mission.target_column].values
    if X.shape[0] != y.shape[0] or X.shape[1] == 0:
        return {}

    try:
        yhat = _safe_predict(model, X)
    except Exception as e:  # noqa: BLE001
        _log.debug("evaluate_at_population: predict failed: %s", e)
        return {}

    metrics = dispatch_metrics(capability_key, y, yhat)
    return {f"{k}_pop": float(v) for k, v in metrics.items() if v is not None}


def cache_clear() -> None:
    """Drop the joined-parquet cache. Useful in tests."""
    _load_joined_cached.cache_clear()
