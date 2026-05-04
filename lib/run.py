"""Step 3 of the 4-step loop: turn a plan dict into a result row + plots.

Pipeline:
    1. Load the per-capability coreset.
    2. Expand features via the DSL (lib.features.expand_features).
    3. Audit leakage (lib.audit).
    4. Fit + score using the capability splitter and metric set.
    5. Run skeptic (lib.skeptic).
    6. Save plots (residuals-vs-fitted + capability-specific).
    7. Return validated ExperimentResult.

The runner sub-agent (Haiku) calls this; this module is pure Python.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.audit import audit_features
from lib.capabilities import get as get_capability
from lib.eval import dispatch_metrics
from lib.features import expand_features
from lib.registry import build as build_model
from lib.schemas.experiment import ExperimentResult, FitMetrics, SkepticResult, TokenUsage
from lib.schemas.mission import Mission
from lib.schemas.plan import PlanDict
from lib.skeptic import evaluate as skeptic_evaluate
from lib.sketch.l4_coresets import load_coreset
from lib.sketch.l2_joint import load_l2
from lib.sketch.manifest import load_manifest

_log = logging.getLogger("eda.run")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  — must come after `use`


def _load_active_coreset(project_dir: Path, capability_key: str) -> pd.DataFrame:
    manifest = load_manifest(project_dir)
    cs_path = next((p for p in manifest.l4_paths if capability_key in p), None)
    if cs_path is None:
        raise FileNotFoundError(f"no L4 coreset for capability {capability_key}")
    return load_coreset(project_dir / cs_path)


def _capability_for_mission(mission: Mission) -> tuple[str, Any]:
    """Pick the matching capability spec for the MISSION's composition."""
    from lib.capabilities import validate_composition

    spec = validate_composition(mission.capability)
    return spec.key, spec


def _safe_predict(model: Any, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))[:, 1]
        except Exception:  # noqa: BLE001
            return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _save_plots(
    project_dir: Path,
    iteration: int,
    *,
    y_val: np.ndarray,
    yhat_val: np.ndarray,
    feature_names: list[str],
    capability_key: str,
) -> list[str]:
    """Save residuals-vs-fitted + a capability-specific diagnostic plot."""
    iter_dir = project_dir / "results" / f"iter_{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Residuals vs fitted (universal).
    fig, ax = plt.subplots(figsize=(5, 4))
    res = y_val - yhat_val
    ax.scatter(yhat_val, res, s=8, alpha=0.6)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("fitted")
    ax.set_ylabel("residual")
    ax.set_title(f"iter {iteration}: residuals vs fitted")
    p1 = iter_dir / "residuals_vs_fitted.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=100)
    plt.close(fig)

    # Capability-specific.
    p2: Optional[Path] = None
    if capability_key in ("tabular_classification", "temporal_classification", "anomaly_detection", "root_cause_attribution"):
        # Calibration curve.
        try:
            from sklearn.calibration import calibration_curve

            prob_true, prob_pred = calibration_curve(y_val.astype(int), np.clip(yhat_val, 0, 1), n_bins=10)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(prob_pred, prob_true, "o-", label="model")
            ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("empirical rate")
            ax.set_title(f"iter {iteration}: calibration")
            ax.legend()
            p2 = iter_dir / "calibration.png"
            fig.tight_layout()
            fig.savefig(p2, dpi=100)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            _log.debug("calibration plot failed: %s", e)
    elif capability_key in ("tabular_regression", "forecasting"):
        # Predicted-vs-actual scatter.
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_val, yhat_val, s=8, alpha=0.6)
        lo = float(min(y_val.min(), yhat_val.min()))
        hi = float(max(y_val.max(), yhat_val.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_title(f"iter {iteration}: predicted vs actual")
        p2 = iter_dir / "predicted_vs_actual.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=100)
        plt.close(fig)

    out = [str(p.relative_to(project_dir)) for p in (p1, p2) if p is not None]
    return out


def execute_plan(
    project_dir: Path,
    mission: Mission,
    plan: PlanDict,
    *,
    seed: int = 0,
    strict_skeptic: bool = False,
) -> ExperimentResult:
    """Run a single plan dict end-to-end and return its experiment row."""
    t0 = time.time()
    project_dir = Path(project_dir)
    capability_key, capability_spec = _capability_for_mission(mission)

    # 1. Load coreset.
    coreset = _load_active_coreset(project_dir, capability_key)

    # 2. Expand features.
    manifest = load_manifest(project_dir)
    l2 = load_l2(project_dir / manifest.l2_path)
    coreset, concrete_features = expand_features(
        coreset, plan.features, mission, sketch_top_interactions=l2.top_interactions
    )

    # 3. Audit.
    audit = audit_features(mission, concrete_features, plan_area=plan.area)
    if not audit.ok:
        return _failed_experiment(plan, mission, capability_spec, t0, seeds={"numpy": seed},
                                  reason=f"audit failed: {audit.warnings}")

    # 4. Fit / score.
    target = mission.target_column
    if target not in coreset.columns:
        return _failed_experiment(plan, mission, capability_spec, t0, seeds={"numpy": seed},
                                  reason=f"target {target} missing from coreset")

    feats = [f for f in concrete_features if f != target and f in coreset.columns]
    if not feats:
        return _failed_experiment(plan, mission, capability_spec, t0, seeds={"numpy": seed},
                                  reason="no usable features after expansion")

    X = coreset[feats].select_dtypes(include="number").fillna(0).values
    y = coreset[target].values
    sample_w = coreset["weight"].values if "weight" in coreset.columns else None

    splitter = _make_splitter(capability_key)
    splits = splitter(
        n_rows=len(coreset),
        time=coreset[mission.time_column] if mission.time_column and mission.time_column in coreset.columns else None,
        groups=coreset[mission.group_column] if mission.group_column and mission.group_column in coreset.columns else None,
        seed=seed,
        y=y,
    )
    if not splits:
        return _failed_experiment(plan, mission, capability_spec, t0, seeds={"numpy": seed},
                                  reason="splitter returned no folds")

    train_metrics_acc: list[dict[str, float]] = []
    val_metrics_acc: list[dict[str, float]] = []
    last_y_val = last_yhat_val = None

    for fold_i, (tr, va, _te) in enumerate(splits):
        try:
            model = build_model(plan.model, seed=seed + fold_i, **(plan.params or {}))
        except KeyError:
            return _failed_experiment(plan, mission, capability_spec, t0, seeds={"numpy": seed},
                                      reason=f"unknown model {plan.model}")
        try:
            try:
                model.fit(X[tr], y[tr], sample_weight=sample_w[tr] if sample_w is not None else None)
            except TypeError:
                model.fit(X[tr], y[tr])
            yhat_tr = _safe_predict(model, X[tr])
            yhat_va = _safe_predict(model, X[va])
        except Exception as e:  # noqa: BLE001
            return _failed_experiment(plan, mission, capability_spec, t0, seeds={"numpy": seed},
                                      reason=f"fit/predict failed: {e}")
        train_metrics_acc.append(dispatch_metrics(capability_key, y[tr], yhat_tr))
        val_metrics_acc.append(dispatch_metrics(capability_key, y[va], yhat_va))
        last_y_val, last_yhat_val = y[va], yhat_va

    # Average metrics across folds.
    metrics_train = _avg(train_metrics_acc)
    metrics_val = _avg(val_metrics_acc)

    primary_metric = capability_spec.primary_metric
    primary_split = mission.success_criterion.on_split
    pm_value = (
        metrics_val.get(primary_metric)
        if primary_split in ("validation", "test")
        else metrics_train.get(primary_metric)
    )
    if pm_value is None or not np.isfinite(pm_value):
        pm_value = float("nan")

    # 6. Plots.
    plots = _save_plots(
        project_dir,
        plan.iteration,
        y_val=np.asarray(last_y_val, dtype=float),
        yhat_val=np.asarray(last_yhat_val, dtype=float),
        feature_names=feats,
        capability_key=capability_key,
    )

    er = ExperimentResult(
        id=plan.id,
        iteration=plan.iteration,
        hypothesis_id=plan.hypothesis_id,
        model=plan.model,
        features_used=feats,
        params=plan.params,
        calibrated=plan.calibrate,
        technique_family=plan.technique_family,
        area=plan.area,
        metrics=FitMetrics(train=metrics_train, validation=metrics_val),
        primary_metric=primary_metric,
        primary_metric_value=float(pm_value if pm_value is not None else float("nan")),
        skeptic=SkepticResult(verdict="ACCEPT"),  # filled below
        plot_paths=plots,
        seeds={"numpy": seed, "model": seed},
        duration_sec=float(time.time() - t0),
        tokens=TokenUsage(),
    )

    # 7. Skeptic.
    er.skeptic = skeptic_evaluate(er, capability_key, strict=strict_skeptic)
    return er


def _make_splitter(capability_key: str):
    """Find the capability module's splitter."""
    import importlib

    mod = importlib.import_module(f"lib.capabilities.{capability_key}")
    return mod.make_splitter()


def _avg(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(r.keys() for r in rows))
    out: dict[str, float] = {}
    for k in keys:
        vals = [r.get(k, float("nan")) for r in rows]
        clean = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        out[k] = float(sum(clean) / len(clean)) if clean else float("nan")
    return out


def _failed_experiment(
    plan: PlanDict,
    mission: Mission,
    capability_spec: Any,
    t0: float,
    *,
    seeds: dict[str, int],
    reason: str,
) -> ExperimentResult:
    return ExperimentResult(
        id=plan.id,
        iteration=plan.iteration,
        hypothesis_id=plan.hypothesis_id,
        model=plan.model,
        features_used=[],
        params=plan.params,
        calibrated=False,
        technique_family=plan.technique_family,
        area=plan.area,
        metrics=FitMetrics(),
        primary_metric=capability_spec.primary_metric,
        primary_metric_value=float("nan"),
        is_best_so_far=False,
        skeptic=SkepticResult(verdict="FAIL", failed_checks=["execution_failed"], notes=reason),
        seeds=seeds,
        duration_sec=float(time.time() - t0),
        error=reason,
    )
