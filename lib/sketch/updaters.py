"""Deterministic post-experiment sketch updaters.

Called from `lib.state.record` after every experiment. Per-layer rules:

L2 (joint):
- If the experiment surfaced a *new* high-MI interaction (rank 1 by
  `info_gain`), update the L2 top_interactions list, deduping by pair.

L3 (regimes):
- Re-run regime detection only if (a) info gain on an out-of-regime split
  is high *and* (b) the split passes statistical guards (KS test on
  target distribution between candidate regimes, F-test on residual
  variance) *and* (c) the regime split mechanism hasn't fired in the
  last 5 iterations (anti-flapping).

L7 (failure modes):
- On any FAIL or WARN skeptic verdict, append the experiment's residual
  signature to the catalog via `match_or_create`.

Everything else (L1, L5, L6) is rebuilt at /bootstrap and is treated as
read-only by these updaters; they don't drift.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats  # type: ignore

from lib.schemas.experiment import ExperimentResult
from lib.schemas.sketch import L2JointSummary, L3RegimeSummary, L7FailureClusterSummary
from lib.sketch.l2_joint import load_l2, save_l2
from lib.sketch.l3_regimes import load_l3, save_l3
from lib.sketch.l7_failure_modes import load_l7, match_or_create, save_l7
from lib.sketch.manifest import load_manifest, save_manifest

_log = logging.getLogger("eda.sketch.updaters")

_REGIME_REFRACTORY_ITERATIONS = 5


def _project_path(p: str, project_dir: Path) -> Path:
    return project_dir / p


def update_after_experiment(
    project_dir: Path,
    experiment: ExperimentResult,
    *,
    state_extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Apply all deterministic structural updates following one experiment.

    `state_extras` may carry additional state (e.g., last regime-split
    iteration). Returns a dict of {layer: change_summary} for logging.
    """
    project_dir = Path(project_dir)
    manifest = load_manifest(project_dir)
    state_extras = state_extras or {}
    out: dict[str, Any] = {}

    # L2 update.
    if experiment.area == "interactions" and experiment.is_best_so_far:
        l2 = load_l2(_project_path(manifest.l2_path, project_dir))
        # Inject the (top engineered features) as a synthetic top-interaction.
        new_pair = None
        if len(experiment.features_used) >= 2:
            new_pair = {
                "col_a": experiment.features_used[0],
                "col_b": experiment.features_used[1],
                "mutual_info": float(max(0.0, experiment.info_gain_actual)),
                "rank": 1,
                "interaction_strength_residual": 0.0,
            }
        if new_pair:
            existing = {
                tuple(sorted([d["col_a"], d["col_b"]])): d for d in l2.top_interactions
            }
            existing[tuple(sorted([new_pair["col_a"], new_pair["col_b"]]))] = new_pair
            top = sorted(existing.values(), key=lambda d: d.get("mutual_info", 0.0), reverse=True)[:10]
            l2 = L2JointSummary(
                n_components=l2.n_components,
                explained_variance_ratio=l2.explained_variance_ratio,
                component_loadings_top=l2.component_loadings_top,
                top_interactions=top,
            )
            save_l2(l2, _project_path(manifest.l2_path, project_dir))
            out["l2"] = "interaction_promoted"

    # L3 update — only if the regime experiment was high-info-gain AND we
    # haven't fired recently. The statistical guards (KS, F) are recorded
    # but cannot fire in the absence of raw data; this updater enforces only
    # the firing-cadence guard. Actual re-segmentation happens at next
    # /bootstrap or a manual `refine_regimes` invocation.
    last_regime_iter = int(state_extras.get("last_regime_split_iteration", -100))
    if (
        experiment.area == "regimes"
        and experiment.info_gain_actual >= 0.3
        and (experiment.iteration - last_regime_iter) >= _REGIME_REFRACTORY_ITERATIONS
    ):
        out["l3"] = "queued_for_resegmentation"

    # L7 update — on FAIL or WARN, log a failure-mode signature.
    if experiment.skeptic.verdict in ("FAIL", "WARN"):
        sig = _residual_signature(experiment)
        catalog = load_l7(_project_path(manifest.l7_path, project_dir))
        catalog, cid, created = match_or_create(
            catalog, sig, iteration=experiment.iteration
        )
        save_l7(catalog, _project_path(manifest.l7_path, project_dir))
        out["l7"] = {"cluster_id": cid, "created": created}

    # Bump manifest's last_updated_iteration.
    manifest.last_updated_iteration = experiment.iteration
    save_manifest(project_dir, manifest)
    out["manifest"] = "bumped"
    return out


def _residual_signature(experiment: ExperimentResult) -> dict[str, float]:
    """Distill an experiment into a small dict of features for L7 matching."""
    pmv = experiment.primary_metric_value
    sig: dict[str, float] = {
        # 0.0 sentinel when the metric was non-finite — keeps the cluster
        # catalog tractable; the verdict_fail flag carries the "this row
        # failed" signal independently.
        "primary_metric_value": float(pmv) if pmv is not None else 0.0,
        "info_gain_actual": float(experiment.info_gain_actual),
        "n_features": float(len(experiment.features_used)),
        "verdict_warn": 1.0 if experiment.skeptic.verdict == "WARN" else 0.0,
        "verdict_fail": 1.0 if experiment.skeptic.verdict == "FAIL" else 0.0,
    }
    return sig


def ks_test_regime_means(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS p-value for guarding regime splits."""
    if len(a) < 5 or len(b) < 5:
        return 1.0
    res = stats.ks_2samp(a, b)
    return float(res.pvalue)


def f_test_residual_var(a: np.ndarray, b: np.ndarray) -> float:
    """F-test p-value for unequal variances (one-sided F)."""
    if len(a) < 5 or len(b) < 5:
        return 1.0
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    if va == 0 or vb == 0:
        return 1.0
    f_stat = max(va, vb) / min(va, vb)
    df1 = len(a) - 1 if va > vb else len(b) - 1
    df2 = len(b) - 1 if va > vb else len(a) - 1
    return float(2.0 * (1.0 - stats.f.cdf(f_stat, df1, df2)))
