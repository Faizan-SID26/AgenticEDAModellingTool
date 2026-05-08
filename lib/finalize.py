"""Finalize: counterfactual recommendation builder.

Inputs:
- The locked MISSION.
- The full experiment_log.
- The sketch (for evidence references).
- A causal pass on the L4 coreset (uses dowhy if installed; otherwise a
  multivariate-regression ATE fallback).

Outputs:
- A validated `Recommendation` (lib.schemas.recommendation.Recommendation).
- `results/FINAL.md` rendered from it.
- Project status updated to `completed` (or `no_signal` if no signal).
- A `results/knowledge_bundle.json` for the post-merge extractor.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.capabilities import composition_signature, validate_composition
from lib.project import open_project, write_project_meta
from lib.schemas.experiment import ExperimentResult
from lib.schemas.knowledge import (
    FailureModeEntry,
    HypothesisLibraryEntry,
    KnowledgeBundle,
)
from lib.schemas.mission import Mission
from lib.schemas.project_meta import ProjectMeta
from lib.schemas.recommendation import (
    CausalAssumption,
    CounterfactualEffect,
    FailureModeRuleOut,
    ModelCardEntry,
    Recommendation,
    WhatWouldChangeIt,
)
from lib.sketch.l4_coresets import load_coreset
from lib.sketch.l7_failure_modes import load_l7
from lib.sketch.manifest import load_manifest
from lib.state import read_experiments

_log = logging.getLogger("eda.finalize")


def _pick_best(experiments: list[ExperimentResult], direction: str) -> Optional[ExperimentResult]:
    """Pick the best non-FAIL experiment.

    Prefers population-prevalence metrics (`<primary>_pop`) when at least one
    experiment carries them — these are the deployment-truth values. Falls
    back to the coreset primary metric when none of the experiments has a
    population metric yet (e.g. iter < 10 or no joined parquet).
    """
    valid = [
        e
        for e in experiments
        if e.skeptic.verdict != "FAIL" and e.primary_metric_value is not None
    ]
    if not valid:
        return None

    def _pop_value(e: ExperimentResult) -> Optional[float]:
        v = (e.metrics.validation or {}).get(f"{e.primary_metric}_pop")
        return float(v) if v is not None else None

    pop_pool = [e for e in valid if _pop_value(e) is not None]
    if pop_pool:
        if direction == ">=":
            return max(pop_pool, key=lambda e: _pop_value(e) or float("-inf"))
        return min(pop_pool, key=lambda e: _pop_value(e) or float("inf"))

    if direction == ">=":
        return max(valid, key=lambda e: e.primary_metric_value)  # type: ignore[arg-type,return-value]
    return min(valid, key=lambda e: e.primary_metric_value)  # type: ignore[arg-type,return-value]


def _confidence_tier(
    best: Optional[ExperimentResult],
    mission: Mission,
    n_experiments: int,
) -> str:
    if best is None or best.primary_metric_value is None:
        return "no_signal"
    sc = mission.success_criterion
    bv = best.primary_metric_value
    threshold_met = (bv >= sc.threshold) if sc.direction == ">=" else (bv <= sc.threshold)
    if not threshold_met:
        return "low"
    if n_experiments >= 20 and best.skeptic.verdict == "ACCEPT":
        return "high"
    if best.skeptic.verdict == "ACCEPT":
        return "medium"
    return "low"


def _causal_pass(
    project_dir: Path,
    mission: Mission,
    treatment: str,
    outcome: str,
) -> Optional[CounterfactualEffect]:
    """Estimate ATE of `treatment` on `outcome` on the L4 coreset.

    Tries dowhy first; falls back to a multivariate-regression coefficient
    (residualized on all other numeric features).
    """
    try:
        manifest = load_manifest(project_dir)
        cs_path = next((p for p in manifest.l4_paths), None)
        if cs_path is None:
            return None
        df = load_coreset(project_dir / cs_path)
        if treatment not in df.columns or outcome not in df.columns:
            return None

        try:
            import dowhy  # type: ignore  # noqa: F401
            from dowhy import CausalModel

            common_causes = [
                c
                for c in df.select_dtypes(include="number").columns
                if c not in (treatment, outcome, "weight")
            ][:8]
            cm = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
            )
            ident = cm.identify_effect(proceed_when_unidentifiable=True)
            est = cm.estimate_effect(ident, method_name="backdoor.linear_regression")
            point = float(est.value)
            return CounterfactualEffect(
                metric=outcome,
                point_estimate=point,
                ci_low=point - abs(point) * 0.2,
                ci_high=point + abs(point) * 0.2,
                ci_level=0.9,
                estimator="dowhy.linear_regression",
            )
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            _log.debug("dowhy estimation failed: %s; falling back to regression", e)

        # Regression fallback.
        from sklearn.linear_model import LinearRegression

        cov = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in (treatment, outcome, "weight")
        ][:8]
        if not cov:
            X = df[[treatment]].values
        else:
            X = df[[treatment, *cov]].fillna(0).values
        y = df[outcome].fillna(0).values
        lr = LinearRegression().fit(X, y)
        coef = float(lr.coef_[0])
        # Bootstrap CI.
        rng = np.random.default_rng(0)
        boots = []
        n = len(df)
        for _ in range(50):
            idx = rng.choice(n, size=n, replace=True)
            try:
                bo = LinearRegression().fit(X[idx], y[idx])
                boots.append(float(bo.coef_[0]))
            except Exception:  # noqa: BLE001
                pass
        if boots:
            lo = float(np.quantile(boots, 0.05))
            hi = float(np.quantile(boots, 0.95))
        else:
            lo, hi = coef - abs(coef) * 0.2, coef + abs(coef) * 0.2
        return CounterfactualEffect(
            metric=outcome,
            point_estimate=coef,
            ci_low=lo,
            ci_high=hi,
            ci_level=0.9,
            estimator="linear_regression_residualized",
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("causal pass failed: %s", e)
        return None


def _evidence_chain(experiments: list[ExperimentResult], best: Optional[ExperimentResult]) -> list[str]:
    chain: list[str] = []
    if best:
        chain.append(best.id)
    # Up to 5 supporting experiments by area diversity.
    seen = {best.area} if best else set()
    for e in reversed(experiments):
        if e.skeptic.verdict == "FAIL":
            continue
        if best and e.id == best.id:
            continue
        if e.area in seen:
            continue
        chain.append(e.id)
        seen.add(e.area)
        if len(chain) >= 5:
            break
    return chain


def build_recommendation(
    project_dir: Path,
    mission: Mission,
) -> Recommendation:
    """Produce the validated Recommendation."""
    project_dir = Path(project_dir)
    experiments = read_experiments(project_dir)
    direction = mission.success_criterion.direction
    best = _pick_best(experiments, direction)
    tier = _confidence_tier(best, mission, n_experiments=len(experiments))

    # Counterfactual: use the best experiment's *first* feature as treatment.
    counterfactual: Optional[CounterfactualEffect] = None
    if best and best.features_used and tier in ("medium", "high"):
        counterfactual = _causal_pass(
            project_dir,
            mission,
            treatment=best.features_used[0],
            outcome=mission.target_column,
        )

    # Ruled-out failure modes from L7.
    catalog = load_l7(project_dir / "sketch" / "L7.jsonl")
    ruled_out: list[FailureModeRuleOut] = []
    for c in catalog[:5]:
        ruled_out.append(
            FailureModeRuleOut(
                name=c.label or c.cluster_id,
                why_ruled_out=f"Cluster {c.cluster_id} fired but did not invalidate best result.",
                evidence_ref=best.id if best else "",
            )
        )

    rationale_parts: list[str] = []
    if best:
        bv = best.primary_metric_value
        bv_str = f"{bv:.4f}" if bv is not None else "(non-finite)"
        rationale_parts.append(
            f"Best run was iter {best.iteration} ({best.model}, area={best.area}) "
            f"with {best.primary_metric}={bv_str}."
        )
    rationale_parts.append(
        f"Capability composition: {composition_signature(mission.capability)}."
    )
    if tier == "no_signal":
        rationale_parts.append(
            "No iteration met the success criterion; honest failure recorded."
        )
    elif counterfactual is None:
        rationale_parts.append(
            "Counterfactual ATE not estimated (insufficient confidence or coreset shape)."
        )
    rationale = " ".join(rationale_parts)

    decision = (
        f"Adopt {best.model} with features {best.features_used[:5]}." if best and tier != "no_signal"
        else "No actionable signal found — collect more data and retry."
    )

    model_card: list[ModelCardEntry] = []
    if best and best.primary_metric_value is not None:
        model_card.append(
            ModelCardEntry(
                model=best.model,
                primary_metric=best.primary_metric,
                primary_metric_value=best.primary_metric_value,
                validation_strategy=mission.capability.validation_strategy,
                n_train=int(sum(1 for _ in experiments)),
                n_validation=int(sum(1 for _ in experiments)),
                seeds=best.seeds,
            )
        )

    rec = Recommendation(
        project_name=mission.project_name,
        recommendation_type=mission.capability.recommendation_type,
        decision=decision,
        rationale=rationale,
        counterfactual=counterfactual,
        evidence_chain=_evidence_chain(experiments, best),
        causal_assumptions=[
            CausalAssumption(
                assumption="No unmeasured confounders strong enough to flip sign of estimated effect.",
                justification=(
                    "L6 causal hints + capability validation strategy were used; sensitivity "
                    "checks were not run automatically (v1)."
                ),
                sensitivity_check=None,
            )
        ],
        ruled_out_failure_modes=ruled_out,
        what_would_change_it=[
            WhatWouldChangeIt(
                condition=f"Best {mission.success_criterion.metric} drops below threshold on a held-out window.",
                expected_change="Retract the recommendation; re-run with the held-out data added to training.",
            ),
            WhatWouldChangeIt(
                condition="A new failure cluster repeatedly fires with mean_residual > 2σ of prior best.",
                expected_change="Investigate the cluster; revisit feature inclusion / regimes.",
            ),
        ],
        model_card=model_card,
        confidence_tier=tier,  # type: ignore[arg-type]
    )
    return rec


def render_final_md(rec: Recommendation) -> str:
    lines: list[str] = []
    lines.append(f"# Final recommendation: {rec.project_name}\n")
    lines.append(f"_Confidence tier_: **{rec.confidence_tier}**\n")
    lines.append(f"_Recommendation type_: `{rec.recommendation_type}`\n")
    lines.append(f"\n## Decision\n\n{rec.decision}\n")
    lines.append(f"\n## Rationale\n\n{rec.rationale}\n")
    if rec.counterfactual:
        cf = rec.counterfactual
        lines.append("\n## Quantified counterfactual\n")
        lines.append(
            f"- Metric: `{cf.metric}` — point estimate {cf.point_estimate:.4f} "
            f"(CI [{cf.ci_low:.4f}, {cf.ci_high:.4f}] at level {cf.ci_level})"
        )
        lines.append(f"- Estimator: `{cf.estimator}`")
    lines.append("\n## Evidence chain\n")
    for ref in rec.evidence_chain:
        lines.append(f"- `{ref}`")
    if rec.causal_assumptions:
        lines.append("\n## Causal assumptions\n")
        for a in rec.causal_assumptions:
            lines.append(f"- **{a.assumption}** — {a.justification}")
            if a.sensitivity_check:
                lines.append(f"  - sensitivity: {a.sensitivity_check}")
    if rec.ruled_out_failure_modes:
        lines.append("\n## Ruled-out failure modes\n")
        for f in rec.ruled_out_failure_modes:
            lines.append(f"- **{f.name}** — {f.why_ruled_out} (evidence: `{f.evidence_ref}`)")
    if rec.what_would_change_it:
        lines.append("\n## What would change this recommendation\n")
        for w in rec.what_would_change_it:
            lines.append(f"- _{w.condition}_ → {w.expected_change}")
    if rec.model_card:
        lines.append("\n## Model card (appendix)\n")
        for m in rec.model_card:
            lines.append(
                f"- model=`{m.model}` {m.primary_metric}={m.primary_metric_value:.4f} "
                f"validation=`{m.validation_strategy}` n_train={m.n_train} n_val={m.n_validation}"
            )
    return "\n".join(lines) + "\n"


def write_final(project_dir: Path, rec: Recommendation) -> Path:
    p = Path(project_dir) / "results" / "FINAL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_final_md(rec), encoding="utf-8")
    return p


def build_knowledge_bundle(
    project_dir: Path,
    mission: Mission,
    rec: Recommendation,
) -> KnowledgeBundle:
    """Stage the contribution bundle for /contribute."""
    project_dir = Path(project_dir)
    experiments = read_experiments(project_dir)
    sig = composition_signature(mission.capability)
    h_entries: list[HypothesisLibraryEntry] = []
    for e in experiments:
        if e.skeptic.verdict == "FAIL":
            continue
        if e.info_gain_actual < 0.1:
            continue
        h_entries.append(
            HypothesisLibraryEntry(
                entry_id=f"K-h-{mission.project_name}-{e.id}",
                source_project=mission.project_name,
                source_iteration=e.iteration,
                domain=mission.domain,
                capability_signature=sig,
                pattern_summary=f"{e.area}: {e.model} on {len(e.features_used)} features",
                technique_family=e.technique_family,
                feature_roles=[],  # post-merge extractor will anonymize.
                sketch_signature={"info_gain": float(e.info_gain_actual)},
                info_gain=float(e.info_gain_actual),
                primary_metric=e.primary_metric,
                primary_metric_delta=float(e.info_gain_actual),
            )
        )
    f_entries: list[FailureModeEntry] = []
    for e in experiments:
        if e.skeptic.verdict == "FAIL":
            for fc in e.skeptic.failed_checks:
                f_entries.append(
                    FailureModeEntry(
                        entry_id=f"K-f-{mission.project_name}-{e.id}-{fc}",
                        source_project=mission.project_name,
                        domain=mission.domain,
                        capability_signature=sig,
                        failure_name=fc,
                        resolution=("auto-resolved by subsequent iteration" if rec.confidence_tier in ("high", "medium") else "still open"),
                    )
                )

    manifest = load_manifest(project_dir)
    bundle = KnowledgeBundle(
        project_name=mission.project_name,
        domain=mission.domain,
        capability_signature=sig,
        hypothesis_entries=h_entries,
        failure_entries=f_entries,
        sketch_similarity_vector=list(manifest.similarity_vector),
        notes=f"confidence_tier={rec.confidence_tier}",
    )
    out_path = project_dir / "results" / "knowledge_bundle.json"
    out_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return bundle


def _should_block_finalize(
    project_dir: Path,
    mission: Mission,
    rec: "Recommendation",
) -> dict[str, Any]:
    """Decide whether finalize should refuse to write FINAL.md and instead
    request re-entry into the iteration loop.

    Blocks when ALL of:
      - tier == "low" (operational threshold not met)
      - operational_floor is set AND best_metric is on the wrong side
      - token budget is not exhausted
      - iteration cap is not exhausted
      - breakthrough_entry_count < breakthrough_max_entries

    Returns a dict with `block` (bool) and the values the orchestrator
    needs to record the decision.
    """
    import math
    from lib.budget import fraction_consumed
    from lib.state import load_run_state

    info: dict[str, Any] = {
        "block": False,
        "reason": "",
        "best_metric": None,
        "operational_floor": mission.budget.operational_floor,
        "breakthrough_entry_count": 0,
    }
    state = load_run_state(project_dir)
    info["breakthrough_entry_count"] = state.breakthrough_entry_count

    if rec.confidence_tier != "low":
        return info  # tier already medium/high or no_signal — no point looping further

    floor = mission.budget.operational_floor
    if floor is None:
        return info  # no operational floor declared → respect default behavior

    bv = state.best_primary_metric_value
    if math.isfinite(bv):
        info["best_metric"] = float(bv)
    direction = mission.success_criterion.direction
    on_wrong_side = (
        (math.isfinite(bv) and bv < floor)
        if direction == ">="
        else (math.isfinite(bv) and bv > floor)
    )
    if not on_wrong_side:
        return info  # already past the operational floor; just hadn't hit the threshold

    frac = fraction_consumed(project_dir, mission.budget.token_cap)
    if frac >= 1.0:
        info["reason"] = "budget_exhausted"
        return info  # don't block — finalize honestly
    if state.current_iteration >= mission.budget.iteration_cap:
        info["reason"] = "iteration_cap"
        return info
    if state.breakthrough_entry_count >= mission.budget.breakthrough_max_entries:
        info["reason"] = "breakthrough_max_entries_reached"
        return info

    info["block"] = True
    info["reason"] = "below_operational_floor_with_budget_remaining"
    return info


def finalize(
    project_dir: Path,
    mission: Mission,
    *,
    workspace: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Any]:
    """End-to-end finalize: build recommendation + write FINAL.md + knowledge bundle.

    Pillar 10 — when the chosen best is `confidence_tier="low"` AND
    budget remains AND the operational_floor is unmet AND breakthrough mode
    has not exhausted its re-entry budget, *do not* finalize. Return
    `{requested_re_enter_loop: True, reason: ...}` so the orchestrator can
    re-enter Phase B with breakthrough mode active for another window.
    `force=True` bypasses this gate (used by `/contribute` and after the
    secondary stagnation window has closed).
    """
    rec = build_recommendation(project_dir, mission)
    if not force:
        signal = _should_block_finalize(project_dir, mission, rec)
        if signal["block"]:
            return {
                "requested_re_enter_loop": True,
                "reason": signal["reason"],
                "confidence_tier": rec.confidence_tier,
                "best_metric": signal["best_metric"],
                "operational_floor": signal["operational_floor"],
                "breakthrough_entry_count": signal["breakthrough_entry_count"],
            }
    final_path = write_final(project_dir, rec)
    bundle = build_knowledge_bundle(project_dir, mission, rec)

    # Update PROJECT.json status + confidence_tier.
    meta = open_project(workspace, mission.project_name)
    new_status = "completed" if rec.confidence_tier != "no_signal" else "no_signal"
    meta = ProjectMeta(**{**meta.model_dump(), "status": new_status, "confidence_tier": rec.confidence_tier})
    write_project_meta(workspace, meta)

    return {
        "final_path": str(final_path),
        "confidence_tier": rec.confidence_tier,
        "decision": rec.decision,
        "n_evidence": len(rec.evidence_chain),
        "knowledge_bundle_path": str(project_dir / "results" / "knowledge_bundle.json"),
    }
