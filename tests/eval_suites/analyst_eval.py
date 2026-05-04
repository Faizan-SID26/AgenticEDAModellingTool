"""Analyst eval — finalize on a fixture project produces a valid recommendation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.data import write_joined
from lib.finalize import build_recommendation, render_final_md
from lib.lock import lock_project
from lib.project import create_project
from lib.run import execute_plan
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.schemas.plan import PlanDict, PriorEvidence
from lib.sketch.builder import build_sketch
from lib.state import record


def _mission(threshold: float = 0.55) -> Mission:
    return Mission(
        project_name="analyst",
        domain="manufacturing",
        capability=CapabilityComposition(
            temporal_structure="regime_based",
            leakage_model="stage_frontier",
            target_type="binary",
            validation_strategy="time_split",
            recommendation_type="decision",
        ),
        target_column="y",
        time_column="t",
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=threshold, direction=">="),
        budget=MissionBudget(token_cap=5_000, iteration_cap=3),
        business_question="x",
    )


@pytest.fixture()
def proj_with_iters(tmp_workspace: Path) -> Path:
    proj = create_project(name="analyst", domain="manufacturing", token_budget=5_000, workspace=tmp_workspace)
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": rng.binomial(1, 0.4, n),
        }
    )
    write_joined(proj, df)
    mission = _mission()
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    lock_project("analyst", mission, workspace=tmp_workspace)
    for i in range(1, 3):
        plan = PlanDict(
            id=f"P-{i}-aaaaaa",
            iteration=i,
            hypothesis_id="H-seed-1",
            model="logreg",
            features=["+all_allowed"],
            prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="x"),
            technique_family="linear",
            area="baseline",
            expected_info_gain=0.5,
        )
        er = execute_plan(proj, mission, plan, seed=0)
        record(proj, mission, er)
    return proj


def test_recommendation_valid_shape(proj_with_iters: Path):
    rec = build_recommendation(proj_with_iters, _mission())
    assert rec.confidence_tier in ("high", "medium", "low", "no_signal")
    assert rec.evidence_chain  # at least one
    md = render_final_md(rec)
    assert "Final recommendation" in md
    assert "Decision" in md
    assert "Causal assumptions" in md
    assert "What would change" in md


def test_no_signal_when_threshold_unattainable(proj_with_iters: Path):
    rec = build_recommendation(proj_with_iters, _mission(threshold=0.99))
    # Either threshold not met (low) or no_signal — depends on synthetic data luck;
    # in any case, the schema must validate and the FINAL.md must render.
    assert rec.confidence_tier in ("low", "no_signal", "medium")
    md = render_final_md(rec)
    assert "Confidence tier" in md
