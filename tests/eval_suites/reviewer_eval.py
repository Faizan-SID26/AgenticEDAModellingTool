"""Reviewer eval — synthesis scaffold + render produce a well-formed report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.data import write_joined
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
from lib.synthesize import build_scaffold, render_synthesis_md


def _mission() -> Mission:
    return Mission(
        project_name="rev",
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
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
        budget=MissionBudget(token_cap=5_000),
        business_question="x",
    )


@pytest.fixture()
def project_with_one_iter(tmp_workspace: Path) -> Path:
    proj = create_project(name="rev", domain="manufacturing", token_budget=5_000, workspace=tmp_workspace)
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x": rng.normal(0, 1, n),
            "y": rng.binomial(1, 0.4, n),
        }
    )
    write_joined(proj, df)
    mission = _mission()
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    plan = PlanDict(
        id="P-1-aaaaaa",
        iteration=1,
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


def test_scaffold_has_required_keys(project_with_one_iter: Path):
    scaffold = build_scaffold(project_with_one_iter, _mission(), iteration=1)
    for k in (
        "iteration",
        "capability_key",
        "primary_metric",
        "best_so_far",
        "plots_for_vision_review",
        "recent_experiments",
        "bandit_posteriors",
    ):
        assert k in scaffold


def test_render_synthesis_md_no_empty_sections(project_with_one_iter: Path):
    scaffold = build_scaffold(project_with_one_iter, _mission(), iteration=1)
    md = render_synthesis_md(scaffold, reviewer_notes="The model is calibrated.")
    assert "# Synthesis at iteration 1" in md
    assert "Bandit posteriors" in md
    assert "Reviewer notes" in md
    assert "calibrated" in md
