"""Single iteration end-to-end: data → joined → sketch → plan → execute → record."""
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
from lib.state import (
    load_run_state,
    next as state_next,
    read_experiments,
    record,
    termination_check,
)


def _mission() -> Mission:
    return Mission(
        project_name="iter1",
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
        forbidden_columns=["downstream_qc"],
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
        budget=MissionBudget(token_cap=10_000, iteration_cap=10, stagnation_window=4),
        business_question="Test.",
    )


def _df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_temp = rng.normal(0, 1, n)
    x_press = rng.normal(0, 1, n)
    # y depends on x_temp + noise → roc_auc should be > 0.7 on average.
    logits = 1.5 * x_temp - 0.5 * x_press + rng.normal(0, 0.3, n)
    y = (logits > 0).astype(int)
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x_temp": x_temp,
            "x_press": x_press,
            "downstream_qc": rng.normal(0, 1, n),
            "y": y,
        }
    )


@pytest.fixture()
def iter_project(tmp_workspace: Path) -> Path:
    proj = create_project(name="iter1", domain="manufacturing", token_budget=10_000, workspace=tmp_workspace)
    df = _df()
    # Persist the joined frame for parity with /bootstrap.
    write_joined(proj, df)
    mission = _mission()
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    return proj


def test_state_next_brief(iter_project: Path):
    brief = state_next(iter_project, _mission())
    assert brief.iteration == 1
    assert brief.primary_metric == "roc_auc"
    assert isinstance(brief.bandit_posteriors, dict)
    assert isinstance(brief.suggested_sketch_queries, list)


def test_execute_plan_and_record(iter_project: Path):
    mission = _mission()
    plan = PlanDict(
        id="P-1-aaaaaa",
        iteration=1,
        hypothesis_id="H-seed-1",
        model="logreg",
        features=["+all_allowed"],
        prior_evidence=PriorEvidence(
            kind="hypothesis_seed",
            reference="H-seed-1",
            summary="naive baseline",
        ),
        technique_family="linear",
        area="baseline",
        expected_info_gain=0.5,
    )
    er = execute_plan(iter_project, mission, plan, seed=0)
    assert er.skeptic.verdict in ("ACCEPT", "WARN", "FAIL")
    # Plots saved.
    assert er.plot_paths
    # Metrics non-trivially populated.
    assert "roc_auc" in er.metrics.validation
    # Record updates state.
    summary = record(iter_project, mission, er)
    assert summary["best_so_far"] is not None
    rs = load_run_state(iter_project)
    assert rs.current_iteration == 1
    # Termination check returns a verdict.
    tv = termination_check(iter_project, mission)
    assert isinstance(tv.halt, bool)


def test_audit_blocks_leakage_outside_probe(iter_project: Path):
    mission = _mission()
    bad_plan = PlanDict(
        id="P-2-bbbbbb",
        iteration=2,
        hypothesis_id="H-seed-1",
        model="logreg",
        features=["downstream_qc", "x_temp"],
        prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="x"),
        technique_family="linear",
        area="baseline",  # not leakage_probe → must fail audit
        expected_info_gain=0.5,
    )
    er = execute_plan(iter_project, mission, bad_plan, seed=0)
    assert er.skeptic.verdict == "FAIL"
    assert er.error and "audit" in er.error.lower()
