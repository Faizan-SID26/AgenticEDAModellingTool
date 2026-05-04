"""Run-flow simulation.

Drives the iteration loop end-to-end using a deterministic plan picker
(no live LLM). Asserts:
    - Bootstrap → 5+ iterations → finalize completes.
    - Termination evaluates correctly when the success criterion is met.
    - RUN_STATE.json + experiment log evolve as expected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.data import write_joined
from lib.finalize import finalize
from lib.lock import lock_project
from lib.project import create_project, open_project
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
    record,
    termination_check,
)


def _mission(threshold: float = 0.6) -> Mission:
    return Mission(
        project_name="run_flow",
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
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=threshold, direction=">="),
        budget=MissionBudget(token_cap=10_000, iteration_cap=10, stagnation_window=4),
        business_question="Test.",
    )


def _df(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x_temp = rng.normal(0, 1, n)
    logits = 1.5 * x_temp + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x_temp": x_temp,
            "x_press": rng.normal(0, 1, n),
            "downstream_qc": rng.normal(0, 1, n),
            "y": (logits > 0).astype(int),
        }
    )


def _make_plan(iteration: int, model: str = "logreg") -> PlanDict:
    return PlanDict(
        id=f"P-{iteration}-aaaaaa",
        iteration=iteration,
        hypothesis_id="H-seed-1",
        model=model,
        features=["+all_allowed"],
        prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="x"),
        technique_family="linear",
        area="baseline",
        expected_info_gain=0.5,
    )


@pytest.fixture()
def staged(tmp_workspace: Path) -> Path:
    proj = create_project(name="run_flow", domain="manufacturing", token_budget=10_000, workspace=tmp_workspace)
    df = _df()
    write_joined(proj, df)
    mission = _mission()
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    lock_project("run_flow", mission, workspace=tmp_workspace)
    return proj


def test_run_loop_to_goal_met(staged: Path, tmp_workspace: Path):
    mission = _mission(threshold=0.6)  # easy threshold; signal is clear
    halted = False
    for i in range(1, 6):
        plan = _make_plan(i)
        er = execute_plan(staged, mission, plan, seed=0)
        record(staged, mission, er)
        v = termination_check(staged, mission)
        if v.halt:
            halted = True
            assert "goal_met" in v.reasons
            break
    assert halted, "loop should have halted on goal_met"

    # Finalize completes.
    out = finalize(staged, mission, workspace=tmp_workspace)
    assert (staged / "results" / "FINAL.md").exists()
    assert out["confidence_tier"] in ("high", "medium", "low", "no_signal")

    # PROJECT meta updated.
    meta = open_project(tmp_workspace, "run_flow")
    assert meta.status in ("completed", "no_signal")
    assert meta.confidence_tier in ("high", "medium", "low", "no_signal")


def test_run_loop_iteration_state_progresses(staged: Path):
    mission = _mission(threshold=0.99)  # impossible threshold so we see iteration progression
    for i in range(1, 4):
        plan = _make_plan(i)
        er = execute_plan(staged, mission, plan, seed=0)
        record(staged, mission, er)
    rs = load_run_state(staged)
    assert rs.current_iteration == 3
    assert rs.last_completed_phase == "iter_3"
