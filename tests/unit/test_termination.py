"""Termination condition tests."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lib.budget import record_event
from lib.project import create_project
from lib.schemas.experiment import ExperimentResult, FitMetrics, SkepticResult
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.state import RunState, append_experiment, save_run_state, termination_check


def _mission(threshold: float = 0.7, iter_cap: int = 5, stagnation: int = 3) -> Mission:
    return Mission(
        project_name="term",
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
        budget=MissionBudget(token_cap=1000, iteration_cap=iter_cap, stagnation_window=stagnation, catastrophic_failure_window=2),
        business_question="x",
    )


@pytest.fixture()
def proj(tmp_workspace: Path) -> Path:
    p = create_project(name="term", domain="manufacturing", token_budget=1000, workspace=tmp_workspace)
    save_run_state(p, RunState(project_name="term"))
    return p


def test_no_halt_initially(proj: Path):
    v = termination_check(proj, _mission())
    assert v.halt is False


def test_halts_on_goal_met(proj: Path):
    rs = RunState(project_name="term", best_primary_metric_value=0.85, current_iteration=2)
    save_run_state(proj, rs)
    v = termination_check(proj, _mission(threshold=0.7))
    assert v.halt and "goal_met" in v.reasons


def test_halts_on_budget(proj: Path):
    record_event(proj, iteration=0, event="bootstrap", role="researcher", cap=1000, input_tokens=2000)
    v = termination_check(proj, _mission())
    assert v.halt and "budget_exhausted" in v.reasons


def test_halts_on_stagnation(proj: Path):
    rs = RunState(project_name="term", iterations_since_improvement=10, current_iteration=4, best_primary_metric_value=0.4)
    save_run_state(proj, rs)
    v = termination_check(proj, _mission(threshold=0.99, stagnation=3))
    assert v.halt and "stagnation" in v.reasons


def test_halts_on_iteration_cap(proj: Path):
    rs = RunState(project_name="term", current_iteration=5, best_primary_metric_value=0.4)
    save_run_state(proj, rs)
    v = termination_check(proj, _mission(threshold=0.99, iter_cap=5))
    assert v.halt and "iteration_cap" in v.reasons


def test_halts_on_catastrophic_skeptic(proj: Path):
    for i in range(2):
        e = ExperimentResult(
            id=f"P-{i+1}-x",
            iteration=i + 1,
            hypothesis_id="H",
            model="logreg",
            features_used=["a"],
            params={},
            calibrated=False,
            technique_family="linear",
            area="baseline",
            metrics=FitMetrics(),
            primary_metric="roc_auc",
            primary_metric_value=0.4,
            skeptic=SkepticResult(verdict="FAIL", failed_checks=["primary_metric_non_finite"]),
        )
        append_experiment(proj, e)
    v = termination_check(proj, _mission(threshold=0.99))
    assert v.halt and any("catastrophic" in r for r in v.reasons)
