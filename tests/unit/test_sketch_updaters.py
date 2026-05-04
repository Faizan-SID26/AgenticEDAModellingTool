"""Updater tests: deterministic L2/L3/L7 updates after a synthetic experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.project import create_project
from lib.schemas.experiment import ExperimentResult, FitMetrics, SkepticResult
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.sketch.builder import build_sketch
from lib.sketch.l7_failure_modes import load_l7
from lib.sketch.updaters import update_after_experiment


@pytest.fixture()
def built_project(tmp_workspace: Path) -> Path:
    proj = create_project(
        name="syn_upd",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        token_budget=10_000,
        workspace=tmp_workspace,
    )
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x_temp": rng.normal(100, 5, n),
            "x_press": rng.normal(1.0, 0.05, n),
            "y": rng.binomial(1, 0.3, n),
        }
    )
    mission = Mission(
        project_name="syn_upd",
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
        budget=MissionBudget(token_cap=10_000),
        business_question="Test.",
    )
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    return proj


def _make_experiment(verdict: str = "ACCEPT", area: str = "baseline", iteration: int = 1) -> ExperimentResult:
    return ExperimentResult(
        id=f"P-{iteration}-abc",
        iteration=iteration,
        hypothesis_id="H-seed-1",
        model="logreg",
        features_used=["x_temp", "x_press"],
        params={},
        calibrated=False,
        technique_family="linear",
        area=area,
        metrics=FitMetrics(validation={"roc_auc": 0.75}),
        primary_metric="roc_auc",
        primary_metric_value=0.75,
        is_best_so_far=True,
        skeptic=SkepticResult(verdict=verdict),
        info_gain_actual=0.4,
    )


def test_l7_appended_on_warn(built_project: Path):
    e = _make_experiment(verdict="WARN", iteration=1)
    out = update_after_experiment(built_project, e)
    assert "l7" in out and out["l7"]["created"] is True
    catalog = load_l7(built_project / "sketch" / "L7.jsonl")
    assert len(catalog) == 1


def test_l7_match_existing_on_repeat_warn(built_project: Path):
    e1 = _make_experiment(verdict="WARN", iteration=1)
    update_after_experiment(built_project, e1)
    # Second WARN with similar metrics → should match (info_gain_actual identical).
    e2 = _make_experiment(verdict="WARN", iteration=2)
    out = update_after_experiment(built_project, e2)
    catalog = load_l7(built_project / "sketch" / "L7.jsonl")
    assert len(catalog) == 1
    assert out["l7"]["created"] is False


def test_l2_promotion_for_interaction_best(built_project: Path):
    e = _make_experiment(area="interactions", iteration=1)
    out = update_after_experiment(built_project, e)
    assert out.get("l2") == "interaction_promoted"
