"""Sketch tool-surface queries (without fit_quick / cross_validate_quick;
those depend on lib.eval and lib.registry which arrive in phase 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.project import create_project
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.sketch import queries
from lib.sketch.builder import build_sketch


@pytest.fixture()
def built_project(tmp_workspace: Path) -> Path:
    proj = create_project(
        name="syn_query",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        token_budget=10_000,
        workspace=tmp_workspace,
    )
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x_temp": np.concatenate([rng.normal(100, 1.0, n // 2), rng.normal(105, 3.0, n - n // 2)]),
            "x_press": rng.normal(1.0, 0.05, n),
            "cat": rng.choice(["A", "B"], n),
            "y": rng.binomial(1, 0.3, n),
        }
    )
    mission = Mission(
        project_name="syn_query",
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


def test_quantile_query(built_project: Path):
    res = queries.quantile(built_project, "x_temp", 0.5)
    assert "value" in res and res["value"] is not None


def test_distribution_query(built_project: Path):
    res = queries.distribution(built_project, "x_temp")
    assert res["dtype"] == "numeric"


def test_cardinality_query(built_project: Path):
    res = queries.cardinality(built_project, "cat")
    assert res["n_unique_estimate"] >= 1


def test_missingness_all_columns(built_project: Path):
    res = queries.missingness(built_project)
    assert "per_column" in res and isinstance(res["per_column"], dict)


def test_top_interactions_query(built_project: Path):
    res = queries.top_interactions(built_project, top_k=3)
    assert "top_interactions" in res
    assert len(res["top_interactions"]) <= 3


def test_principal_components_query(built_project: Path):
    res = queries.principal_components(built_project, top_k=3)
    assert "explained_variance_ratio" in res


def test_regimes_query(built_project: Path):
    res = queries.regimes(built_project)
    assert res["n_regimes"] >= 1


def test_failure_clusters_initially_empty(built_project: Path):
    res = queries.failure_clusters(built_project)
    assert res["clusters"] == []


def test_match_residuals_unmatched(built_project: Path):
    res = queries.match_residuals(built_project, {"primary_metric_value": 0.5})
    assert res["matched"] is False


def test_conditional_dependence(built_project: Path):
    res = queries.conditional_dependence(built_project, "x_temp", "x_press")
    assert "edge_present" in res
