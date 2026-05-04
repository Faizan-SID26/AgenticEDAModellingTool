"""Bootstrap end-to-end: load tables → join → build sketch → manifest persists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.data import execute_join_plan, load_tables, write_joined
from lib.project import create_project
from lib.schemas.mission import (
    CapabilityComposition,
    JoinSpec,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.sketch.builder import build_sketch
from lib.sketch.manifest import load_manifest


def _mission_for_join() -> Mission:
    return Mission(
        project_name="bs",
        domain="manufacturing",
        capability=CapabilityComposition(
            temporal_structure="regime_based",
            leakage_model="stage_frontier",
            target_type="binary",
            validation_strategy="time_split",
            recommendation_type="decision",
        ),
        target_column="defect",
        time_column="batch_time",
        join_plan=[
            JoinSpec(
                left_table="process",
                right_table="qa",
                on=["batch_id"],
                how="inner",
            )
        ],
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
        budget=MissionBudget(token_cap=10_000),
        business_question="x",
    )


@pytest.fixture()
def staged(tmp_workspace: Path) -> Path:
    proj = create_project(name="bs", domain="manufacturing", token_budget=10_000, workspace=tmp_workspace)
    rng = np.random.default_rng(0)
    n = 800
    process = pd.DataFrame(
        {
            "batch_id": np.arange(n),
            "batch_time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "reactor_temp": rng.normal(0, 1, n),
        }
    )
    qa = pd.DataFrame(
        {
            "batch_id": np.arange(n),
            "defect": rng.binomial(1, 0.3, n),
        }
    )
    process.to_parquet(proj / "data" / "process.parquet")
    qa.to_parquet(proj / "data" / "qa.parquet")
    return proj


def test_bootstrap_end_to_end(staged: Path):
    mission = _mission_for_join()
    tables = load_tables(staged)
    df = execute_join_plan(tables, mission)
    assert "defect" in df.columns and "reactor_temp" in df.columns and "batch_time" in df.columns

    write_joined(staged, df)
    build_sketch(staged, df, mission, capability_keys=["temporal_classification"], seed=0)
    manifest = load_manifest(staged)
    assert manifest.n_rows_source == len(df)
    assert manifest.l1_path.endswith("L1.json")
    assert manifest.total_size_bytes > 0
