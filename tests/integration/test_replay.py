"""Deterministic replay test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.data import write_joined
from lib.lock import lock_project
from lib.project import create_project
from lib.replay import replay_project
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


def _df(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x": x,
            "x2": rng.normal(1, 0.3, n),
            "qc": rng.normal(0, 1, n),
            "y": (1.5 * x + rng.normal(0, 0.3, n) > 0).astype(int),
        }
    )


def _mission() -> Mission:
    return Mission(
        project_name="rep",
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
        forbidden_columns=["qc"],
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
        budget=MissionBudget(token_cap=5_000, iteration_cap=5),
        business_question="x.",
    )


@pytest.fixture()
def replayable(tmp_workspace: Path) -> Path:
    proj = create_project(name="rep", domain="manufacturing", token_budget=5_000, workspace=tmp_workspace)
    df = _df()
    # Drop joined parquet under data/ so replay's load_tables can re-read it.
    (proj / "data" / "joined.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proj / "data" / "joined.parquet")
    write_joined(proj, df)
    mission = _mission()
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    lock_project("rep", mission, workspace=tmp_workspace)
    # Run a couple of iterations.
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


def test_replay_runs_clean(replayable: Path, tmp_workspace: Path):
    res = replay_project("rep", workspace=tmp_workspace)
    assert res["replayed"] >= 2
    # Drift on primary metric should be ~0 — exactly identical for these
    # deterministic models.
    for d in res["drift"]:
        if d["original"] is not None and d["replayed"] is not None:
            assert d["abs_delta"] < 1e-6, d
