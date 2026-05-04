"""End-to-end /contribute (after a finalized project)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.contribute import prepare
from lib.data import write_joined
from lib.finalize import finalize
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


@pytest.fixture()
def finalized(tmp_workspace: Path) -> Path:
    proj = create_project(name="contrib", domain="manufacturing", token_budget=5_000, workspace=tmp_workspace)
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
    mission = Mission(
        project_name="contrib",
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
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.55, direction=">="),
        budget=MissionBudget(token_cap=5_000, iteration_cap=3),
        business_question="x",
    )
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    lock_project("contrib", mission, workspace=tmp_workspace)
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
    finalize(proj, mission, workspace=tmp_workspace)
    return proj


def test_contribute_writes_scaffold(finalized: Path, tmp_workspace: Path):
    res = prepare("contrib", workspace=tmp_workspace)
    p = finalized / "CONTRIBUTION.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "Knowledge to be merged" in text
    assert "git" in res["branch"] or "project" in res["branch"]
