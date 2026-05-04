"""Hypothesis generation, synthesis, and finalize."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.data import write_joined
from lib.finalize import build_recommendation, finalize, render_final_md, write_final
from lib.generate_hypotheses import generate
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
from lib.state import record
from lib.synthesize import build_scaffold, render_synthesis_md, write_synthesis


def _mission() -> Mission:
    return Mission(
        project_name="phase6",
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
        budget=MissionBudget(token_cap=10_000, iteration_cap=20, stagnation_window=4),
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


@pytest.fixture()
def built(tmp_workspace: Path) -> Path:
    proj = create_project(name="phase6", domain="manufacturing", token_budget=10_000, workspace=tmp_workspace)
    df = _df()
    write_joined(proj, df)
    mission = _mission()
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    # Seed memory/HYPOTHESES.jsonl with universal seeds.
    lock_project("phase6", mission, workspace=tmp_workspace)
    return proj


def test_hypotheses_cold_start(built: Path):
    hyps = generate(built, _mission(), iteration=1)
    assert 1 <= len(hyps) <= 5
    assert all("hypothesis_id" in h for h in hyps)


def test_hypotheses_warm_start(built: Path):
    hyps = generate(built, _mission(), iteration=10)
    assert 1 <= len(hyps) <= 5


def _run_one_iter(built: Path, mission: Mission, iteration: int) -> None:
    plan = PlanDict(
        id=f"P-{iteration}-aaaaaa",
        iteration=iteration,
        hypothesis_id="H-seed-1",
        model="logreg",
        features=["+all_allowed"],
        prior_evidence=PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="x"),
        technique_family="linear",
        area="baseline",
        expected_info_gain=0.5,
    )
    er = execute_plan(built, mission, plan, seed=0)
    record(built, mission, er)


def test_synthesis_scaffold(built: Path):
    mission = _mission()
    _run_one_iter(built, mission, 1)
    scaffold = build_scaffold(built, mission, iteration=1)
    md = render_synthesis_md(scaffold, reviewer_notes="The model is calibrated.")
    p = write_synthesis(built, mission, iteration=1, reviewer_notes="The model is calibrated.")
    assert p.exists()
    assert "Reviewer notes" in p.read_text(encoding="utf-8")


def test_finalize_writes_recommendation(built: Path, tmp_workspace: Path):
    mission = _mission()
    _run_one_iter(built, mission, 1)
    rec = build_recommendation(built, mission)
    md = render_final_md(rec)
    assert rec.confidence_tier in ("high", "medium", "low", "no_signal")
    assert rec.evidence_chain  # at least the best experiment id

    out = finalize(built, mission, workspace=tmp_workspace)
    assert (built / "results" / "FINAL.md").exists()
    assert (built / "results" / "knowledge_bundle.json").exists()
    meta = open_project(tmp_workspace, "phase6")
    assert meta.status in ("completed", "no_signal")
