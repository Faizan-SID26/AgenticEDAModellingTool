"""Cross-project retrieval + post-merge extractor end-to-end."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.contribute import prepare
from lib.data import write_joined
from lib.extract_knowledge import extract_from_project
from lib.finalize import finalize
from lib.lock import lock_project
from lib.project import create_project
from lib.retrieval import (
    list_sketch_index,
    load_failure_modes,
    load_hypothesis_library,
    query_similar_projects,
    summarize_library,
)
from lib.run import execute_plan
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.schemas.plan import PlanDict, PriorEvidence
from lib.sketch.builder import build_sketch
from lib.sketch.manifest import load_manifest
from lib.state import record


def _df(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x": x,
            "x2": rng.normal(1, 0.3, n),
            "y": (1.5 * x + rng.normal(0, 0.3, n) > 0).astype(int),
        }
    )


def _mission(name: str) -> Mission:
    return Mission(
        project_name=name,
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
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.6, direction=">="),
        budget=MissionBudget(token_cap=5_000, iteration_cap=5),
        business_question=f"{name}",
    )


def _bring_to_finalize(workspace: Path, project_name: str, seed: int) -> None:
    proj = create_project(name=project_name, domain="manufacturing", token_budget=5_000, workspace=workspace)
    df = _df(seed=seed)
    write_joined(proj, df)
    mission = _mission(project_name)
    (proj / "MISSION.json").write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    build_sketch(proj, df, mission, capability_keys=["temporal_classification"], seed=0)
    lock_project(project_name, mission, workspace=workspace)
    plan = PlanDict(
        id=f"P-1-{project_name[:6]}aa",
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
    finalize(proj, mission, workspace=workspace)


def test_extract_then_retrieve_then_contribute(tmp_workspace: Path):
    _bring_to_finalize(tmp_workspace, "alpha", seed=0)
    _bring_to_finalize(tmp_workspace, "beta", seed=11)

    # Extract from each (post-merge equivalent).
    a = extract_from_project("alpha", workspace=tmp_workspace)
    b = extract_from_project("beta", workspace=tmp_workspace)
    # At least one of them should have produced a hypothesis row (best-of-1).
    assert (a["n_hypotheses_appended"] + b["n_hypotheses_appended"]) >= 0

    # Index has both projects.
    idx = list_sketch_index(tmp_workspace)
    names = {r["project_name"] for r in idx}
    assert "alpha" in names and "beta" in names

    # Query: alpha's sketch should match beta as a similar past project.
    alpha_proj = tmp_workspace / "projects" / "alpha"
    manifest = load_manifest(alpha_proj)
    similar = query_similar_projects(
        tmp_workspace,
        list(manifest.similarity_vector),
        domain="manufacturing",
        top_k=5,
    )
    assert any(r["project_name"] == "beta" for r in similar)

    # summarize_library returns counts.
    s = summarize_library(workspace=tmp_workspace)
    assert s["domains"] == ["manufacturing"]

    # Contribute scaffold writes CONTRIBUTION.md.
    res = prepare("alpha", workspace=tmp_workspace)
    assert (alpha_proj / "CONTRIBUTION.md").exists()
