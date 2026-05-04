"""Integration test for /plan.

Simulates user answers to the question batches and asserts:
    - assemble_mission produces a valid Mission.
    - lock_project writes MISSION + COLUMNS + JOIN_PLAN + HYPOTHESES + bumps PROJECT.
    - Universal seeds + recipe seeds + domain seeds end up in HYPOTHESES.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.inspect import inspect_project
from lib.lock import lock_project
from lib.planning import (
    assemble_mission,
    build_followup_batch,
    build_initial_batch,
    collect_resolved_answers,
    load_recipe,
)
from lib.project import create_project, open_project
from lib.schemas.question import QuestionAnswer


@pytest.fixture()
def planned_project(tmp_workspace: Path) -> Path:
    proj = create_project(
        name="syn_plan",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        token_budget=20_000,
        workspace=tmp_workspace,
    )
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "batch_id": np.arange(n),
            "batch_time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "reactor_temp": rng.normal(100, 5, n),
            "downstream_qc": rng.normal(0, 1, n),
            "defect": rng.binomial(1, 0.1, n),
        }
    )
    df.to_parquet(proj / "data" / "joined.parquet")
    inspect_project(proj)
    return proj


def _simulate_user(batch, accept_inference: bool = True, overrides: dict[str, object] | None = None):
    """Build QuestionAnswer objects accepting inferred values by default."""
    overrides = overrides or {}
    out: list[QuestionAnswer] = []
    for q in batch.questions:
        if q.target_mission_path in overrides:
            out.append(
                QuestionAnswer(
                    question_id=q.question_id,
                    answer=overrides[q.target_mission_path],
                    confirmed_inference=False,
                )
            )
            continue
        if q.kind == "confirm_inference" and accept_inference:
            out.append(
                QuestionAnswer(
                    question_id=q.question_id,
                    answer=q.inferred_answer,
                    confirmed_inference=True,
                )
            )
            continue
        # free_text default
        out.append(QuestionAnswer(question_id=q.question_id, answer=""))
    return out


def test_planning_full_flow(planned_project: Path, tmp_workspace: Path):
    # Recipe + project meta loaded from the workspace where it lives.
    proj_dir = planned_project
    meta = open_project(tmp_workspace, "syn_plan")

    # The test workspace has no recipes/, so recipe will be None — but the
    # mission can still be assembled if we provide capability via answers.
    # To keep this realistic, we copy the recipe from the repo.
    repo_root = Path(__file__).resolve().parents[2]
    recipe_src = repo_root / "recipes" / "manufacturing_defect_classification.json"
    (tmp_workspace / "recipes").mkdir(exist_ok=True)
    (tmp_workspace / "recipes" / "manufacturing_defect_classification.json").write_text(
        recipe_src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    recipe = load_recipe(tmp_workspace, meta.recipe)
    assert recipe is not None

    profile = json.loads((proj_dir / "memory" / "INIT_PROFILE.json").read_text(encoding="utf-8"))
    batch1 = build_initial_batch(profile, recipe, meta.domain)
    batch1.answers = _simulate_user(batch1, accept_inference=True)
    answers = collect_resolved_answers([batch1])

    batch2 = build_followup_batch(profile, recipe, answers, iteration=1)
    batch2.answers = _simulate_user(
        batch2,
        accept_inference=True,
        overrides={"business_question": "Predict downstream defects from upstream sensors."},
    )

    answers = collect_resolved_answers([batch1, batch2])
    mission = assemble_mission(
        project_name=meta.project_name,
        domain_key=meta.domain,
        recipe=recipe,
        answers=answers,
        token_budget=meta.token_budget,
    )
    assert mission.target_column == "defect"
    assert mission.time_column == "batch_time"
    assert mission.business_question.startswith("Predict")
    assert "downstream_qc" in mission.forbidden_columns

    artifacts = lock_project(meta.project_name, mission, workspace=tmp_workspace, recipe=recipe)
    for k in ("mission", "columns", "join_plan", "hypotheses", "project_meta"):
        assert artifacts[k].exists(), k

    # PROJECT status bumped.
    meta2 = open_project(tmp_workspace, meta.project_name)
    assert meta2.status == "planned"

    # HYPOTHESES.jsonl contains the 5 universal seeds plus recipe + domain seeds.
    hyp_lines = (proj_dir / "memory" / "HYPOTHESES.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(l) for l in hyp_lines if l.strip()]
    universal_ids = {f"H-seed-{i}" for i in range(1, 6)}
    assert universal_ids.issubset({p["hypothesis_id"] for p in parsed})
    # At least one recipe seed and one domain seed.
    assert any(p.get("source") == "recipe" for p in parsed)
    assert any(p.get("source") == "domain" for p in parsed)
