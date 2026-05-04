"""Planner eval — schema validation + heuristic checks against fixture INIT_PROFILEs.

No API calls. Asserts that the deterministic planner machinery
(`build_initial_batch`, `assemble_mission`, `lock_project`) produces a
locked MISSION that:
    - Validates the capability composition.
    - Names the right target / time / forbidden columns.
    - Carries the recipe-default success criterion when accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.lock import lock_project
from lib.planning import (
    assemble_mission,
    build_initial_batch,
    collect_resolved_answers,
    load_recipe,
)
from lib.project import create_project, open_project
from lib.schemas.question import QuestionAnswer


_FIXTURE_PROFILE = {
    "schema_version": "1",
    "framework_version": "0.1.0",
    "project_dir": ".",
    "n_files": 1,
    "files": [
        {
            "table_name": "joined",
            "path": "data/joined.parquet",
            "n_rows": 5000,
            "n_columns": 5,
            "columns": [
                {"name": "batch_time", "dtype": "datetime", "n_rows": 5000, "n_missing": 0, "n_unique": 5000},
                {"name": "reactor_temp", "dtype": "numeric", "n_rows": 5000, "n_missing": 0, "n_unique": 4999, "min": 80.0, "max": 120.0, "mean": 100.0, "stdev": 5.0},
                {"name": "reactor_pressure", "dtype": "numeric", "n_rows": 5000, "n_missing": 0, "n_unique": 4900, "mean": 1.0, "stdev": 0.05},
                {"name": "downstream_qc_score", "dtype": "numeric", "n_rows": 5000, "n_missing": 0, "n_unique": 4998, "mean": 0.0, "stdev": 1.0},
                {"name": "defect", "dtype": "categorical", "n_rows": 5000, "n_missing": 0, "n_unique": 2, "top_categories": [["0", 4500], ["1", 500]]},
            ],
            "likely_time": ["batch_time"],
            "likely_target": ["defect"],
            "likely_id": [],
        }
    ],
    "proposed_joins": [],
}


def test_planner_full_assembly(tmp_workspace: Path):
    # Materialize a fixture project + INIT_PROFILE.
    proj = create_project(
        name="ev_planner",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        token_budget=10_000,
        workspace=tmp_workspace,
    )
    (proj / "memory" / "INIT_PROFILE.json").write_text(
        json.dumps(_FIXTURE_PROFILE, indent=2), encoding="utf-8"
    )
    # Copy recipe.
    repo_root = Path(__file__).resolve().parents[2]
    recipe_src = repo_root / "recipes" / "manufacturing_defect_classification.json"
    (tmp_workspace / "recipes").mkdir(exist_ok=True)
    (tmp_workspace / "recipes" / "manufacturing_defect_classification.json").write_text(
        recipe_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    meta = open_project(tmp_workspace, "ev_planner")
    recipe = load_recipe(tmp_workspace, meta.recipe)
    batch = build_initial_batch(_FIXTURE_PROFILE, recipe, meta.domain)
    # Simulate user-confirms-everything.
    batch.answers = [
        QuestionAnswer(question_id=q.question_id, answer=q.inferred_answer, confirmed_inference=True)
        if q.kind == "confirm_inference"
        else QuestionAnswer(question_id=q.question_id, answer="Test bq.")
        for q in batch.questions
    ]
    answers = collect_resolved_answers([batch])
    answers["business_question"] = "Predict downstream defects from upstream signals."
    mission = assemble_mission(
        project_name="ev_planner",
        domain_key="manufacturing",
        recipe=recipe,
        answers=answers,
        token_budget=10_000,
    )
    arts = lock_project("ev_planner", mission, workspace=tmp_workspace, recipe=recipe)
    assert mission.target_column == "defect"
    assert mission.time_column == "batch_time"
    assert "downstream_qc_score" in mission.forbidden_columns
    assert mission.capability.target_type == "binary"
    assert mission.success_criterion.metric == "roc_auc"
    for k in ("mission", "columns", "join_plan", "hypotheses"):
        assert arts[k].exists(), k
