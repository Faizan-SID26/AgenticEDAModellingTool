"""Adaptive planning Q&A.

Takes INIT_PROFILE + recipe + domain priors, produces ordered question
batches. Batch 1 = high-confidence inferences for batch confirmation.
Subsequent batches target unresolved fields with `depends_on` ordering.
Termination = MISSION passes consistency checks.

The planner skill (`.claude/skills/planner/SKILL.md`) drives the loop;
this module is the deterministic plumbing the agent calls.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from lib.domains import get as get_domain
from lib.schemas.mission import (
    CapabilityComposition,
    JoinSpec,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.schemas.question import Question, QuestionAnswer, QuestionBatch
from lib.workspace import recipes_dir, resolve_workspace, project_path

_log = logging.getLogger("eda.planning")


def load_recipe(workspace: Optional[Path], recipe_key: Optional[str]) -> Optional[dict[str, Any]]:
    """Load the named recipe JSON, or None."""
    if recipe_key is None:
        return None
    ws = resolve_workspace(workspace)
    rp = recipes_dir(ws) / f"{recipe_key}.json"
    if not rp.exists():
        _log.warning("recipe %s not found at %s", recipe_key, rp)
        return None
    return json.loads(rp.read_text(encoding="utf-8"))


def load_init_profile(project_dir: Path) -> dict[str, Any]:
    """Load `memory/INIT_PROFILE.json` from a project."""
    p = Path(project_dir) / "memory" / "INIT_PROFILE.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


# --- Batch construction --------------------------------------------------


def _flatten_columns(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all columns across all files, tagged with table name."""
    out = []
    for f in profile.get("files", []):
        for c in f.get("columns", []):
            out.append({**c, "table_name": f.get("table_name")})
    return out


def _pick_target(profile: dict[str, Any], recipe: Optional[dict[str, Any]]) -> Optional[str]:
    """Highest-confidence target column inference."""
    candidates: list[str] = []
    for f in profile.get("files", []):
        candidates.extend(f.get("likely_target", []))
    return candidates[0] if candidates else None


def _pick_time(profile: dict[str, Any]) -> Optional[str]:
    candidates: list[str] = []
    for f in profile.get("files", []):
        candidates.extend(f.get("likely_time", []))
    return candidates[0] if candidates else None


def _pick_group(profile: dict[str, Any]) -> Optional[str]:
    candidates: list[str] = []
    for f in profile.get("files", []):
        candidates.extend(f.get("likely_id", []))
    return candidates[0] if candidates else None


def _forbidden_from_domain(domain_key: str, all_columns: list[str]) -> list[str]:
    """Apply the domain's default_forbidden patterns to surface columns."""
    spec = get_domain(domain_key)
    out = []
    for col in all_columns:
        cl = col.lower()
        for pat in spec.default_forbidden:
            if pat.lower() in cl:
                out.append(col)
                break
    return out


def build_initial_batch(
    profile: dict[str, Any],
    recipe: Optional[dict[str, Any]],
    domain_key: str,
) -> QuestionBatch:
    """Build batch B-1: high-confidence inferences for batch confirmation."""
    all_cols = [c["name"] for c in _flatten_columns(profile)]
    inferred_target = _pick_target(profile, recipe)
    inferred_time = _pick_time(profile)
    inferred_group = _pick_group(profile)
    inferred_forbidden = _forbidden_from_domain(domain_key, all_cols)

    qs: list[Question] = []
    n = 1
    if inferred_target:
        qs.append(
            Question(
                question_id=f"Q-1-{n}",
                kind="confirm_inference",
                prompt=f"Is `{inferred_target}` the target column?",
                inferred_answer=inferred_target,
                confidence=0.85,
                impact="mission_field",
                target_mission_path="target_column",
            )
        )
        n += 1
    else:
        qs.append(
            Question(
                question_id=f"Q-1-{n}",
                kind="free_text",
                prompt="Which column is the target / outcome?",
                confidence=0.0,
                impact="mission_field",
                target_mission_path="target_column",
            )
        )
        n += 1

    if recipe and recipe.get("capability", {}).get("temporal_structure", "none") != "none":
        if inferred_time:
            qs.append(
                Question(
                    question_id=f"Q-1-{n}",
                    kind="confirm_inference",
                    prompt=f"Is `{inferred_time}` the time-ordering column?",
                    inferred_answer=inferred_time,
                    confidence=0.85,
                    impact="mission_field",
                    target_mission_path="time_column",
                )
            )
            n += 1

    if recipe and recipe.get("capability", {}).get("validation_strategy") == "group_kfold":
        if inferred_group:
            qs.append(
                Question(
                    question_id=f"Q-1-{n}",
                    kind="confirm_inference",
                    prompt=f"Is `{inferred_group}` the entity / group column?",
                    inferred_answer=inferred_group,
                    confidence=0.7,
                    impact="mission_field",
                    target_mission_path="group_column",
                )
            )
            n += 1

    if inferred_forbidden:
        qs.append(
            Question(
                question_id=f"Q-1-{n}",
                kind="confirm_inference",
                prompt=(
                    "Treat these columns as forbidden (downstream of target / leakage risk)?"
                ),
                inferred_answer=inferred_forbidden,
                confidence=0.7,
                impact="leakage_policy",
                target_mission_path="forbidden_columns",
            )
        )
        n += 1

    # Always-asked: success_criterion threshold (recipe-default suggested).
    if recipe and recipe.get("default_success_criterion"):
        sc = recipe["default_success_criterion"]
        qs.append(
            Question(
                question_id=f"Q-1-{n}",
                kind="confirm_inference",
                prompt=(
                    f"Use the recipe-default success criterion: {sc['metric']} {sc['direction']} {sc['threshold']}?"
                ),
                inferred_answer=sc,
                confidence=0.6,
                impact="success_criterion",
                target_mission_path="success_criterion",
            )
        )
        n += 1

    return QuestionBatch(
        batch_id="B-1",
        iteration=0,
        questions=qs,
        notes="Initial batch — high-confidence inferences for batch confirmation.",
    )


def build_followup_batch(
    profile: dict[str, Any],
    recipe: Optional[dict[str, Any]],
    answers: dict[str, Any],
    iteration: int,
) -> QuestionBatch:
    """Build a follow-up batch for fields that remain unresolved.

    `answers` is a dict mapping `target_mission_path` → user answer (after
    confirmation logic). The follow-up questions probe gaps such as a
    forecast horizon, the business question, and join confirmation.
    """
    qs: list[Question] = []
    n = 1
    if "business_question" not in answers:
        qs.append(
            Question(
                question_id=f"Q-{iteration + 1}-{n}",
                kind="free_text",
                prompt="In one sentence: what is the business question this project must answer?",
                confidence=0.0,
                impact="mission_field",
                target_mission_path="business_question",
            )
        )
        n += 1
    if profile.get("proposed_joins") and "join_plan" not in answers:
        qs.append(
            Question(
                question_id=f"Q-{iteration + 1}-{n}",
                kind="confirm_inference",
                prompt="Use the proposed joins?",
                inferred_answer=profile["proposed_joins"],
                confidence=0.65,
                impact="join_plan",
                target_mission_path="join_plan",
            )
        )
        n += 1
    return QuestionBatch(batch_id=f"B-{iteration + 1}", iteration=iteration, questions=qs)


# --- Assembly ------------------------------------------------------------


def _resolve_answer(q: Question, ans: QuestionAnswer) -> Any:
    """Turn a (Question, QuestionAnswer) into a final value."""
    if q.kind == "confirm_inference":
        if ans.confirmed_inference and q.inferred_answer is not None:
            return q.inferred_answer
        return ans.answer
    return ans.answer


def collect_resolved_answers(batches: list[QuestionBatch]) -> dict[str, Any]:
    """Walk batches, returning a {target_mission_path: value} dict."""
    out: dict[str, Any] = {}
    for batch in batches:
        ans_map = {a.question_id: a for a in batch.answers}
        for q in batch.questions:
            if q.question_id not in ans_map:
                continue
            value = _resolve_answer(q, ans_map[q.question_id])
            if value is not None:
                out[q.target_mission_path] = value
    return out


def assemble_mission(
    project_name: str,
    domain_key: str,
    recipe: Optional[dict[str, Any]],
    answers: dict[str, Any],
    token_budget: int,
    iteration_budget: int = 100,
    notes: str = "",
) -> Mission:
    """Build a `Mission` from collected planning answers + recipe defaults.

    Validation happens in the Mission constructor; failures bubble up as
    `pydantic.ValidationError`.
    """
    cap_dict = (recipe or {}).get("capability") or {}
    capability = CapabilityComposition(
        temporal_structure=answers.get("capability.temporal_structure", cap_dict.get("temporal_structure", "none")),
        leakage_model=answers.get("capability.leakage_model", cap_dict.get("leakage_model", "none")),
        target_type=answers.get("capability.target_type", cap_dict.get("target_type")),
        validation_strategy=answers.get(
            "capability.validation_strategy", cap_dict.get("validation_strategy", "stratified")
        ),
        recommendation_type=answers.get(
            "capability.recommendation_type", cap_dict.get("recommendation_type", "decision")
        ),
    )

    sc_in = answers.get("success_criterion") or (recipe or {}).get("default_success_criterion") or {
        "metric": "roc_auc",
        "threshold": 0.7,
        "direction": ">=",
        "on_split": "validation",
    }
    sc = SuccessCriterion(**sc_in)
    budget = MissionBudget(token_cap=token_budget, iteration_cap=iteration_budget)

    join_plan_input = answers.get("join_plan", []) or []
    join_plan: list[JoinSpec] = []
    for jp in join_plan_input:
        # Some tests / users may pass already-validated JoinSpec dicts.
        if isinstance(jp, JoinSpec):
            join_plan.append(jp)
            continue
        join_plan.append(
            JoinSpec(
                left_table=jp.get("left_table"),
                right_table=jp.get("right_table"),
                on=list(jp.get("on", [])),
                how=jp.get("how", "inner"),
                lag_policy=jp.get("lag_policy"),
            )
        )

    return Mission(
        project_name=project_name,
        domain=domain_key,
        recipe=(recipe or {}).get("recipe_key"),
        capability=capability,
        target_column=answers.get("target_column", ""),
        time_column=answers.get("time_column"),
        group_column=answers.get("group_column"),
        forbidden_columns=list(answers.get("forbidden_columns", [])),
        allowed_columns=list(answers.get("allowed_columns", [])),
        join_plan=join_plan,
        success_criterion=sc,
        budget=budget,
        business_question=answers.get("business_question", ""),
        notes=notes,
    )


def write_mission(project_dir: Path, mission: Mission) -> Path:
    """Persist the locked MISSION.json."""
    p = Path(project_dir) / "MISSION.json"
    p.write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    return p
