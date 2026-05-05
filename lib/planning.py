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


def domain_docs_path(project_dir: Path) -> Path:
    """Path to memory/DOMAIN_DOCS.md (may not exist if no docs were dropped)."""
    return Path(project_dir) / "memory" / "DOMAIN_DOCS.md"


def has_domain_docs(project_dir: Path) -> bool:
    """True iff /init wrote a DOMAIN_DOCS.md (i.e., user dropped a PUD/spec/SOP)."""
    return domain_docs_path(project_dir).exists()


def _q(qid: str, **kwargs) -> Question:
    return Question(question_id=qid, **kwargs)


def _process_knowledge_batch(
    profile: dict[str, Any],
    recipe: Optional[dict[str, Any]],
    domain_key: str,
    answers: dict[str, Any],
    iteration: int,
) -> list[Question]:
    """Batch 2 — process knowledge.

    For industrial / process-shaped domains, knowing how the process
    runs is often the difference between a useful model and a useless
    one. Ask deliberately, even when it feels like a lot — the planner
    only happens once.
    """
    qid = lambda n: f"Q-{iteration + 1}-{n}"
    qs: list[Question] = []
    n = 1

    has_temporal = (
        (recipe and recipe.get("capability", {}).get("temporal_structure", "none") != "none")
        or "time_column" in answers
    )

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "In 1-3 sentences: what is the **process** behind this data? "
                "(e.g. 'raw material → mixing → reactor → curing → final QA, "
                "with 4-hour residence in the reactor and a daily campaign change'). "
                "Be specific about stage order and timing."
            ),
            confidence=0.0,
            impact="mission_field",
            target_mission_path="process_description",
        )
    )
    n += 1

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "Which 2-5 process variables do **you** expect to drive the target most strongly, "
                "and why? (Used to seed hypotheses; the model is free to disagree.)"
            ),
            confidence=0.0,
            impact="mission_field",
            target_mission_path="expected_drivers",
        )
    )
    n += 1

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "What known **leakage pitfalls** exist beyond the obvious downstream-QC columns? "
                "E.g. inspection columns recorded after the fact, technician notes that "
                "summarize the outcome, audit fields populated post-event. List any you can think of."
            ),
            confidence=0.0,
            impact="leakage_policy",
            target_mission_path="extra_forbidden_columns",
        )
    )
    n += 1

    if has_temporal:
        qs.append(
            _q(
                qid(n),
                kind="free_text",
                prompt=(
                    "How is the time column **interpreted**? "
                    "(Timezone? Sampling rate? Are gaps real downtime or instrument outages? "
                    "Are timestamps event-time or recording-time?)"
                ),
                confidence=0.0,
                impact="mission_field",
                target_mission_path="time_interpretation",
            )
        )
        n += 1

        qs.append(
            _q(
                qid(n),
                kind="free_text",
                prompt=(
                    "Do you expect **operating regimes / campaigns / modes** in this data? "
                    "(e.g. summer vs winter; product A vs product B; pre vs post a known intervention). "
                    "If yes, name them and roughly when each occurred."
                ),
                confidence=0.0,
                impact="mission_field",
                target_mission_path="expected_regimes",
            )
        )
        n += 1

        if domain_key == "manufacturing":
            qs.append(
                _q(
                    qid(n),
                    kind="free_text",
                    prompt=(
                        "What's the typical **lag** between an upstream process change and "
                        "its effect at the downstream measurement point? (Used to set the "
                        "asof-join policy. Default is 'use_immediate_prior'.)"
                    ),
                    confidence=0.0,
                    impact="join_plan",
                    target_mission_path="lag_policy_note",
                )
            )
            n += 1

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "What **previous attempts** have been made on this problem and what did NOT work? "
                "(Even if informal. Helps the framework avoid known dead-ends.)"
            ),
            confidence=0.0,
            impact="mission_field",
            target_mission_path="prior_attempts_note",
        )
    )
    n += 1

    return qs


def _project_context_batch(
    profile: dict[str, Any],
    recipe: Optional[dict[str, Any]],
    domain_key: str,
    answers: dict[str, Any],
    iteration: int,
) -> list[Question]:
    """Batch 3 — project context, constraints, joins, business question."""
    qid = lambda n: f"Q-{iteration + 1}-{n}"
    qs: list[Question] = []
    n = 1

    if "business_question" not in answers:
        qs.append(
            _q(
                qid(n),
                kind="free_text",
                prompt=(
                    "In one sentence: what is the **business question** this project must answer? "
                    "(Used by the analyst at finalize. Write what you'd tell a stakeholder.)"
                ),
                confidence=0.0,
                impact="mission_field",
                target_mission_path="business_question",
            )
        )
        n += 1

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "What does **success in deployment** look like? "
                "(e.g., 'a binary alert reviewed daily by an operator', "
                "'a ranked list of 5 candidate causes per incident', "
                "'a 7-day forecast feeding a procurement script'.)"
            ),
            confidence=0.0,
            impact="mission_field",
            target_mission_path="deployment_shape",
        )
    )
    n += 1

    cap = (recipe or {}).get("capability") or {}
    if cap.get("target_type") == "binary":
        qs.append(
            _q(
                qid(n),
                kind="free_text",
                prompt=(
                    "How should the model trade **false positives vs false negatives**? "
                    "(e.g., 'one missed defect costs ~10 false alarms in our line'. "
                    "Used to bias the threshold + skeptic checks.)"
                ),
                confidence=0.0,
                impact="success_criterion",
                target_mission_path="fp_fn_tradeoff",
            )
        )
        n += 1

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "Are there **interpretability or latency** constraints on the deployed model? "
                "(e.g., 'must be a linear model', 'must score in <50ms', 'must be reviewable by safety'). "
                "Defaults to 'none' if you skip."
            ),
            confidence=0.0,
            impact="mission_field",
            target_mission_path="model_constraints",
        )
    )
    n += 1

    qs.append(
        _q(
            qid(n),
            kind="free_text",
            prompt=(
                "Anything **else** the planner should add to `forbidden_columns`? "
                "(Free-text list of column names. Leave blank if nothing.)"
            ),
            confidence=0.0,
            impact="leakage_policy",
            target_mission_path="extra_forbidden_columns_explicit",
        )
    )
    n += 1

    if profile.get("proposed_joins") and "join_plan" not in answers:
        qs.append(
            _q(
                qid(n),
                kind="confirm_inference",
                prompt="Use the proposed joins?",
                inferred_answer=profile["proposed_joins"],
                confidence=0.65,
                impact="join_plan",
                target_mission_path="join_plan",
            )
        )
        n += 1

    return qs


def _domain_doc_cross_check_batch(
    project_dir: Path,
    answers: dict[str, Any],
    iteration: int,
) -> list[Question]:
    """Batch 4 — when a PUD/spec/SOP is present, cross-check the planner's
    interpretation. Emits a single targeted question that nudges the user to
    confirm or correct the planner's reading of the document. The actual
    document content is in `memory/DOMAIN_DOCS.md`; the planner skill reads
    it before issuing the batch and substitutes the bullets it extracted.
    """
    qid = lambda n: f"Q-{iteration + 1}-{n}"
    qs: list[Question] = []
    qs.append(
        _q(
            qid(1),
            kind="free_text",
            prompt=(
                "I read `memory/DOMAIN_DOCS.md`. Before I lock the MISSION, please correct "
                "anything I got wrong from the document. Specifically: "
                "(a) named process stages, "
                "(b) hard physical constraints / sensor ranges, "
                "(c) known sensor failure modes, "
                "(d) anything I should add to forbidden_columns. "
                "Free-text, multi-line is fine; respond with 'looks right' if I got it all."
            ),
            confidence=0.0,
            impact="mission_field",
            target_mission_path="domain_doc_corrections",
        )
    )
    return qs


def build_followup_batch(
    profile: dict[str, Any],
    recipe: Optional[dict[str, Any]],
    answers: dict[str, Any],
    iteration: int,
    *,
    project_dir: Optional[Path] = None,
    domain_key: Optional[str] = None,
) -> QuestionBatch:
    """Build a follow-up batch for fields that remain unresolved.

    The planner walks through three to four themed batches:

        iteration=1  → process_knowledge
        iteration=2  → project_context
        iteration=3  → domain_doc_cross_check (only if DOMAIN_DOCS.md exists)
        iteration>=4 → empty (terminates the planning loop)

    Earlier versions of this function emitted a single 2-question batch;
    the expansion is intentional — industrial projects benefit from
    deeper upfront questioning, and the answers feed both MISSION
    metadata and the researcher's exploration brief.
    """
    qid_prefix = f"B-{iteration + 1}"
    qs: list[Question] = []

    if iteration == 1:
        qs = _process_knowledge_batch(
            profile, recipe, domain_key or "general", answers, iteration
        )
    elif iteration == 2:
        qs = _project_context_batch(
            profile, recipe, domain_key or "general", answers, iteration
        )
    elif iteration == 3 and project_dir and has_domain_docs(project_dir):
        qs = _domain_doc_cross_check_batch(project_dir, answers, iteration)
    # iteration >= 4 (or 3 without docs) — empty, signalling "done planning".

    return QuestionBatch(batch_id=qid_prefix, iteration=iteration, questions=qs)


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

    # Aggregate every free-text answer from the deeper question batches
    # into MISSION.notes so the researcher sees process knowledge,
    # expected drivers, prior attempts, deployment shape, etc. on every
    # iteration brief.
    note_keys = (
        ("process_description", "Process description"),
        ("expected_drivers", "Expected drivers"),
        ("time_interpretation", "Time interpretation"),
        ("expected_regimes", "Expected regimes"),
        ("lag_policy_note", "Lag policy note"),
        ("prior_attempts_note", "Prior attempts (what didn't work)"),
        ("deployment_shape", "Deployment shape"),
        ("fp_fn_tradeoff", "FP / FN tradeoff"),
        ("model_constraints", "Model constraints"),
        ("domain_doc_corrections", "Domain-doc corrections"),
    )
    note_lines: list[str] = []
    if notes:
        note_lines.append(notes.strip())
    for key, label in note_keys:
        v = answers.get(key)
        if v and isinstance(v, str) and v.strip() and v.strip().lower() not in ("looks right", "n/a", "none"):
            note_lines.append(f"{label}:\n{v.strip()}")
    aggregated_notes = "\n\n".join(note_lines)

    # Merge any extra forbidden columns the user named (free-text answers
    # are comma- or newline-separated).
    forbidden = list(answers.get("forbidden_columns", []))
    for key in ("extra_forbidden_columns", "extra_forbidden_columns_explicit"):
        v = answers.get(key)
        if isinstance(v, str) and v.strip():
            for tok in v.replace(",", "\n").splitlines():
                tok = tok.strip().strip("`'\"")
                if tok and tok.lower() not in ("none", "n/a") and tok not in forbidden:
                    forbidden.append(tok)

    return Mission(
        project_name=project_name,
        domain=domain_key,
        recipe=(recipe or {}).get("recipe_key"),
        capability=capability,
        target_column=answers.get("target_column", ""),
        time_column=answers.get("time_column"),
        group_column=answers.get("group_column"),
        forbidden_columns=forbidden,
        allowed_columns=list(answers.get("allowed_columns", [])),
        join_plan=join_plan,
        success_criterion=sc,
        budget=budget,
        business_question=answers.get("business_question", ""),
        notes=aggregated_notes,
    )


def write_mission(project_dir: Path, mission: Mission) -> Path:
    """Persist the locked MISSION.json."""
    p = Path(project_dir) / "MISSION.json"
    p.write_text(mission.model_dump_json(indent=2), encoding="utf-8")
    return p
