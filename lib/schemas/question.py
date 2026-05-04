"""Adaptive planning question schemas.

`/plan` is a conversational adaptive Q&A. The planner emits batches of
high-confidence inferences for batch confirmation, then targeted questions
for unresolved fields. Each question records confidence, dependency
chains, and expected impact on MISSION assembly.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel

QuestionKind = Literal[
    "confirm_inference",
    "choose_one",
    "free_text",
    "multi_select",
    "numeric",
]
QuestionImpact = Literal["mission_field", "join_plan", "leakage_policy", "success_criterion", "budget"]


class Question(VersionedModel):
    """A single adaptive planning question."""

    question_id: str = Field(
        ...,
        description="Stable id of the form 'Q-<batch>-<n>'.",
    )
    kind: QuestionKind = Field(...)
    prompt: str = Field(..., description="Human-facing question text.")
    options: Optional[list[str]] = Field(
        default=None,
        description="For choose_one / multi_select.",
    )
    inferred_answer: Optional[Any] = Field(
        default=None,
        description=(
            "What the planner inferred from INIT_PROFILE+recipe+priors. "
            "Confirm-inference questions ask 'is this right?'."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Planner's confidence in `inferred_answer` (0..1).",
    )
    impact: QuestionImpact = Field(
        ...,
        description="Which part of the MISSION this question affects.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Question ids that must be answered before this one.",
    )
    target_mission_path: str = Field(
        ...,
        description=(
            "Dotted path into the MISSION schema this answer fills "
            "(e.g., 'capability.target_type', 'success_criterion.threshold'). "
            "Used by lib.planning to assemble the locked MISSION."
        ),
    )


class QuestionAnswer(VersionedModel):
    """The user's answer to one question."""

    question_id: str = Field(...)
    answer: Any = Field(...)
    confirmed_inference: bool = Field(
        default=False,
        description="True iff the user accepted the planner's `inferred_answer`.",
    )


class QuestionBatch(VersionedModel):
    """A batch of questions presented to the user at once."""

    batch_id: str = Field(...)
    iteration: int = Field(..., ge=0, description="Which planning iteration this is.")
    questions: list[Question] = Field(...)
    answers: list[QuestionAnswer] = Field(default_factory=list)
    notes: str = Field(default="")
