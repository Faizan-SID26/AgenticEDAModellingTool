"""Pydantic v2 schemas for every artifact the framework produces or consumes.

Every model has a `schema_version` field whose default is `SCHEMA_VERSION`
from `lib`. Read paths validate via `Model.model_validate(...)`; write paths
emit via `model.model_dump_json(...)`.
"""
from __future__ import annotations

from lib.schemas.budget import BudgetLedgerEntry
from lib.schemas.experiment import ExperimentResult
from lib.schemas.knowledge import (
    FailureModeEntry,
    HypothesisLibraryEntry,
    KnowledgeBundle,
)
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.schemas.plan import PlanDict, PriorEvidence
from lib.schemas.project_meta import ConfidenceTier, ProjectMeta, ProjectStatus
from lib.schemas.question import Question, QuestionAnswer, QuestionBatch
from lib.schemas.recommendation import (
    CausalAssumption,
    CounterfactualEffect,
    Recommendation,
)
from lib.schemas.sketch import (
    L1ColumnSummary,
    L2JointSummary,
    L3RegimeSummary,
    L4CoresetSummary,
    L5TimeseriesSummary,
    L6CausalSummary,
    L7FailureClusterSummary,
    SketchAnnotation,
    SketchManifest,
)

__all__ = [
    "BudgetLedgerEntry",
    "CapabilityComposition",
    "CausalAssumption",
    "ConfidenceTier",
    "CounterfactualEffect",
    "ExperimentResult",
    "FailureModeEntry",
    "HypothesisLibraryEntry",
    "KnowledgeBundle",
    "L1ColumnSummary",
    "L2JointSummary",
    "L3RegimeSummary",
    "L4CoresetSummary",
    "L5TimeseriesSummary",
    "L6CausalSummary",
    "L7FailureClusterSummary",
    "Mission",
    "MissionBudget",
    "PlanDict",
    "PriorEvidence",
    "ProjectMeta",
    "ProjectStatus",
    "Question",
    "QuestionAnswer",
    "QuestionBatch",
    "Recommendation",
    "SketchAnnotation",
    "SketchManifest",
    "SuccessCriterion",
]
