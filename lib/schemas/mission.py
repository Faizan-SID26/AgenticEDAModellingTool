"""MISSION.json schema — the agreement that scopes a project.

A MISSION is locked at the end of `/plan` and never modified during `/run`.
It declares a *capability composition* rather than a single problem-type
enum: modules dispatch on individual capability fields, which is what makes
the framework extensible to new problem shapes without explosion of
special-case branches.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from lib.schemas._base import VersionedModel

# --- Capability enums ----------------------------------------------------

TemporalStructure = Literal["regime_based", "seasonal", "none"]
LeakageModel = Literal["stage_frontier", "forecast_horizon", "none"]
TargetType = Literal[
    "binary",
    "continuous",
    "time_to_event",
    "multi_horizon",
    "rank",
    "outlier_score",
]
ValidationStrategy = Literal["time_split", "rolling_origin", "group_kfold", "stratified"]
RecommendationType = Literal["decision", "forecast", "ranked_factors", "alert_policy"]


class CapabilityComposition(VersionedModel):
    """The 5-tuple capability composition a MISSION declares.

    Every `lib.capabilities.*` and `lib.skeptic` check dispatches on these
    fields rather than on a single problem-type label.
    """

    temporal_structure: TemporalStructure = Field(
        ...,
        description=(
            "Whether the data has temporal structure and of what kind. "
            "`regime_based` indicates change-point regimes (e.g., process "
            "campaigns). `seasonal` indicates calendar/seasonal cycles. "
            "`none` indicates IID assumption is acceptable."
        ),
    )
    leakage_model: LeakageModel = Field(
        ...,
        description=(
            "How leakage is constrained. `stage_frontier` means features "
            "must come from upstream stages of a process flow (manufacturing "
            "default). `forecast_horizon` means features at time t can only "
            "use information available before t-h. `none` disables leakage "
            "checks (only for synthetic / non-temporal data)."
        ),
    )
    target_type: TargetType = Field(
        ...,
        description="Statistical type of the prediction target.",
    )
    validation_strategy: ValidationStrategy = Field(
        ...,
        description="How to split data for unbiased generalization estimates.",
    )
    recommendation_type: RecommendationType = Field(
        ...,
        description="Shape of the final recommendation produced by `/finalize`.",
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> "CapabilityComposition":
        """Reject internally-inconsistent compositions early."""
        # time_to_event needs temporal structure to make sense.
        if self.target_type == "time_to_event" and self.temporal_structure == "none":
            raise ValueError(
                "target_type=time_to_event requires temporal_structure != 'none'"
            )
        # forecast_horizon leakage only meaningful with temporal data.
        if self.leakage_model == "forecast_horizon" and self.temporal_structure == "none":
            raise ValueError(
                "leakage_model=forecast_horizon requires temporal_structure != 'none'"
            )
        # multi_horizon target requires forecast_horizon leakage model.
        if self.target_type == "multi_horizon" and self.leakage_model != "forecast_horizon":
            raise ValueError(
                "target_type=multi_horizon requires leakage_model=forecast_horizon"
            )
        # rolling_origin / time_split require temporal structure.
        if self.validation_strategy in ("rolling_origin", "time_split"):
            if self.temporal_structure == "none":
                raise ValueError(
                    f"validation_strategy={self.validation_strategy} requires "
                    "temporal_structure != 'none'"
                )
        return self


# --- Sub-models ---------------------------------------------------------


class SuccessCriterion(VersionedModel):
    """Concrete, evaluable goal for the project.

    Evaluated by `lib.state.termination_check` on every iteration.
    """

    metric: str = Field(
        ...,
        description=(
            "Metric name as keyed in the capability's metric dispatcher "
            "(e.g., 'roc_auc', 'rmse', 'concordance_index')."
        ),
    )
    threshold: float = Field(
        ...,
        description="Numeric threshold the metric must satisfy.",
    )
    direction: Literal[">=", "<="] = Field(
        ...,
        description="Whether higher (>=) or lower (<=) is success.",
    )
    on_split: Literal["validation", "test", "holdout"] = Field(
        default="validation",
        description="Which split the criterion is evaluated on.",
    )


class MissionBudget(VersionedModel):
    """Token / iteration budget for `/run`."""

    token_cap: int = Field(
        ...,
        gt=0,
        description="Maximum total tokens (input+output) before /run halts.",
    )
    iteration_cap: int = Field(
        default=100,
        gt=0,
        description="Maximum iterations before /run halts.",
    )
    stagnation_window: int = Field(
        default=12,
        gt=0,
        description=(
            "If no improvement in best metric for this many iterations, "
            "/run halts (unless the best metric is below `operational_floor`, "
            "in which case the framework enters breakthrough mode instead)."
        ),
    )
    catastrophic_failure_window: int = Field(
        default=3,
        gt=0,
        description="Same severe skeptic failure repeating this many times → halt.",
    )
    operational_floor: Optional[float] = Field(
        default=None,
        description=(
            "Absolute primary-metric floor below which stagnation must NOT "
            "halt /run. Direction-aware: with direction='>=' the floor is a "
            "minimum the metric must exceed; with '<=' it is a maximum it "
            "must remain under. When set and the best primary metric is on "
            "the wrong side, the framework enters breakthrough mode (heavier "
            "registry, paper grounding, structural diversification) instead "
            "of finalizing on a weak result. Default None preserves legacy "
            "behavior (`success_criterion.threshold` is the only gate)."
        ),
    )
    breakthrough_stagnation_window: int = Field(
        default=20,
        gt=0,
        description=(
            "Secondary stagnation window applied only after breakthrough mode "
            "has been entered. Halts /run only after this many additional "
            "iterations without improvement; gives the framework a "
            "guaranteed long second window to find an escape."
        ),
    )
    breakthrough_max_entries: int = Field(
        default=3,
        ge=1,
        description=(
            "Cap on how many times breakthrough mode can be entered per /run "
            "before finalize is allowed to write FINAL.md regardless of "
            "operational_floor compliance. Bounds runaway re-entry loops."
        ),
    )


class JoinSpec(VersionedModel):
    """Single join in the join plan, as proposed by /init and confirmed by /plan."""

    left_table: str = Field(..., description="Logical name of the left table.")
    right_table: str = Field(..., description="Logical name of the right table.")
    on: list[str] = Field(
        ...,
        min_length=1,
        description="Columns to join on (must exist in both tables).",
    )
    how: Literal["inner", "left", "right", "outer", "asof"] = Field(
        ...,
        description="Join kind. `asof` is for time-aligned process data.",
    )
    lag_policy: Optional[str] = Field(
        default=None,
        description=(
            "For asof joins: lag policy from the domain module "
            "(e.g., 'use_immediate_prior'). None for non-asof joins."
        ),
    )


# --- The Mission itself --------------------------------------------------


class Mission(VersionedModel):
    """The locked project specification produced by `/plan`.

    After lock, `lib.state` consumes this on every iteration and modules
    dispatch on its `capability` composition. It is read-only during
    `/run`.
    """

    project_name: str = Field(..., description="Human-friendly project slug.")
    domain: str = Field(
        ...,
        description=(
            "Domain module key (must be present in lib.domains.__init__'s "
            "registry, e.g., 'manufacturing', 'forecasting_demand', 'general')."
        ),
    )
    recipe: Optional[str] = Field(
        default=None,
        description="Recipe key from `recipes/` if /plan started from one.",
    )
    capability: CapabilityComposition = Field(
        ...,
        description="The capability-composition 5-tuple.",
    )

    target_column: str = Field(
        ...,
        description="Name of the target column in the joined dataset.",
    )
    time_column: Optional[str] = Field(
        default=None,
        description=(
            "Time column for ordering. Required if temporal_structure != 'none'."
        ),
    )
    group_column: Optional[str] = Field(
        default=None,
        description=(
            "Grouping column for group_kfold, or for entity-level "
            "predictive maintenance."
        ),
    )

    forbidden_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns the audit gate will reject if a plan dict tries to use "
            "them. Includes future-information columns and post-hoc labels."
        ),
    )
    allowed_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Whitelist used by the +all_allowed feature DSL. Empty means "
            "'use everything not in forbidden_columns'."
        ),
    )

    join_plan: list[JoinSpec] = Field(
        default_factory=list,
        description="Ordered list of joins produced by /init and confirmed by /plan.",
    )

    success_criterion: SuccessCriterion = Field(
        ...,
        description="Concrete evaluable goal.",
    )
    budget: MissionBudget = Field(
        ...,
        description="Token / iteration budget for /run.",
    )

    business_question: str = Field(
        ...,
        description=(
            "One-sentence, plain-English statement of what the user is trying "
            "to learn. Used by the analyst at /finalize."
        ),
    )
    notes: str = Field(
        default="",
        description="Free-form notes captured during /plan.",
    )

    @field_validator("project_name")
    @classmethod
    def _slug_ok(cls, v: str) -> str:
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("project_name must be alphanumeric (with _ or -)")
        return v

    @model_validator(mode="after")
    def _temporal_requires_time(self) -> "Mission":
        if self.capability.temporal_structure != "none" and not self.time_column:
            raise ValueError(
                "time_column is required when capability.temporal_structure != 'none'"
            )
        if self.capability.validation_strategy == "group_kfold" and not self.group_column:
            raise ValueError(
                "group_column is required when validation_strategy=group_kfold"
            )
        # forbidden ∩ allowed must be empty.
        if self.allowed_columns:
            overlap = set(self.allowed_columns) & set(self.forbidden_columns)
            if overlap:
                raise ValueError(
                    f"columns appear in both allowed_columns and forbidden_columns: {sorted(overlap)}"
                )
        if self.target_column in self.forbidden_columns:
            raise ValueError("target_column cannot be in forbidden_columns")
        return self
