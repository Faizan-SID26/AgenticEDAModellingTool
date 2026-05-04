"""Experiment result row schema — what `lib.run` writes to experiment_log.jsonl.

One row per iteration. Append-only. Used for replay, synthesis, finalize,
hypothesis generation, and the bandit posterior update.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel
from lib.schemas.plan import Area, TechniqueFamily

SkepticVerdict = Literal["ACCEPT", "WARN", "FAIL"]


class SkepticResult(VersionedModel):
    """Output of the deterministic skeptic on this experiment."""

    verdict: SkepticVerdict = Field(
        ...,
        description=(
            "ACCEPT: trust the result. WARN: caveats noted. FAIL: do not "
            "use this result; in strict mode this also halts iteration if "
            "repeated."
        ),
    )
    failed_checks: list[str] = Field(
        default_factory=list,
        description="Keys of checks that failed (see lib.skeptic).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Keys of checks that warned but did not fail.",
    )
    notes: str = Field(default="", description="Free-form skeptic narrative (short).")


class FitMetrics(VersionedModel):
    """Per-split metrics produced by `lib.eval` for this experiment."""

    train: dict[str, float] = Field(default_factory=dict)
    validation: dict[str, float] = Field(default_factory=dict)
    test: dict[str, float] = Field(default_factory=dict)
    holdout: dict[str, float] = Field(default_factory=dict)


class TokenUsage(VersionedModel):
    """Token cost of the experiment, including agent and runner."""

    researcher_tokens: int = Field(default=0, ge=0)
    runner_tokens: int = Field(default=0, ge=0)
    skeptic_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ExperimentResult(VersionedModel):
    """Append-only row in experiment_log.jsonl.

    The `id` field matches the plan id that produced it; replay rebuilds
    artifacts by replaying these rows in order.
    """

    id: str = Field(..., description="Plan id (P-<iter>-<hash>).")
    iteration: int = Field(..., ge=0)
    hypothesis_id: str = Field(...)

    model: str = Field(...)
    features_used: list[str] = Field(
        ...,
        description="Concrete columns after DSL expansion (post-audit).",
    )
    params: dict[str, Any] = Field(default_factory=dict)
    calibrated: bool = Field(default=False)

    technique_family: TechniqueFamily = Field(...)
    area: Area = Field(...)

    metrics: FitMetrics = Field(...)
    primary_metric: str = Field(
        ...,
        description="Key of the metric that drives best-tracking (matches MISSION.success_criterion.metric).",
    )
    primary_metric_value: float = Field(
        ...,
        description="Value of `primary_metric` on `success_criterion.on_split`.",
    )
    is_best_so_far: bool = Field(
        default=False,
        description="True iff this iteration improved the best primary_metric_value.",
    )

    skeptic: SkepticResult = Field(...)
    plot_paths: list[str] = Field(
        default_factory=list,
        description="Relative paths under results/iter_NNN/ for saved plots.",
    )
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of artifact_kind → relative path. E.g., 'fitted_model' → "
            "'results/iter_007/model.joblib'."
        ),
    )

    seeds: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "All RNG seeds used: 'numpy', 'sklearn', 'lightgbm', etc. Replay "
            "uses these verbatim."
        ),
    )
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    duration_sec: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = Field(
        default=None,
        description="If the experiment failed catastrophically, the truncated traceback.",
    )

    info_gain_actual: float = Field(
        default=0.0,
        description=(
            "Realized information gain (post-hoc): typically (delta in best "
            "primary metric)/sigma_baseline. Used to update the bandit."
        ),
    )

    notes: str = Field(default="")
