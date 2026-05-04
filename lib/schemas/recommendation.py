"""Final recommendation schema (results/FINAL.md is rendered from this).

Counterfactual-shaped: the recommendation states a decision/forecast with a
quantified expected effect, evidence chain, causal assumptions, and
explicit "what would change this" conditions.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel


class CounterfactualEffect(VersionedModel):
    """Estimated counterfactual impact of the recommended action."""

    metric: str = Field(..., description="Outcome metric being impacted.")
    point_estimate: float = Field(...)
    ci_low: float = Field(..., description="Lower bound of confidence interval.")
    ci_high: float = Field(..., description="Upper bound of confidence interval.")
    ci_level: float = Field(default=0.9, ge=0.0, le=1.0)
    estimator: str = Field(
        ...,
        description=(
            "Causal estimator used (e.g., 'dowhy.linear_regression', "
            "'dowhy.propensity_score', 'no_estimator_observational')."
        ),
    )


class CausalAssumption(VersionedModel):
    """One causal assumption the recommendation relies on."""

    assumption: str = Field(...)
    justification: str = Field(...)
    sensitivity_check: Optional[str] = Field(
        default=None,
        description="If a sensitivity analysis was run, what it concluded.",
    )


class FailureModeRuleOut(VersionedModel):
    """A failure mode considered and ruled out."""

    name: str = Field(...)
    why_ruled_out: str = Field(...)
    evidence_ref: str = Field(
        ...,
        description="Experiment id or sketch query result that ruled it out.",
    )


class WhatWouldChangeIt(VersionedModel):
    """A condition under which the recommendation would change."""

    condition: str = Field(...)
    expected_change: str = Field(...)


class ModelCardEntry(VersionedModel):
    """Appendix: one line about the model that produced the recommendation."""

    model: str = Field(...)
    primary_metric: str = Field(...)
    primary_metric_value: float = Field(...)
    validation_strategy: str = Field(...)
    n_train: int = Field(...)
    n_validation: int = Field(...)
    seeds: dict[str, int] = Field(default_factory=dict)


class Recommendation(VersionedModel):
    """The locked recommendation produced by `/finalize`."""

    project_name: str = Field(...)
    recommendation_type: Literal["decision", "forecast", "ranked_factors", "alert_policy"] = Field(...)

    decision: str = Field(
        ...,
        description=(
            "One-sentence recommended action. May be 'No actionable signal "
            "found — collect more data on X' if honest failure."
        ),
    )
    rationale: str = Field(..., description="Multi-sentence rationale.")

    counterfactual: Optional[CounterfactualEffect] = Field(
        default=None,
        description="Quantified expected impact (None if no actionable signal).",
    )
    evidence_chain: list[str] = Field(
        ...,
        description="Ordered list of evidence references (experiment ids, sketch query keys).",
    )
    causal_assumptions: list[CausalAssumption] = Field(default_factory=list)
    ruled_out_failure_modes: list[FailureModeRuleOut] = Field(default_factory=list)
    what_would_change_it: list[WhatWouldChangeIt] = Field(default_factory=list)

    model_card: list[ModelCardEntry] = Field(default_factory=list)

    confidence_tier: Literal["high", "medium", "low", "no_signal"] = Field(
        ...,
        description=(
            "Coarse confidence grade. 'no_signal' indicates honest failure "
            "and is shippable."
        ),
    )

    notes: str = Field(default="")
