"""Plan dict schema — the agent's per-iteration output.

The researcher emits one of these per iteration. Step 3 of the loop
(`lib.run`) consumes it. `prior_evidence` is mandatory; without it the plan
is rejected so the agent cannot run experiments unmoored from the sketch.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from lib.schemas._base import VersionedModel

TechniqueFamily = Literal[
    "linear",
    "tree",
    "boosted_tree",
    "neural",
    "ensemble",
    "rule_based",
    "survival",
    "anomaly",
    "forecasting_classical",
    "forecasting_neural",
]
"""The arms of the bandit. The bandit (`lib.bandit`) maintains a posterior
over each family's expected info gain."""

Area = Literal[
    "baseline",
    "features",
    "interactions",
    "regimes",
    "calibration",
    "robustness",
    "leakage_probe",
    "causal",
    "ensembling",
]
"""High-level theme of an experiment, used for diversification and
visualization in synthesis reports."""


class PriorEvidence(VersionedModel):
    """Mandatory justification for a plan dict.

    Either references a sketch query (which fact in the sketch motivates
    this experiment?) or a prior experiment (what result are we building on
    or contradicting?). This is the wall against unmoored experimentation.
    """

    kind: Literal["sketch_query", "prior_experiment", "hypothesis_seed", "domain_prior"] = (
        Field(
            ...,
            description=(
                "Source category. `sketch_query` references a sketch tool "
                "result. `prior_experiment` references a row in the experiment "
                "log. `hypothesis_seed` references one of the universal seeds "
                "or a recipe seed. `domain_prior` references a domain-module "
                "constant such as PHYSICS_RELATIONS."
            ),
        )
    )
    reference: str = Field(
        ...,
        description=(
            "Stable identifier of the source: sketch query name + arg hash, "
            "experiment.id, hypothesis_id, or domain prior key."
        ),
    )
    summary: str = Field(
        ...,
        max_length=400,
        description="One-sentence summary of what the source says, in the agent's words.",
    )


class PlanDict(VersionedModel):
    """The plan the researcher emits per iteration.

    Validated before being passed to the runner sub-agent. Required fields
    capture both *what* to run and *why*.
    """

    id: str = Field(
        ...,
        description=(
            "Unique plan id, of the form 'P-<iteration>-<short_hash>'. "
            "Stable across replay."
        ),
    )
    iteration: int = Field(
        ...,
        ge=0,
        description="Iteration number (matches the experiment_log row).",
    )
    hypothesis_id: str = Field(
        ...,
        description=(
            "Reference to a HYPOTHESES.jsonl entry (or 'H-seed-N' for the 5 "
            "universal seeds)."
        ),
    )

    model: str = Field(
        ...,
        description=(
            "Model factory key from `lib.registry` (e.g., 'logreg', "
            "'lgbm_binary', 'cox_ph')."
        ),
    )
    features: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Feature DSL list. Supports literal column names, `+all_allowed` "
            "(expanded to MISSION.allowed_columns), `+lag_downstream` for "
            "manufacturing lag joins, and `engineered:<GROUP>` for catalog "
            "entries from lib.features."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters passed to the registry factory.",
    )
    calibrate: bool = Field(
        default=False,
        description=(
            "Whether to apply post-hoc probability calibration (only "
            "meaningful for binary/probabilistic targets)."
        ),
    )

    prior_evidence: PriorEvidence = Field(
        ...,
        description="Mandatory justification (see PriorEvidence).",
    )

    technique_family: TechniqueFamily = Field(
        ...,
        description="Bandit arm key for posterior updates.",
    )
    area: Area = Field(
        ...,
        description="Theme of the experiment.",
    )
    expected_info_gain: float = Field(
        ...,
        ge=0.0,
        description=(
            "Researcher's prior on info gain (0..1). Calibrated post-hoc "
            "against actual delta in best metric."
        ),
    )

    notes: str = Field(default="", description="Optional free-form notes.")

    @field_validator("id")
    @classmethod
    def _id_form(cls, v: str) -> str:
        if not v.startswith("P-"):
            raise ValueError("plan id must start with 'P-' (e.g., 'P-7-a1b2c3')")
        return v

    @field_validator("features")
    @classmethod
    def _no_empty_token(cls, v: list[str]) -> list[str]:
        if any(not tok.strip() for tok in v):
            raise ValueError("features must not contain empty/whitespace tokens")
        return v


__all__ = ["PlanDict", "PriorEvidence", "TechniqueFamily", "Area"]
