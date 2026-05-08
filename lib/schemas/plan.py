"""Plan dict schema — the agent's per-iteration output.

The researcher emits one of these per iteration. Step 3 of the loop
(`lib.run`) consumes it. `prior_evidence` is mandatory; without it the plan
is rejected so the agent cannot run experiments unmoored from the sketch.

The orchestrator passes `validation_context` into `PlanDict.model_validate`:

    PlanDict.model_validate(
        plan_payload,
        context={
            "recent_fingerprints": [...],   # last `window` doomed fingerprints
            "breakthrough_mode_active": True,
        },
    )

When `breakthrough_mode_active` is True, the validator requires
`prior_evidence.kind == "domain_prior"` and a URL-/arxiv-/doi-shaped
`reference`. When `recent_fingerprints` is non-empty, the validator rejects
plans whose own fingerprint matches any of them (this is how the
disk-backed doom-loop forces structural diversification).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from lib.schemas._base import VersionedModel


_DOMAIN_PRIOR_REF_RX = re.compile(r"^(https?://|arxiv:|doi:)", flags=re.IGNORECASE)

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
    technique_summary: Optional[str] = Field(
        default=None,
        max_length=600,
        description=(
            "When kind == 'domain_prior' and the source is a paper, a short "
            "implementable summary of the technique to fold into params/features. "
            "Optional outside breakthrough mode."
        ),
    )
    paper_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Publication year if the prior is a paper. Optional.",
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

    def fingerprint(self) -> str:
        """Stable fingerprint of `(model, technique_family, area, sorted(features))`.
        Used by the doom-loop check and the breakthrough-mode validator. Defined
        on the model so the schema is the single source of truth for what
        counts as a "structurally identical" plan."""
        payload = "|".join(
            [self.model, self.technique_family, self.area, ",".join(sorted(self.features))]
        ).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]

    @model_validator(mode="after")
    def _enforce_context_rules(self, info):  # type: ignore[no-untyped-def]
        """Context-aware rules. The orchestrator passes context via
        `PlanDict.model_validate(..., context={...})`. Two rules:

        1) Anti-doom: when `recent_fingerprints` is non-empty, reject plans
           whose fingerprint matches any of them.
        2) Breakthrough grounding: when `breakthrough_mode_active==True`,
           require `prior_evidence.kind == "domain_prior"` and a URL-/arxiv-
           /doi-shaped reference.

        When neither key is present in context (the common case for cold-start
        and warm iterations outside breakthrough mode), this validator is a
        no-op.
        """
        ctx = getattr(info, "context", None) or {}
        if not ctx:
            return self
        recent = ctx.get("recent_fingerprints") or []
        if recent:
            fp = self.fingerprint()
            if fp in list(recent):
                raise ValueError(
                    f"plan fingerprint {fp!r} matches a recent doomed plan; "
                    "doom-loop active — pick a structurally different plan "
                    "(different model OR technique_family OR area OR features)."
                )
        if ctx.get("breakthrough_mode_active"):
            ev = self.prior_evidence
            if ev.kind != "domain_prior":
                raise ValueError(
                    "breakthrough mode requires prior_evidence.kind == 'domain_prior' "
                    f"(got {ev.kind!r}). Ground the plan in a paper or domain prior."
                )
            if not _DOMAIN_PRIOR_REF_RX.match(ev.reference):
                raise ValueError(
                    "breakthrough mode requires prior_evidence.reference to be a URL, "
                    f"arxiv: id, or doi: id (got {ev.reference!r})."
                )
        return self


__all__ = ["PlanDict", "PriorEvidence", "TechniqueFamily", "Area"]
