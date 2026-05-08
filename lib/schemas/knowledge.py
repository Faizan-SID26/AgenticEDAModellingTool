"""Cross-project knowledge schemas.

These are the structured rows that the post-merge extractor writes into
`knowledge/`. Column names are anonymized to *semantic roles* via the
domain module (e.g., `temp_zone_a` → `<sensor:temperature>`); raw column
names never appear in `knowledge/`.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel


class HypothesisLibraryEntry(VersionedModel):
    """One pattern extracted from a successful experiment."""

    entry_id: str = Field(...)
    source_project: str = Field(
        ...,
        description="Project that produced this entry (for provenance).",
    )
    source_iteration: int = Field(..., ge=0)
    domain: str = Field(...)
    capability_signature: str = Field(
        ...,
        description="Hash of the capability composition this came from.",
    )

    pattern_summary: str = Field(
        ...,
        description="One-sentence description of the pattern in domain-agnostic terms.",
    )
    technique_family: str = Field(...)
    feature_roles: list[str] = Field(
        default_factory=list,
        description="Semantic role tags of features used (anonymized).",
    )
    sketch_signature: dict[str, float] = Field(
        default_factory=dict,
        description="Salient sketch values (e.g., n_regimes, top mutual_info) for retrieval.",
    )

    info_gain: float = Field(
        ...,
        description="Realized info gain in the source experiment.",
    )
    primary_metric: str = Field(...)
    primary_metric_delta: float = Field(...)

    # Hydration fields (Pillar 3) — let cross-project knowledge be reused as
    # actual plans, not just labels. Optional for back-compat with older
    # bundles that did not record them.
    model: Optional[str] = Field(
        default=None,
        description=(
            "Registry model key the source experiment used (e.g. 'lgbm_focal'). "
            "Used by `_cross_project_hypotheses` to materialize a plan that "
            "actually replays the source's choice of model rather than "
            "collapsing to a generic LGBM."
        ),
    )
    feature_dsl: list[str] = Field(
        default_factory=list,
        description=(
            "Feature DSL tokens the source experiment used. Empty for "
            "legacy bundles; the generator falls back to '+all_allowed' in "
            "that case."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Hyperparameters the source experiment used. Replayed verbatim by "
            "cross-project hypotheses."
        ),
    )


class FailureModeEntry(VersionedModel):
    """One failure mode that fired on a project and was resolved."""

    entry_id: str = Field(...)
    source_project: str = Field(...)
    domain: str = Field(...)
    capability_signature: str = Field(...)

    failure_name: str = Field(
        ...,
        description="Skeptic check key or named failure mode.",
    )
    failure_signature: dict[str, float] = Field(default_factory=dict)
    resolution: str = Field(
        ...,
        description="One-sentence summary of what made the failure go away.",
    )
    n_iterations_to_resolve: int = Field(default=0, ge=0)


class KnowledgeBundle(VersionedModel):
    """The full bundle produced by `/contribute` and consumed by the extractor.

    Lives in the project at `results/knowledge_bundle.json`; the extractor
    appends entries to the cross-project JSONL files post-merge.
    """

    project_name: str = Field(...)
    domain: str = Field(...)
    capability_signature: str = Field(...)
    hypothesis_entries: list[HypothesisLibraryEntry] = Field(default_factory=list)
    failure_entries: list[FailureModeEntry] = Field(default_factory=list)
    sketch_similarity_vector: list[float] = Field(default_factory=list)
    notes: Optional[str] = Field(default=None)
