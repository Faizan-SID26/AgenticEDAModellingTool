"""Schemas for the Process Data Sketch.

The sketch has 7 *structural* layers (L1..L7), each independently versioned
and updated deterministically by `lib.sketch.updaters`. A separate
*annotations* layer is LLM-written at vision checkpoints (Reviewer role)
and is **never** consulted by the structural updaters.

The actual binary representations of L1-L7 live in per-layer parquet/numpy
files under `<project>/sketch/`. The schemas below describe each layer's
*summary* (what the agent gets when it queries) and the manifest that ties
everything together.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from lib.schemas._base import VersionedModel


# --- L1: per-column distribution summary --------------------------------


class L1ColumnSummary(VersionedModel):
    """Distribution summary for a single column.

    Backed by a t-digest (quantiles), count-min sketch (frequencies), and a
    HyperLogLog (cardinality estimate).
    """

    column: str = Field(...)
    dtype: Literal["numeric", "categorical", "boolean", "datetime", "text"] = Field(...)
    n_total: int = Field(..., ge=0, description="Rows considered (after coreset).")
    n_missing: int = Field(..., ge=0)
    n_unique_estimate: int = Field(..., ge=0, description="HLL cardinality estimate.")

    quantiles: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "t-digest quantile dump; keys are string-formatted percentiles "
            "such as '0.01', '0.05', '0.25', '0.5', '0.75', '0.95', '0.99'. "
            "Numeric only."
        ),
    )
    top_categories: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top-K category, count pairs (categorical only).",
    )

    mean: Optional[float] = Field(default=None, description="Numeric mean if applicable.")
    stdev: Optional[float] = Field(default=None, description="Numeric stdev if applicable.")


# --- L2: joint structure -------------------------------------------------


class L2JointSummary(VersionedModel):
    """Low-rank joint structure + sparse interaction residual."""

    n_components: int = Field(..., ge=1, description="PCA components retained.")
    explained_variance_ratio: list[float] = Field(
        default_factory=list,
        description="Per-component explained variance fraction.",
    )
    component_loadings_top: dict[str, list[tuple[str, float]]] = Field(
        default_factory=dict,
        description=(
            "Per-component, the top-N column,loading pairs. Stored sparsely."
        ),
    )
    top_interactions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Top-K pairwise interactions detected by mutual info on the "
            "coreset. Each dict has keys: col_a, col_b, mutual_info, "
            "rank, interaction_strength_residual."
        ),
    )


# --- L3: regimes ---------------------------------------------------------


class L3RegimeSummary(VersionedModel):
    """Regime / change-point structure (PELT-detected)."""

    n_regimes: int = Field(..., ge=1)
    boundary_indices: list[int] = Field(
        default_factory=list,
        description="Row indices (in time-sorted order) where regimes start.",
    )
    regime_means: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Per regime, per primary column (target + top features), mean.",
    )
    regime_sizes: list[int] = Field(default_factory=list)
    regime_target_distribution: list[dict[str, float]] = Field(
        default_factory=list,
        description="Per regime, summary of target distribution (rate, mean, etc.).",
    )


# --- L4: coresets --------------------------------------------------------


class L4CoresetSummary(VersionedModel):
    """Per-capability importance-weighted samples."""

    capability_key: str = Field(
        ...,
        description="Capability the coreset is tuned for (e.g., 'tabular_classification').",
    )
    n_rows: int = Field(..., ge=1)
    weight_l2_norm: float = Field(default=0.0, description="||weights||_2 — sanity check.")
    path: str = Field(
        ...,
        description="Project-relative path to the coreset parquet.",
    )


# --- L5: timeseries ------------------------------------------------------


class L5TimeseriesSummary(VersionedModel):
    """Per time-series column SAX + matrix-profile summary."""

    column: str = Field(...)
    sax_alphabet_size: int = Field(default=8, ge=2)
    sax_word_length: int = Field(default=8, ge=2)
    sax_top_motifs: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Top SAX words by occurrence, with counts.",
    )
    matrix_profile_window: int = Field(default=64, ge=4)
    matrix_profile_top_motifs: list[dict[str, Any]] = Field(default_factory=list)
    matrix_profile_top_discords: list[dict[str, Any]] = Field(default_factory=list)


# --- L6: causal hints ----------------------------------------------------


class L6CausalSummary(VersionedModel):
    """PC-algorithm-derived DAG fragment hints."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Each edge: {src, dst, kind: 'directed'|'undirected', "
            "ci_test_pval, weight}."
        ),
    )
    test: Literal["partial_correlation", "kci"] = Field(default="partial_correlation")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)


# --- L7: failure modes ---------------------------------------------------


class L7FailureClusterSummary(VersionedModel):
    """One online-maintained failure cluster (Mahalanobis matched)."""

    cluster_id: str = Field(...)
    n_observations: int = Field(..., ge=1)
    centroid: dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature mean of cluster members (Welford).",
    )
    inv_cov_diag: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Diagonal of inverse covariance — used for Mahalanobis-distance "
            "matching of new residuals."
        ),
    )
    mean_residual: float = Field(default=0.0)
    last_updated_iteration: int = Field(default=0, ge=0)
    label: Optional[str] = Field(
        default=None,
        description="LLM-assigned label (set via annotations layer, never used by updaters).",
    )


# --- Annotations (LLM-written, separate from structural layers) ---------


class SketchAnnotation(VersionedModel):
    """An LLM-written annotation attached to some sketch element.

    Stored under `<project>/sketch/annotations/<kind>.jsonl`. The structural
    updaters never read these files.
    """

    kind: Literal[
        "regime_label",
        "failure_cluster_label",
        "interaction_explanation",
        "motif_label",
        "general_observation",
    ] = Field(...)
    target_id: str = Field(
        ...,
        description="Element id (regime index, cluster id, motif key, etc.).",
    )
    iteration: int = Field(..., ge=0, description="Iteration at which this was written.")
    text: str = Field(..., max_length=2000)
    author_role: Literal["reviewer", "researcher", "analyst"] = Field(default="reviewer")


# --- Manifest ------------------------------------------------------------


class SketchManifest(VersionedModel):
    """Top-level manifest tying together all layers and their files.

    Committed to git. The binary L1..L7 files are gitignored but their
    paths and shapes are recorded here so similarity queries can compare
    sketches without loading the binaries.
    """

    project_name: str = Field(...)
    seed: int = Field(..., description="Build-time RNG seed (deterministic replay).")
    n_rows_source: int = Field(..., ge=0, description="Rows in the source joined parquet.")
    n_columns_source: int = Field(..., ge=0)
    capabilities: list[str] = Field(
        default_factory=list,
        description="Capability keys the sketch was built for (controls L4 coresets).",
    )

    l1_path: str = Field(..., description="Path to L1 layer summary file.")
    l2_path: str = Field(..., description="Path to L2 layer summary file.")
    l3_path: str = Field(..., description="Path to L3 layer summary file.")
    l4_paths: list[str] = Field(default_factory=list, description="Per-capability coreset paths.")
    l5_path: str = Field(..., description="Path to L5 layer summary file.")
    l6_path: str = Field(..., description="Path to L6 layer summary file.")
    l7_path: str = Field(..., description="Path to L7 layer summary file.")

    annotations_dir: str = Field(default="sketch/annotations")
    total_size_bytes: int = Field(
        ...,
        ge=0,
        description="Sum of L1..L7 binary sizes; sanity check (<1MB target).",
    )
    similarity_vector: list[float] = Field(
        default_factory=list,
        description=(
            "Compact fingerprint used by lib.retrieval to find similar past "
            "sketches across projects."
        ),
    )
    last_updated_iteration: int = Field(default=0, ge=0)
