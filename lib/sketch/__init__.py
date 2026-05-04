"""The Process Data Sketch.

7 structural layers (L1..L7) + a separate annotations layer.
- Built once at /bootstrap from a joined parquet.
- Updated deterministically after every iteration.
- Queried via the MCP tool surface.

Public API mirrors the layers and adds `build`, `load_manifest`,
`update_after_experiment`.
"""
from __future__ import annotations

from lib.sketch.builder import build_sketch
from lib.sketch.manifest import load_manifest, save_manifest
from lib.sketch.queries import (
    cardinality,
    conditional_dependence,
    distribution,
    failure_clusters,
    missingness,
    motifs,
    principal_components,
    quantile,
    regimes,
    top_interactions,
)
from lib.sketch.updaters import update_after_experiment

__all__ = [
    "build_sketch",
    "load_manifest",
    "save_manifest",
    "update_after_experiment",
    "cardinality",
    "conditional_dependence",
    "distribution",
    "failure_clusters",
    "missingness",
    "motifs",
    "principal_components",
    "quantile",
    "regimes",
    "top_interactions",
]
