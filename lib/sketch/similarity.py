"""Similarity vectors for cross-project sketch retrieval.

Compact, fixed-length numeric fingerprints derived from sketch layers.
Two sketches are *similar* if their fingerprints are close in cosine
distance. Used by `lib.retrieval` to seed new projects with patterns
from past similar projects.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from lib.schemas.sketch import (
    L1ColumnSummary,
    L2JointSummary,
    L3RegimeSummary,
    L7FailureClusterSummary,
    SketchManifest,
)


def _quantile_signature(l1: list[L1ColumnSummary]) -> list[float]:
    """Average of robust quantile statistics across numeric columns."""
    p = ["0.25", "0.50", "0.75", "0.95"]
    sig = []
    for q in p:
        vals = [c.quantiles.get(q) for c in l1 if c.dtype == "numeric" and c.quantiles.get(q) is not None]
        sig.append(float(np.mean(vals)) if vals else 0.0)
    sig.append(float(np.mean([c.n_missing / max(c.n_total, 1) for c in l1])))
    sig.append(float(np.mean([c.n_unique_estimate / max(c.n_total, 1) for c in l1])))
    return sig


def _l2_signature(l2: L2JointSummary) -> list[float]:
    """Top-5 PCA variance ratios padded with zeros, plus top-3 MI scores."""
    evr = list(l2.explained_variance_ratio[:5]) + [0.0] * (5 - min(len(l2.explained_variance_ratio), 5))
    mi_scores = [float(it.get("mutual_info", 0.0)) for it in (l2.top_interactions or [])[:3]]
    mi_scores += [0.0] * (3 - len(mi_scores))
    return evr + mi_scores


def _l3_signature(l3: L3RegimeSummary) -> list[float]:
    """Number of regimes, max regime fraction, target distribution heterogeneity."""
    n_total = sum(l3.regime_sizes) or 1
    max_frac = max(l3.regime_sizes, default=0) / n_total
    n_reg = float(l3.n_regimes)
    if l3.regime_target_distribution:
        rates = [float(d.get("rate", d.get("mean", 0.0))) for d in l3.regime_target_distribution]
        het = float(np.std(rates)) if len(rates) > 1 else 0.0
    else:
        het = 0.0
    return [n_reg, float(max_frac), het]


def _l7_signature(catalog: list[L7FailureClusterSummary]) -> list[float]:
    """Number of clusters, total observations, mean cluster size."""
    n = float(len(catalog))
    obs = float(sum(c.n_observations for c in catalog))
    return [n, obs, obs / n if n > 0 else 0.0]


def compute_similarity_vector(
    l1: list[L1ColumnSummary],
    l2: L2JointSummary,
    l3: L3RegimeSummary,
    l7: list[L7FailureClusterSummary],
) -> list[float]:
    """Concatenate per-layer signatures into a single fixed-length vector."""
    return _quantile_signature(l1) + _l2_signature(l2) + _l3_signature(l3) + _l7_signature(l7)


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity, robust to zero norms."""
    av = np.array(list(a), dtype=float)
    bv = np.array(list(b), dtype=float)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return 0.0
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def manifest_similarity(a: SketchManifest, b: SketchManifest) -> float:
    """Cosine of the two manifests' similarity_vector fields."""
    return cosine(a.similarity_vector, b.similarity_vector)
