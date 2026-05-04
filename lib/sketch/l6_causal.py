"""L6: causal DAG hints via PC algorithm.

Implements a *basic* PC: start from the complete graph on numeric
columns; for each edge (a,b) test conditional independence given a small
candidate separator set; remove the edge if the test rejects dependence.
Orient v-structures lightly (a→c←b when c is not in the separator of (a,b)).

Independence test: partial correlation (Fisher z-transform). Robust to
small n; cheap; consistent with the "structural updaters never call LLM"
constraint.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

from lib.schemas.sketch import L6CausalSummary

_log = logging.getLogger("eda.sketch.l6")


def _partial_corr(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None) -> float:
    """Partial correlation of x,y given z (z may be None for marginal corr)."""
    if z is None or z.shape[1] == 0:
        if x.size < 3:
            return 0.0
        c, _ = stats.pearsonr(x, y)
        return 0.0 if np.isnan(c) else float(c)
    # Regress x on z, y on z, then correlate residuals.
    from numpy.linalg import lstsq

    z_aug = np.column_stack([np.ones(z.shape[0]), z])
    bx, *_ = lstsq(z_aug, x, rcond=None)
    by, *_ = lstsq(z_aug, y, rcond=None)
    rx = x - z_aug @ bx
    ry = y - z_aug @ by
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    c, _ = stats.pearsonr(rx, ry)
    return 0.0 if np.isnan(c) else float(c)


def _ci_test_pval(r: float, n: int, k_cond: int) -> float:
    """Fisher z-transform p-value for partial correlation."""
    if n - k_cond - 3 <= 0:
        return 1.0
    z = 0.5 * np.log((1 + r) / (1 - r + 1e-12))
    se = 1.0 / np.sqrt(n - k_cond - 3)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z) / se))
    return float(p)


def build_l6(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    max_cond_set: int = 2,
    max_columns: int = 25,
    seed: int = 0,
) -> L6CausalSummary:
    """Run a small PC-style pass on the top numeric columns by variance."""
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty or num.shape[0] < 30:
        return L6CausalSummary(nodes=[], edges=[], alpha=alpha)
    # Pick the top-N highest-variance columns; deterministic.
    var = num.var(numeric_only=True).sort_values(ascending=False)
    cols = list(var.index[:max_columns])
    sub = num[cols].dropna()
    if sub.empty:
        return L6CausalSummary(nodes=cols, edges=[], alpha=alpha)
    n = len(sub)
    arr = sub.values

    edges: dict[tuple[str, str], dict] = {}
    # Initialize complete undirected graph with marginal correlation.
    for i, j in combinations(range(len(cols)), 2):
        r = _partial_corr(arr[:, i], arr[:, j])
        p = _ci_test_pval(r, n, k_cond=0)
        edges[(cols[i], cols[j])] = {
            "src": cols[i],
            "dst": cols[j],
            "kind": "undirected",
            "ci_test_pval": p,
            "weight": float(abs(r)),
        }

    # Phase 1: remove edges with conditional independence given small subsets.
    cond_sets: dict[tuple[str, str], list[str]] = {}
    for size in range(1, max_cond_set + 1):
        to_remove: list[tuple[str, str]] = []
        for (a, b), edge in list(edges.items()):
            adj = [c for c in cols if c != a and c != b]
            for combo in combinations(adj, size):
                z = sub[list(combo)].values
                r = _partial_corr(sub[a].values, sub[b].values, z)
                p = _ci_test_pval(r, n, k_cond=size)
                if p > alpha:
                    to_remove.append((a, b))
                    cond_sets[(a, b)] = list(combo)
                    break
        for k in to_remove:
            edges.pop(k, None)

    # Phase 2: orient v-structures (a-c-b with c NOT in cond_sets[(a,b)]).
    for (a, b), sep in list(cond_sets.items()):
        for c in cols:
            if c in (a, b):
                continue
            if (a, c) in edges and (c, b) in edges and c not in sep:
                edges[(a, c)]["kind"] = "directed"
                edges[(a, c)]["src"], edges[(a, c)]["dst"] = a, c
                edges[(c, b)]["kind"] = "directed"
                edges[(c, b)]["src"], edges[(c, b)]["dst"] = b, c

    return L6CausalSummary(
        nodes=cols,
        edges=list(edges.values()),
        test="partial_correlation",
        alpha=alpha,
    )


def save_l6(summary: L6CausalSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def load_l6(path: Path) -> L6CausalSummary:
    return L6CausalSummary.model_validate_json(Path(path).read_text(encoding="utf-8"))
