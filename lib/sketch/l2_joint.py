"""L2: joint structure — top-k PCA + sparse top-K interactions.

Built on the numeric subset of the source frame. PCA is done on
standardized features. Interactions are scored by mutual information
(sklearn) on the coreset and kept sparsely as (col_a, col_b, mutual_info).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from lib.schemas.sketch import L2JointSummary

_log = logging.getLogger("eda.sketch.l2")


def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric subset of `df` with NaNs filled by column median."""
    num = df.select_dtypes(include=[np.number]).copy()
    return num.fillna(num.median(numeric_only=True)) if not num.empty else num


def _safe_pca(num: pd.DataFrame, k: int, seed: int = 0) -> tuple[list[float], dict[str, list[tuple[str, float]]]]:
    """PCA with k components; returns (explained_variance_ratio, top loadings per component)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if num.shape[1] == 0 or num.shape[0] < 2:
        return [], {}
    k_eff = min(k, num.shape[1], num.shape[0])
    if k_eff < 1:
        return [], {}
    X = StandardScaler().fit_transform(num.values)
    pca = PCA(n_components=k_eff, random_state=seed)
    pca.fit(X)
    evr = list(map(float, pca.explained_variance_ratio_))
    cols = list(num.columns)
    top_load: dict[str, list[tuple[str, float]]] = {}
    for i, comp in enumerate(pca.components_):
        idx = np.argsort(np.abs(comp))[-min(8, len(comp)) :][::-1]
        top_load[f"PC{i + 1}"] = [(cols[j], float(comp[j])) for j in idx]
    return evr, top_load


def _safe_top_interactions(
    df: pd.DataFrame,
    target: str,
    *,
    top_k: int = 10,
    forbidden: Optional[Iterable[str]] = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compute top-K *feature-feature* interactions on the numeric subset.

    Filters ``target`` and ``forbidden`` columns out of both sides of every
    pair before materializing — top_interactions is consumed directly by
    the ``engineered:interactions_top5`` DSL token, so leaving the target
    in any pair would manifest as a leakage feature
    (``X__feature_x_target``).

    Strategy:
        1. Drop target + forbidden + non-numeric columns.
        2. Rank candidate features by univariate MI vs the target (so we
           shortlist the most predictive features), then return top-K
           feature-feature pairs from that shortlist by absolute Pearson
           correlation. This keeps interactions semantically pair-wise
           (no target on either side) while still biasing toward pairs
           that involve high-signal columns.
        3. If MI is unavailable (no target / failure), fall back to
           ranking pairs purely by absolute Pearson correlation.
    """
    forbidden_set = set(forbidden or ())
    forbidden_set.add(target)

    num = _select_numeric(df).drop(columns=list(forbidden_set), errors="ignore")
    if num.empty or num.shape[1] < 2:
        return []

    candidate_cols = list(num.columns)

    # Step 1: shortlist candidates by univariate MI vs target (if available).
    if target in df.columns:
        try:
            y = df[target]
            if pd.api.types.is_numeric_dtype(y):
                from sklearn.feature_selection import mutual_info_regression as mi_fn

                yv = y.fillna(y.median()).values
            else:
                from sklearn.feature_selection import mutual_info_classif as mi_fn

                yv = pd.factorize(y)[0]
            mi = mi_fn(num.values, yv, random_state=seed)
            order = np.argsort(mi)[::-1]
            # Keep top-2K candidates so we still see (top × second-top)
            # interactions without the pair count blowing up.
            n_keep = min(len(candidate_cols), max(2 * top_k, 8))
            candidate_cols = [candidate_cols[i] for i in order[:n_keep]]
        except Exception as e:  # noqa: BLE001
            _log.debug("MI shortlist failed (%s); using all features", e)

    # Step 2: rank feature-feature pairs by |corr| within the shortlist.
    sub = num[candidate_cols]
    corr = sub.corr(method="pearson").abs()
    np.fill_diagonal(corr.values, np.nan)
    flat = corr.unstack().dropna().sort_values(ascending=False)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for (a, b), v in flat.items():
        # Defense in depth: must never include target or forbidden.
        if a in forbidden_set or b in forbidden_set or a == b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "col_a": a,
                "col_b": b,
                "mutual_info": float(v),
                "rank": len(pairs) + 1,
                "interaction_strength_residual": 0.0,
            }
        )
        if len(pairs) >= top_k:
            break
    return pairs


def build_l2(
    df: pd.DataFrame,
    target: str = "",
    *,
    k: int = 20,
    top_k_interactions: int = 10,
    forbidden: Optional[Iterable[str]] = None,
    seed: int = 0,
) -> L2JointSummary:
    """Build the L2 joint summary.

    ``forbidden`` (typically ``MISSION.forbidden_columns``) is stripped
    from both PCA inputs and the interaction pool so downstream
    ``engineered:interactions_top5`` cannot accidentally materialize a
    leakage feature.
    """
    forbidden_set = set(forbidden or ())
    if target:
        forbidden_set.add(target)
    num = _select_numeric(df).drop(columns=list(forbidden_set), errors="ignore")
    evr, loadings = _safe_pca(num, k=k, seed=seed)
    interactions = _safe_top_interactions(
        df,
        target=target,
        top_k=top_k_interactions,
        forbidden=forbidden,
        seed=seed,
    )
    return L2JointSummary(
        n_components=max(1, len(evr) or 1),
        explained_variance_ratio=evr,
        component_loadings_top=loadings,
        top_interactions=interactions,
    )


def save_l2(summary: L2JointSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def load_l2(path: Path) -> L2JointSummary:
    return L2JointSummary.model_validate_json(Path(path).read_text(encoding="utf-8"))
