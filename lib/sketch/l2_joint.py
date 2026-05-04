"""L2: joint structure — top-k PCA + sparse top-K interactions.

Built on the numeric subset of the source frame. PCA is done on
standardized features. Interactions are scored by mutual information
(sklearn) on the coreset and kept sparsely as (col_a, col_b, mutual_info).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
    df: pd.DataFrame, target: str, top_k: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Compute pairwise mutual information between numeric columns and target.

    Returns ranked pairs. If `target` is not in df, falls back to scoring
    pairs by absolute Pearson correlation in the numeric subset.
    """
    num = _select_numeric(df)
    if num.empty:
        return []

    pairs: list[dict[str, Any]] = []
    if target in df.columns:
        y = df[target]
        try:
            if pd.api.types.is_numeric_dtype(y):
                from sklearn.feature_selection import mutual_info_regression as mi_fn

                yv = y.fillna(y.median()).values
            else:
                from sklearn.feature_selection import mutual_info_classif as mi_fn

                yv = pd.factorize(y)[0]
            X = num.drop(columns=[target], errors="ignore").values
            mi = mi_fn(X, yv, random_state=seed)
            cols = [c for c in num.columns if c != target]
            order = np.argsort(mi)[::-1][:top_k]
            for rank, idx in enumerate(order, start=1):
                pairs.append(
                    {
                        "col_a": cols[idx],
                        "col_b": target,
                        "mutual_info": float(mi[idx]),
                        "rank": int(rank),
                        "interaction_strength_residual": 0.0,
                    }
                )
            if pairs:
                return pairs
        except Exception as e:  # noqa: BLE001
            _log.debug("MI to target failed: %s; falling back to correlation", e)

    # Pairwise correlation fallback.
    corr = num.corr(method="pearson").abs()
    np.fill_diagonal(corr.values, np.nan)
    flat = corr.unstack().dropna().sort_values(ascending=False)
    seen = set()
    for (a, b), v in flat.items():
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


def build_l2(df: pd.DataFrame, target: str = "", *, k: int = 20, top_k_interactions: int = 10, seed: int = 0) -> L2JointSummary:
    """Build the L2 joint summary."""
    num = _select_numeric(df)
    evr, loadings = _safe_pca(num, k=k, seed=seed)
    interactions = _safe_top_interactions(df, target=target, top_k=top_k_interactions, seed=seed)
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
