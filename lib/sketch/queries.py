"""Sketch tool surface — typed functions returning JSON-serializable dicts.

Wired into the MCP server (`mcp_servers/sketch_server.py`). Each function:
- Takes a `project_dir` (or expects to be invoked with cwd = project dir).
- Reads the appropriate sketch layer.
- Returns JSON-safe primitives.

These are *queries* only. Updates go through `lib.sketch.updaters`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from lib.sketch.l1_distributions import load_l1
from lib.sketch.l2_joint import load_l2
from lib.sketch.l3_regimes import load_l3
from lib.sketch.l5_timeseries import load_l5
from lib.sketch.l6_causal import load_l6
from lib.sketch.l7_failure_modes import load_l7
from lib.sketch.manifest import load_manifest


def _abs(project_dir: Path, rel: str) -> Path:
    return Path(project_dir) / rel


# --- Read queries -------------------------------------------------------


def quantile(project_dir: Path, column: str, q: float) -> dict[str, Any]:
    """Return the q-th quantile of `column` (interpolated from stored percentiles)."""
    manifest = load_manifest(project_dir)
    l1 = load_l1(_abs(project_dir, manifest.l1_path))
    for c in l1:
        if c.column == column and c.dtype == "numeric":
            keys = sorted([(float(k), v) for k, v in c.quantiles.items()])
            if not keys:
                return {"column": column, "q": q, "value": None}
            xs = [k for k, _ in keys]
            ys = [v for _, v in keys]
            if q <= xs[0]:
                return {"column": column, "q": q, "value": float(ys[0])}
            if q >= xs[-1]:
                return {"column": column, "q": q, "value": float(ys[-1])}
            for i in range(len(xs) - 1):
                if xs[i] <= q <= xs[i + 1]:
                    t = (q - xs[i]) / (xs[i + 1] - xs[i])
                    return {"column": column, "q": q, "value": float(ys[i] + t * (ys[i + 1] - ys[i]))}
    return {"column": column, "q": q, "value": None, "error": "column not found or not numeric"}


def distribution(project_dir: Path, column: str) -> dict[str, Any]:
    """Return the L1 column summary for `column`."""
    manifest = load_manifest(project_dir)
    l1 = load_l1(_abs(project_dir, manifest.l1_path))
    for c in l1:
        if c.column == column:
            return c.model_dump()
    return {"error": f"column {column} not found"}


def cardinality(project_dir: Path, column: str) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l1 = load_l1(_abs(project_dir, manifest.l1_path))
    for c in l1:
        if c.column == column:
            return {"column": column, "n_unique_estimate": int(c.n_unique_estimate)}
    return {"error": f"column {column} not found"}


def missingness(project_dir: Path, column: Optional[str] = None) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l1 = load_l1(_abs(project_dir, manifest.l1_path))
    if column is None:
        return {"per_column": {c.column: c.n_missing / max(c.n_total, 1) for c in l1}}
    for c in l1:
        if c.column == column:
            return {
                "column": column,
                "n_missing": int(c.n_missing),
                "fraction_missing": c.n_missing / max(c.n_total, 1),
            }
    return {"error": f"column {column} not found"}


def top_interactions(project_dir: Path, top_k: int = 5) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l2 = load_l2(_abs(project_dir, manifest.l2_path))
    return {"top_interactions": list(l2.top_interactions[:top_k])}


def conditional_dependence(
    project_dir: Path,
    a: str,
    b: str,
    given: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Approximate via L6 edge weight (or absence) — exact CI tests would
    require raw data, which the sketch deliberately does not retain.
    """
    manifest = load_manifest(project_dir)
    l6 = load_l6(_abs(project_dir, manifest.l6_path))
    given = given or []
    for e in l6.edges:
        if {e["src"], e["dst"]} == {a, b}:
            return {
                "a": a,
                "b": b,
                "given": given,
                "edge_present": True,
                "ci_test_pval": float(e.get("ci_test_pval", 1.0)),
                "weight": float(e.get("weight", 0.0)),
                "kind": e.get("kind"),
            }
    return {"a": a, "b": b, "given": given, "edge_present": False}


def principal_components(project_dir: Path, top_k: int = 5) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l2 = load_l2(_abs(project_dir, manifest.l2_path))
    return {
        "n_components": l2.n_components,
        "explained_variance_ratio": list(l2.explained_variance_ratio[:top_k]),
        "component_loadings_top": {k: l2.component_loadings_top.get(k, []) for k in list(l2.component_loadings_top.keys())[:top_k]},
    }


def regimes(project_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l3 = load_l3(_abs(project_dir, manifest.l3_path))
    return l3.model_dump()


def regime_compare(project_dir: Path, regime_a: int, regime_b: int) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l3 = load_l3(_abs(project_dir, manifest.l3_path))
    n = l3.n_regimes
    if regime_a < 0 or regime_b < 0 or regime_a >= n or regime_b >= n:
        return {"error": f"regime indices out of range (n={n})"}
    means_a = {k: v[regime_a] for k, v in l3.regime_means.items() if len(v) > regime_a}
    means_b = {k: v[regime_b] for k, v in l3.regime_means.items() if len(v) > regime_b}
    return {
        "regime_a": regime_a,
        "regime_b": regime_b,
        "size_a": l3.regime_sizes[regime_a] if regime_a < len(l3.regime_sizes) else None,
        "size_b": l3.regime_sizes[regime_b] if regime_b < len(l3.regime_sizes) else None,
        "means_a": means_a,
        "means_b": means_b,
        "target_dist_a": l3.regime_target_distribution[regime_a]
        if regime_a < len(l3.regime_target_distribution)
        else {},
        "target_dist_b": l3.regime_target_distribution[regime_b]
        if regime_b < len(l3.regime_target_distribution)
        else {},
    }


def motifs(project_dir: Path, column: str) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l5 = load_l5(_abs(project_dir, manifest.l5_path))
    for c in l5:
        if c.column == column:
            return {
                "column": column,
                "sax_top": list(c.sax_top_motifs),
                "matrix_profile_motifs": list(c.matrix_profile_top_motifs),
            }
    return {"error": f"column {column} not in L5"}


def discords(project_dir: Path, column: str) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l5 = load_l5(_abs(project_dir, manifest.l5_path))
    for c in l5:
        if c.column == column:
            return {"column": column, "matrix_profile_discords": list(c.matrix_profile_top_discords)}
    return {"error": f"column {column} not in L5"}


def causal_neighbors(project_dir: Path, node: str, top_k: int = 5) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    l6 = load_l6(_abs(project_dir, manifest.l6_path))
    edges = [
        e for e in l6.edges if e["src"] == node or e["dst"] == node
    ]
    edges = sorted(edges, key=lambda e: float(e.get("weight", 0.0)), reverse=True)[:top_k]
    return {"node": node, "edges": edges}


def confounder_candidates(project_dir: Path, treatment: str, outcome: str, top_k: int = 5) -> dict[str, Any]:
    """Return nodes adjacent to BOTH treatment and outcome (confounder candidates)."""
    manifest = load_manifest(project_dir)
    l6 = load_l6(_abs(project_dir, manifest.l6_path))
    nbrs_t = {e["src"] if e["dst"] == treatment else e["dst"] for e in l6.edges if treatment in (e["src"], e["dst"])}
    nbrs_o = {e["src"] if e["dst"] == outcome else e["dst"] for e in l6.edges if outcome in (e["src"], e["dst"])}
    cand = list((nbrs_t & nbrs_o) - {treatment, outcome})[:top_k]
    return {"treatment": treatment, "outcome": outcome, "candidates": cand}


def failure_clusters(project_dir: Path, top_k: int = 5) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    catalog = load_l7(_abs(project_dir, manifest.l7_path))
    catalog = sorted(catalog, key=lambda c: c.n_observations, reverse=True)[:top_k]
    return {"clusters": [c.model_dump() for c in catalog]}


def match_residuals(project_dir: Path, signature: dict[str, float]) -> dict[str, Any]:
    """Return the closest existing failure cluster to `signature` (without modifying)."""
    from lib.sketch.l7_failure_modes import _mahalanobis_diag

    manifest = load_manifest(project_dir)
    catalog = load_l7(_abs(project_dir, manifest.l7_path))
    if not catalog:
        return {"matched": False, "best_cluster_id": None, "distance": None}
    best = min(catalog, key=lambda c: _mahalanobis_diag(signature, c))
    d = _mahalanobis_diag(signature, best)
    return {"matched": True, "best_cluster_id": best.cluster_id, "distance": float(d)}


# --- Quick fits (used by the planner / hypothesis generator) ------------


def fit_quick(
    project_dir: Path,
    capability_key: str,
    features: list[str],
    target: str,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a small model on the L4 coreset and return calibrated metric.

    Used by the agent to cheaply triage candidate features before
    committing to a full iteration. Falls back to a single train/val
    split using the coreset.
    """
    import pandas as pd

    from lib.eval import dispatch_metrics  # may be a stub at this stage
    from lib.registry import factory as registry_factory  # likewise

    manifest = load_manifest(project_dir)
    cs_path = next((p for p in manifest.l4_paths if capability_key in p), None)
    if cs_path is None:
        return {"error": f"no L4 coreset for capability {capability_key}"}
    df = pd.read_parquet(_abs(project_dir, cs_path))
    if target not in df.columns:
        return {"error": f"target {target} not in coreset"}
    feats = [f for f in features if f in df.columns and f != target]
    if not feats:
        return {"error": "no requested features available in coreset"}
    X = df[feats].select_dtypes(include="number").fillna(0).values
    y = df[target].values
    sample_w = df["weight"].values if "weight" in df.columns else None

    n = len(X)
    n_train = int(n * 0.8)
    Xtr, Xva = X[:n_train], X[n_train:]
    ytr, yva = y[:n_train], y[n_train:]

    model = registry_factory(capability_key).default(seed=seed)
    if sample_w is not None:
        try:
            model.fit(Xtr, ytr, sample_weight=sample_w[:n_train])
        except TypeError:
            model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr)

    yp_va = _predict_proba_or_value(model, Xva)
    metrics = dispatch_metrics(capability_key, yva, yp_va)
    return {
        "capability": capability_key,
        "features": feats,
        "metrics": metrics,
        "n_train": int(n_train),
        "n_val": int(n - n_train),
    }


def cross_validate_quick(
    project_dir: Path,
    capability_key: str,
    features: list[str],
    target: str,
    *,
    seed: int = 0,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Like fit_quick but with k-fold."""
    import numpy as np
    import pandas as pd

    from lib.eval import dispatch_metrics
    from lib.registry import factory as registry_factory

    manifest = load_manifest(project_dir)
    cs_path = next((p for p in manifest.l4_paths if capability_key in p), None)
    if cs_path is None:
        return {"error": f"no L4 coreset for capability {capability_key}"}
    df = pd.read_parquet(_abs(project_dir, cs_path))
    feats = [f for f in features if f in df.columns and f != target]
    if not feats:
        return {"error": "no requested features available in coreset"}
    X = df[feats].select_dtypes(include="number").fillna(0).values
    y = df[target].values

    n = len(X)
    folds = np.array_split(np.arange(n), n_splits)
    metrics_per_fold: list[dict[str, float]] = []
    for k in range(n_splits):
        va = folds[k]
        tr = np.concatenate([folds[i] for i in range(n_splits) if i != k])
        model = registry_factory(capability_key).default(seed=seed + k)
        model.fit(X[tr], y[tr])
        yp = _predict_proba_or_value(model, X[va])
        metrics_per_fold.append(dispatch_metrics(capability_key, y[va], yp))
    return {
        "capability": capability_key,
        "features": feats,
        "metrics_per_fold": metrics_per_fold,
        "metrics_mean": _mean_dicts(metrics_per_fold),
    }


def _predict_proba_or_value(model: Any, X) -> Any:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:  # noqa: BLE001
            return model.predict(X)
    return model.predict(X)


def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(r.keys() for r in rows))
    return {k: float(sum(r.get(k, 0.0) for r in rows) / len(rows)) for k in keys}
