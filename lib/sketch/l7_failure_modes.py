"""L7: failure-mode online cluster catalog.

Starts empty at /bootstrap. The updater appends one or more failure
clusters per failed/warning experiment, using:
- Welford online mean and (diagonal) covariance updates.
- Mahalanobis-distance matching to existing clusters before deciding to
  add a new one.

Persisted as a JSONL file (one cluster per line) for cheap append-only
updates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from lib.schemas.sketch import L7FailureClusterSummary


_DEFAULT_MATCH_THRESHOLD = 5.0  # Mahalanobis distance threshold (squared)


def _mahalanobis_diag(point: dict[str, float], cluster: L7FailureClusterSummary) -> float:
    """Squared Mahalanobis distance using only the diagonal of the inverse covariance."""
    d2 = 0.0
    for k, v in point.items():
        mu = cluster.centroid.get(k)
        inv_var = cluster.inv_cov_diag.get(k)
        if mu is None or inv_var is None:
            continue
        d2 += inv_var * (v - mu) ** 2
    return d2


def match_or_create(
    catalog: list[L7FailureClusterSummary],
    point: dict[str, float],
    *,
    iteration: int,
    threshold: float = _DEFAULT_MATCH_THRESHOLD,
    cluster_id_factory=None,
) -> tuple[list[L7FailureClusterSummary], str, bool]:
    """Match `point` to an existing cluster or create a new one.

    Returns the updated catalog, the matched cluster id, and a `created`
    flag. Welford-update the matched cluster's centroid and per-feature
    inverse-variance proxy.
    """
    cluster_id_factory = cluster_id_factory or (
        lambda i: f"FC-{i:03d}"
    )
    best: Optional[L7FailureClusterSummary] = None
    best_d = float("inf")
    for c in catalog:
        d = _mahalanobis_diag(point, c)
        if d < best_d:
            best_d = d
            best = c
    if best is not None and best_d <= threshold:
        # Welford online update.
        n_new = best.n_observations + 1
        new_centroid = dict(best.centroid)
        new_inv = dict(best.inv_cov_diag)
        for k, v in point.items():
            old_mu = new_centroid.get(k, v)
            new_mu = old_mu + (v - old_mu) / n_new
            # Maintain a running "M2" via inv_cov_diag = 1 / (var + eps).
            cur_var = 1.0 / max(new_inv.get(k, 1e-3), 1e-9)
            cur_M2 = cur_var * max(best.n_observations - 1, 1)
            new_M2 = cur_M2 + (v - old_mu) * (v - new_mu)
            new_var = new_M2 / max(n_new - 1, 1)
            new_inv[k] = 1.0 / max(new_var, 1e-9)
            new_centroid[k] = new_mu
        updated = L7FailureClusterSummary(
            cluster_id=best.cluster_id,
            n_observations=n_new,
            centroid=new_centroid,
            inv_cov_diag=new_inv,
            mean_residual=best.mean_residual,
            last_updated_iteration=iteration,
            label=best.label,
        )
        new_catalog = [c if c.cluster_id != best.cluster_id else updated for c in catalog]
        return new_catalog, best.cluster_id, False

    # Create new cluster.
    new_id = cluster_id_factory(len(catalog) + 1)
    new_cluster = L7FailureClusterSummary(
        cluster_id=new_id,
        n_observations=1,
        centroid=dict(point),
        inv_cov_diag={k: 1.0 for k in point},  # variance unknown → identity
        mean_residual=0.0,
        last_updated_iteration=iteration,
    )
    return [*catalog, new_cluster], new_id, True


def empty_catalog() -> list[L7FailureClusterSummary]:
    """Initial L7 state."""
    return []


def save_l7(catalog: Iterable[L7FailureClusterSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for c in catalog:
            f.write(c.model_dump_json() + "\n")


def load_l7(path: Path) -> list[L7FailureClusterSummary]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[L7FailureClusterSummary] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(L7FailureClusterSummary.model_validate_json(line))
    return out
