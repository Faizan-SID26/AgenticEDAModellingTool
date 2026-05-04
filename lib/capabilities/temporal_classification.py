"""Time-ordered classification with regime structure (manufacturing default)."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="temporal_classification",
    description="Binary classification on time-ordered data with regime structure.",
    composition=CapabilityComposition(
        temporal_structure="regime_based",
        leakage_model="stage_frontier",
        target_type="binary",
        validation_strategy="time_split",
        recommendation_type="decision",
    ),
    required_mission_fields=("target_column", "time_column", "success_criterion"),
    default_models=("logreg", "lgbm_binary"),
    default_metrics=("roc_auc", "average_precision", "log_loss", "brier"),
    primary_metric="roc_auc",
    primary_metric_direction=">=",
    sketch_extras_needed=(
        "L1_distributions",
        "L2_joint",
        "L3_regimes",
        "L4_coresets",
        "L5_timeseries",
    ),
    seed_hypothesis_recipe_keys=(
        "regime_specific_submodel",
        "stage_frontier_baseline",
        "lag_join_with_immediate_prior",
    ),
)


def make_splitter():
    """Return a forward-time-split splitter (no shuffling)."""

    def split(
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
        n_splits: int = 5,
    ) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        # Sort by time ascending. If time is None, treat current order as time order.
        if time is None:
            order = np.arange(n_rows)
        else:
            order = np.argsort(np.asarray(list(time)), kind="stable")
        # Expanding window: train on growing prefix, validate on next slice.
        out = []
        chunk = max(1, n_rows // (n_splits + 1))
        for i in range(1, n_splits + 1):
            train_end = i * chunk
            val_end = min((i + 1) * chunk, n_rows)
            tr = order[:train_end]
            va = order[train_end:val_end]
            if len(va) == 0:
                continue
            out.append((tr, va, None))
        if not out:
            # Tiny dataset fallback.
            mid = max(1, n_rows // 2)
            out.append((order[:mid], order[mid:], None))
        return out

    return split
