"""Tabular binary classification (no temporal structure)."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="tabular_classification",
    description="Binary classification on tabular data without temporal structure.",
    composition=CapabilityComposition(
        temporal_structure="none",
        leakage_model="none",
        target_type="binary",
        validation_strategy="stratified",
        recommendation_type="decision",
    ),
    required_mission_fields=("target_column", "success_criterion"),
    default_models=("logreg", "lgbm_binary"),
    default_metrics=("roc_auc", "average_precision", "log_loss", "brier"),
    primary_metric="roc_auc",
    primary_metric_direction=">=",
    sketch_extras_needed=("L1_distributions", "L2_joint", "L4_coresets"),
    seed_hypothesis_recipe_keys=("regularized_baseline", "lgbm_default"),
)


def make_splitter():
    """Return a stratified k-fold splitter."""
    from sklearn.model_selection import StratifiedKFold

    def split(
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
    ) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if y is None:
            raise ValueError("stratified split requires y")
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return [(tr, va, None) for tr, va in kf.split(np.zeros(n_rows), y)]

    return split
