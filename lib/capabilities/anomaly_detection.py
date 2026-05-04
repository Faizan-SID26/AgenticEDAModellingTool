"""Unsupervised anomaly detection."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="anomaly_detection",
    description="Unsupervised anomaly detection on tabular or temporal data.",
    composition=CapabilityComposition(
        temporal_structure="none",
        leakage_model="none",
        target_type="outlier_score",
        validation_strategy="stratified",
        recommendation_type="alert_policy",
    ),
    required_mission_fields=("target_column", "success_criterion"),
    default_models=("isolation_forest", "ocsvm", "lof"),
    default_metrics=("roc_auc", "average_precision", "precision_at_k"),
    primary_metric="average_precision",
    primary_metric_direction=">=",
    sketch_extras_needed=("L1_distributions", "L2_joint", "L7_failure_modes"),
    seed_hypothesis_recipe_keys=("isolation_forest_baseline", "robust_pca_residual"),
)


def make_splitter():
    """Stratified k-fold; if no labels, simple k-fold."""
    from sklearn.model_selection import KFold, StratifiedKFold

    def split(
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
    ) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if y is not None and len(np.unique(y)) > 1:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            return [(tr, va, None) for tr, va in kf.split(np.zeros(n_rows), y)]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        return [(tr, va, None) for tr, va in kf.split(np.zeros(n_rows))]

    return split
