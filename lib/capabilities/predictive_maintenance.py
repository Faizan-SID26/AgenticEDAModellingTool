"""Predictive maintenance: time-to-event / survival analysis."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="predictive_maintenance",
    description="Time-to-event modeling (e.g., failure prediction) per entity.",
    composition=CapabilityComposition(
        temporal_structure="regime_based",
        leakage_model="forecast_horizon",
        target_type="time_to_event",
        validation_strategy="group_kfold",
        recommendation_type="alert_policy",
    ),
    required_mission_fields=(
        "target_column",
        "time_column",
        "group_column",
        "success_criterion",
    ),
    default_models=("cox_ph", "lgbm_survival"),
    default_metrics=("concordance_index", "ibs", "cumulative_dynamic_auc"),
    primary_metric="concordance_index",
    primary_metric_direction=">=",
    sketch_extras_needed=("L1_distributions", "L3_regimes", "L5_timeseries", "L6_causal"),
    seed_hypothesis_recipe_keys=(
        "cox_baseline",
        "regime_specific_hazard",
        "leading_indicator_features",
    ),
)


def make_splitter():
    """Group-aware k-fold so an entity never appears in both train and val."""
    from sklearn.model_selection import GroupKFold

    def split(
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
    ) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if groups is None:
            raise ValueError("predictive_maintenance requires a group_column")
        gkf = GroupKFold(n_splits=5)
        return [
            (tr, va, None)
            for tr, va in gkf.split(np.zeros(n_rows), y, groups=np.asarray(list(groups)))
        ]

    return split
