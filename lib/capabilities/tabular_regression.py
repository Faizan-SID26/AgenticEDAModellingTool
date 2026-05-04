"""Tabular regression on continuous targets without temporal structure."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="tabular_regression",
    description="Continuous-target regression on tabular data without temporal structure.",
    composition=CapabilityComposition(
        temporal_structure="none",
        leakage_model="none",
        target_type="continuous",
        validation_strategy="group_kfold",
        recommendation_type="ranked_factors",
    ),
    required_mission_fields=("target_column", "success_criterion"),
    default_models=("ridge", "lgbm_regressor"),
    default_metrics=("rmse", "mae", "r2", "mape"),
    primary_metric="rmse",
    primary_metric_direction="<=",
    sketch_extras_needed=("L1_distributions", "L2_joint", "L4_coresets", "L6_causal"),
    seed_hypothesis_recipe_keys=("ridge_baseline", "lgbm_regressor"),
)


def make_splitter():
    """Return a (group-aware) k-fold splitter."""
    from sklearn.model_selection import GroupKFold, KFold

    def split(
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
        y: Optional[np.ndarray] = None,
    ) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if groups is not None:
            gkf = GroupKFold(n_splits=5)
            return [
                (tr, va, None)
                for tr, va in gkf.split(np.zeros(n_rows), y, groups=np.asarray(list(groups)))
            ]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        return [(tr, va, None) for tr, va in kf.split(np.zeros(n_rows))]

    return split
