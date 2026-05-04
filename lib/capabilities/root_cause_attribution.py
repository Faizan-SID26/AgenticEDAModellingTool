"""Root cause attribution: given a known defect, rank features by contribution."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

SPEC = CapabilitySpec(
    key="root_cause_attribution",
    description="Given a defined defect/incident, rank candidate causes by attribution.",
    composition=CapabilityComposition(
        temporal_structure="none",
        leakage_model="none",
        target_type="rank",
        validation_strategy="stratified",
        recommendation_type="ranked_factors",
    ),
    required_mission_fields=("target_column", "success_criterion"),
    default_models=("logreg", "lgbm_binary", "permutation_importance"),
    default_metrics=("roc_auc", "ndcg_at_10", "spearman_rank_corr"),
    primary_metric="ndcg_at_10",
    primary_metric_direction=">=",
    sketch_extras_needed=("L2_joint", "L6_causal", "L7_failure_modes"),
    seed_hypothesis_recipe_keys=(
        "marginal_lift_per_feature",
        "shapley_global",
        "causal_neighbors_only",
    ),
)


def make_splitter():
    """Stratified k-fold."""
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
