"""Capability composition + registry tests."""
from __future__ import annotations

import pytest

from lib.capabilities import (
    composition_signature,
    find_compatible,
    get,
    list_capabilities,
    validate_composition,
)
from lib.schemas.mission import CapabilityComposition


def test_all_capabilities_registered():
    keys = list_capabilities()
    expected = {
        "tabular_classification",
        "tabular_regression",
        "temporal_classification",
        "forecasting",
        "predictive_maintenance",
        "anomaly_detection",
        "root_cause_attribution",
    }
    assert expected.issubset(set(keys))


def test_get_returns_spec():
    spec = get("tabular_classification")
    assert spec.primary_metric == "roc_auc"


def test_validate_composition_picks_best():
    cap = CapabilityComposition(
        temporal_structure="regime_based",
        leakage_model="stage_frontier",
        target_type="binary",
        validation_strategy="time_split",
        recommendation_type="decision",
    )
    spec = validate_composition(cap)
    assert spec.key == "temporal_classification"


def test_find_compatible_returns_at_least_one():
    cap = CapabilityComposition(
        temporal_structure="none",
        leakage_model="none",
        target_type="binary",
        validation_strategy="stratified",
        recommendation_type="decision",
    )
    matches = find_compatible(cap)
    assert any(s.key == "tabular_classification" for s in matches)


def test_signature_is_stable():
    cap = CapabilityComposition(
        temporal_structure="regime_based",
        leakage_model="stage_frontier",
        target_type="binary",
        validation_strategy="time_split",
        recommendation_type="decision",
    )
    sig = composition_signature(cap)
    assert sig == "binary|regime_based|stage_frontier|time_split|decision"


def test_unknown_composition_raises():
    cap = CapabilityComposition(
        temporal_structure="seasonal",
        leakage_model="forecast_horizon",
        target_type="multi_horizon",
        validation_strategy="rolling_origin",
        recommendation_type="forecast",
    )
    spec = validate_composition(cap)
    assert spec.key == "forecasting"
