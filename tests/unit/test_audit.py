"""Audit gate tests."""
from __future__ import annotations

from lib.audit import audit_features
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)


def _mission() -> Mission:
    return Mission(
        project_name="audit_test",
        domain="manufacturing",
        capability=CapabilityComposition(
            temporal_structure="regime_based",
            leakage_model="stage_frontier",
            target_type="binary",
            validation_strategy="time_split",
            recommendation_type="decision",
        ),
        target_column="y",
        time_column="t",
        forbidden_columns=["downstream_qc"],
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
        budget=MissionBudget(token_cap=1),
        business_question="x",
    )


def test_audit_passes_clean_features():
    res = audit_features(_mission(), ["x_temp", "x_press"])
    assert res.ok and not res.forbidden_used


def test_audit_blocks_forbidden_outside_probe():
    res = audit_features(_mission(), ["x_temp", "downstream_qc"])
    assert res.ok is False
    assert "downstream_qc" in res.forbidden_used


def test_audit_allows_leakage_probe():
    res = audit_features(_mission(), ["x_temp", "downstream_qc"], plan_area="leakage_probe")
    assert res.ok is True
    assert "downstream_qc" in res.forbidden_used  # still recorded as used
    assert any("leakage probe" in w for w in res.warnings)


def test_audit_blocks_target_in_features():
    res = audit_features(_mission(), ["y", "x_temp"])
    assert res.ok is False
    assert res.target_used_as_feature is True
