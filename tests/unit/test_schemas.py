"""Schema validation tests across all artifacts."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lib import SCHEMA_VERSION, __version__
from lib.schemas.budget import BudgetLedgerEntry
from lib.schemas.experiment import ExperimentResult, FitMetrics, SkepticResult
from lib.schemas.knowledge import FailureModeEntry, HypothesisLibraryEntry, KnowledgeBundle
from lib.schemas.mission import (
    CapabilityComposition,
    JoinSpec,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.schemas.plan import PlanDict, PriorEvidence
from lib.schemas.project_meta import ProjectMeta
from lib.schemas.question import Question, QuestionBatch
from lib.schemas.recommendation import Recommendation
from lib.schemas.sketch import L1ColumnSummary, SketchAnnotation, SketchManifest


# --- Capability composition validators ----------------------------------


def test_capability_basic_ok():
    cap = CapabilityComposition(
        temporal_structure="regime_based",
        leakage_model="stage_frontier",
        target_type="binary",
        validation_strategy="time_split",
        recommendation_type="decision",
    )
    assert cap.target_type == "binary"
    assert cap.schema_version == SCHEMA_VERSION
    assert cap.framework_version == __version__


def test_capability_time_to_event_requires_temporal():
    with pytest.raises(ValidationError, match="temporal_structure"):
        CapabilityComposition(
            temporal_structure="none",
            leakage_model="none",
            target_type="time_to_event",
            validation_strategy="stratified",
            recommendation_type="decision",
        )


def test_capability_forecast_horizon_requires_temporal():
    with pytest.raises(ValidationError, match="temporal_structure"):
        CapabilityComposition(
            temporal_structure="none",
            leakage_model="forecast_horizon",
            target_type="continuous",
            validation_strategy="stratified",
            recommendation_type="forecast",
        )


def test_capability_multi_horizon_requires_forecast_horizon():
    with pytest.raises(ValidationError, match="leakage_model"):
        CapabilityComposition(
            temporal_structure="seasonal",
            leakage_model="none",
            target_type="multi_horizon",
            validation_strategy="rolling_origin",
            recommendation_type="forecast",
        )


def test_capability_rolling_origin_requires_temporal():
    with pytest.raises(ValidationError, match="temporal_structure"):
        CapabilityComposition(
            temporal_structure="none",
            leakage_model="none",
            target_type="continuous",
            validation_strategy="rolling_origin",
            recommendation_type="forecast",
        )


# --- Mission ------------------------------------------------------------


def _ok_capability_binary_temporal() -> CapabilityComposition:
    return CapabilityComposition(
        temporal_structure="regime_based",
        leakage_model="stage_frontier",
        target_type="binary",
        validation_strategy="time_split",
        recommendation_type="decision",
    )


def _ok_mission_binary_temporal() -> Mission:
    return Mission(
        project_name="demo",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        capability=_ok_capability_binary_temporal(),
        target_column="defect",
        time_column="batch_time",
        forbidden_columns=["downstream_qc"],
        allowed_columns=[],
        success_criterion=SuccessCriterion(
            metric="roc_auc",
            threshold=0.78,
            direction=">=",
        ),
        budget=MissionBudget(token_cap=200_000, iteration_cap=40),
        business_question="Which upstream features predict the defect downstream?",
    )


def test_mission_round_trip():
    m = _ok_mission_binary_temporal()
    j = m.model_dump_json()
    again = Mission.model_validate_json(j)
    assert again.project_name == "demo"
    assert again.capability.target_type == "binary"


def test_mission_temporal_requires_time_column():
    with pytest.raises(ValidationError, match="time_column"):
        Mission(
            project_name="demo",
            domain="manufacturing",
            capability=_ok_capability_binary_temporal(),
            target_column="defect",
            success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.78, direction=">="),
            budget=MissionBudget(token_cap=200_000),
            business_question="?",
        )


def test_mission_target_in_forbidden_rejected():
    with pytest.raises(ValidationError, match="target_column"):
        Mission(
            project_name="demo",
            domain="manufacturing",
            capability=_ok_capability_binary_temporal(),
            target_column="defect",
            time_column="ts",
            forbidden_columns=["defect"],
            success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.78, direction=">="),
            budget=MissionBudget(token_cap=100),
            business_question="?",
        )


def test_mission_group_kfold_requires_group_column():
    cap = CapabilityComposition(
        temporal_structure="none",
        leakage_model="none",
        target_type="binary",
        validation_strategy="group_kfold",
        recommendation_type="decision",
    )
    with pytest.raises(ValidationError, match="group_column"):
        Mission(
            project_name="demo",
            domain="general",
            capability=cap,
            target_column="y",
            success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
            budget=MissionBudget(token_cap=100),
            business_question="?",
        )


def test_mission_invalid_slug():
    with pytest.raises(ValidationError, match="alphanumeric"):
        Mission(
            project_name="bad name!",
            domain="general",
            capability=CapabilityComposition(
                temporal_structure="none",
                leakage_model="none",
                target_type="binary",
                validation_strategy="stratified",
                recommendation_type="decision",
            ),
            target_column="y",
            success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
            budget=MissionBudget(token_cap=100),
            business_question="?",
        )


def test_join_spec_min_one_key():
    with pytest.raises(ValidationError):
        JoinSpec(left_table="a", right_table="b", on=[], how="inner")


# --- Plan dict ----------------------------------------------------------


def test_plan_dict_id_must_start_with_p_dash():
    pe = PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="naive baseline")
    with pytest.raises(ValidationError, match="P-"):
        PlanDict(
            id="X-1-abc",
            iteration=1,
            hypothesis_id="H-seed-1",
            model="logreg",
            features=["+all_allowed"],
            prior_evidence=pe,
            technique_family="linear",
            area="baseline",
            expected_info_gain=0.5,
        )


def test_plan_dict_features_no_empty_token():
    pe = PriorEvidence(kind="hypothesis_seed", reference="H-seed-1", summary="x")
    with pytest.raises(ValidationError, match="empty/whitespace"):
        PlanDict(
            id="P-1-abc",
            iteration=1,
            hypothesis_id="H-seed-1",
            model="logreg",
            features=["a", " "],
            prior_evidence=pe,
            technique_family="linear",
            area="baseline",
            expected_info_gain=0.5,
        )


def test_plan_dict_round_trip():
    pe = PriorEvidence(
        kind="sketch_query",
        reference="top_interactions:abc123",
        summary="cols (X1,X3) have mutual_info=0.42, the highest pair",
    )
    p = PlanDict(
        id="P-3-deadbe",
        iteration=3,
        hypothesis_id="H-seed-4",
        model="lgbm_binary",
        features=["X1", "X3", "engineered:interactions_top5"],
        params={"num_leaves": 31},
        calibrate=True,
        prior_evidence=pe,
        technique_family="boosted_tree",
        area="interactions",
        expected_info_gain=0.7,
    )
    assert PlanDict.model_validate_json(p.model_dump_json()).id == "P-3-deadbe"


# --- ExperimentResult ---------------------------------------------------


def test_experiment_result_round_trip():
    er = ExperimentResult(
        id="P-1-abc",
        iteration=1,
        hypothesis_id="H-seed-1",
        model="logreg",
        features_used=["x1", "x2"],
        params={},
        calibrated=False,
        technique_family="linear",
        area="baseline",
        metrics=FitMetrics(
            train={"roc_auc": 0.71},
            validation={"roc_auc": 0.66},
        ),
        primary_metric="roc_auc",
        primary_metric_value=0.66,
        is_best_so_far=True,
        skeptic=SkepticResult(verdict="ACCEPT"),
    )
    again = ExperimentResult.model_validate_json(er.model_dump_json())
    assert again.primary_metric_value == 0.66


# --- Sketch -------------------------------------------------------------


def test_sketch_l1_round_trip():
    s = L1ColumnSummary(
        column="X1",
        dtype="numeric",
        n_total=1000,
        n_missing=3,
        n_unique_estimate=987,
        quantiles={"0.5": 0.4, "0.95": 0.9},
        mean=0.42,
        stdev=0.21,
    )
    again = L1ColumnSummary.model_validate_json(s.model_dump_json())
    assert again.column == "X1"


def test_sketch_manifest_round_trip():
    m = SketchManifest(
        project_name="demo",
        seed=42,
        n_rows_source=10_000,
        n_columns_source=120,
        capabilities=["tabular_classification"],
        l1_path="sketch/L1.parquet",
        l2_path="sketch/L2.json",
        l3_path="sketch/L3.json",
        l4_paths=["sketch/L4_tabular_classification.parquet"],
        l5_path="sketch/L5.json",
        l6_path="sketch/L6.json",
        l7_path="sketch/L7.jsonl",
        total_size_bytes=512_000,
    )
    again = SketchManifest.model_validate_json(m.model_dump_json())
    assert again.seed == 42


def test_sketch_annotation_min():
    a = SketchAnnotation(kind="regime_label", target_id="2", iteration=12, text="post-cleaning")
    assert a.author_role == "reviewer"


# --- Misc ---------------------------------------------------------------


def test_budget_ledger_entry():
    e = BudgetLedgerEntry(
        sequence=0,
        iteration=0,
        event="bootstrap",
        role="researcher",
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cumulative_total=120,
        cap=100_000,
        fraction_consumed=120 / 100_000,
    )
    again = BudgetLedgerEntry.model_validate_json(e.model_dump_json())
    assert again.sequence == 0


def test_question_round_trip():
    q = Question(
        question_id="Q-1-1",
        kind="confirm_inference",
        prompt="Is the target column `defect`?",
        inferred_answer="defect",
        confidence=0.85,
        impact="mission_field",
        target_mission_path="target_column",
    )
    QuestionBatch(batch_id="B-1", iteration=0, questions=[q])


def test_recommendation_no_signal():
    r = Recommendation(
        project_name="demo",
        recommendation_type="decision",
        decision="No actionable signal — collect more sensor data.",
        rationale="ROC-AUC plateaued at chance after 30 iterations.",
        evidence_chain=["P-30-abc"],
        confidence_tier="no_signal",
    )
    assert r.confidence_tier == "no_signal"


def test_knowledge_bundle_round_trip():
    h = HypothesisLibraryEntry(
        entry_id="K-h-1",
        source_project="demo",
        source_iteration=12,
        domain="manufacturing",
        capability_signature="abc",
        pattern_summary="regime-specific submodel beats global on regime-2",
        technique_family="boosted_tree",
        feature_roles=["<sensor:temperature>", "<process:flowrate>"],
        info_gain=0.8,
        primary_metric="roc_auc",
        primary_metric_delta=0.04,
    )
    f = FailureModeEntry(
        entry_id="K-f-1",
        source_project="demo",
        domain="manufacturing",
        capability_signature="abc",
        failure_name="leak_via_downstream_qc",
        resolution="added qc cols to forbidden_columns",
    )
    bundle = KnowledgeBundle(
        project_name="demo",
        domain="manufacturing",
        capability_signature="abc",
        hypothesis_entries=[h],
        failure_entries=[f],
        sketch_similarity_vector=[0.1, 0.2, 0.3],
    )
    again = KnowledgeBundle.model_validate_json(bundle.model_dump_json())
    assert len(again.hypothesis_entries) == 1


def test_project_meta_round_trip():
    pm = ProjectMeta(
        project_name="demo",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        branch="project/team/demo",
        framework_version_pin=__version__,
        token_budget=100_000,
    )
    assert pm.status == "created"
