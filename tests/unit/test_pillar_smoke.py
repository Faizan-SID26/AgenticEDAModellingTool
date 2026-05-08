"""Smoke tests for the multi-agent + breakthrough-mode upgrade.

Hermetic and fast — no LLM, no real /run, no large fixtures. The goal is
to prove every new module + every modified pathway loads and produces the
expected behavior on tiny inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# --- Pillar 1: operational_floor + breakthrough state -------------------


def test_pillar1_mission_budget_has_breakthrough_fields():
    from lib.schemas.mission import MissionBudget
    b = MissionBudget(token_cap=100_000)
    # Defaults preserve back-compat semantics for projects that don't set them.
    assert b.operational_floor is None
    assert b.stagnation_window == 12  # bumped 8→12
    assert b.breakthrough_stagnation_window == 20
    assert b.breakthrough_max_entries == 3


def test_pillar1_run_state_has_breakthrough_fields_and_round_trips():
    from lib.state import RunState
    s = RunState(project_name="t")
    assert s.breakthrough_mode_active is False
    assert s.iterations_in_breakthrough == 0
    assert s.breakthrough_started_at_iteration is None
    assert s.breakthrough_entry_count == 0
    s.breakthrough_mode_active = True
    s.iterations_in_breakthrough = 7
    s.breakthrough_entry_count = 1
    d = s.to_dict()
    s2 = RunState.from_dict(d)
    assert s2.breakthrough_mode_active is True
    assert s2.iterations_in_breakthrough == 7
    assert s2.breakthrough_entry_count == 1


def test_pillar1_run_state_drops_unknown_keys():
    """Forward-compatibility: a state file with extra keys (older or newer
    framework version) loads without raising."""
    from lib.state import RunState
    s = RunState.from_dict({
        "project_name": "t",
        "best_primary_metric_value": None,
        "future_field_we_havent_added_yet": "extra",
    })
    assert s.project_name == "t"


# --- Pillar 8: identifier regex -----------------------------------------


def test_pillar8_id_regex_catches_generics():
    from lib.inspect import _matches_id_pattern as m
    positives = [
        "SerialNumber", "Serial Number", "serial_int_x",
        "sku_run_idx", "recipe_run_idx", "SPRAY PART ID", "DRY PART ID",
        "POD_Number_Serial", "user_id", "UUID", "order_code", "batch_lot",
        "LotNumber", "token", "session_hash", "customer_idx", "part_index",
        "event_2024061500001", "customer-id", "HashCode",
    ]
    negatives = [
        "temperature", "pressure", "value", "score", "target",
        "feature_1", "model", "timestamp", "created_at", "spray_temp",
        "event_at",
    ]
    for n in positives:
        assert m(n), f"should match: {n}"
    for n in negatives:
        assert not m(n), f"should NOT match: {n}"


# --- Pillar 7a: top-K + ECE ----------------------------------------------


def test_pillar7a_recall_at_top_pct_handcomputed():
    """Recall@top-20% on a 10-row fixture: 4 positives, 2 in the top 20%
    by score → recall = 2/4 = 0.5."""
    from lib.eval import _FN
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    yp = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.6, 0.2, 0.1, 0.05, 0.0])
    assert _FN["recall_at_top_pct_20"](y, yp) == pytest.approx(0.5)
    assert _FN["precision_at_top_pct_20"](y, yp) == pytest.approx(1.0)
    assert _FN["lift_at_top_pct_20"](y, yp) == pytest.approx(1.0 / 0.4)


def test_pillar7a_ece_returns_finite_in_zero_one():
    """ECE returns a finite number in [0, 1] for valid probabilities and
    NaN when input is not probability-shaped (max > 1)."""
    from lib.eval import _FN
    y = np.array([1, 0, 1, 0, 1])
    yp = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    val = _FN["ece"](y, yp)
    assert 0.0 <= val <= 1.0
    # Out-of-range probabilities → NaN.
    yp_bad = np.array([1.5, -0.1, 0.5, 0.5, 0.5])
    assert np.isnan(_FN["ece"](y, yp_bad))


def test_pillar7a_top_k_in_dispatch_for_classification():
    from lib.eval import _BY_CAP
    cls = _BY_CAP["tabular_classification"]
    assert "recall_at_top_pct_10" in cls
    assert "expected_calibration_error" in cls


# --- Pillar 2: agent_inbox + anti_doom + PlanDict validator -------------


def test_pillar2_agent_inbox_round_trip(tmp_path: Path):
    from lib.agent_inbox import write, read, exists, list_messages
    write(tmp_path, 7, "researcher_proposal", {"plan": "x", "iteration": 7})
    assert exists(tmp_path, 7, "researcher_proposal")
    assert read(tmp_path, 7, "researcher_proposal")["iteration"] == 7
    assert list_messages(tmp_path, 7) == ["researcher_proposal"]


def test_pillar2_anti_doom_disk(tmp_path: Path):
    from lib.anti_doom import append_fingerprint, load_recent_fingerprints
    for i, fp in enumerate(["a", "a", "a"]):
        append_fingerprint(tmp_path, fp, iteration=i)
    fps = load_recent_fingerprints(tmp_path, window=3)
    assert fps == ["a", "a", "a"]


def _base_plan_payload(**overrides):
    p = {
        "id": "P-1-abc",
        "iteration": 1,
        "hypothesis_id": "H-seed-1",
        "model": "lgbm_binary",
        "features": ["+all_allowed"],
        "params": {},
        "calibrate": False,
        "prior_evidence": {
            "kind": "sketch_query",
            "reference": "top_interactions(top_k=5)",
            "summary": "x",
        },
        "technique_family": "boosted_tree",
        "area": "baseline",
        "expected_info_gain": 0.5,
    }
    p.update(overrides)
    return p


def test_pillar2_plan_validator_no_context_is_lenient():
    from lib.schemas.plan import PlanDict
    PlanDict.model_validate(_base_plan_payload())  # no exception


def test_pillar2_plan_validator_breakthrough_requires_domain_prior_url():
    from lib.schemas.plan import PlanDict
    from pydantic import ValidationError

    # Wrong kind.
    with pytest.raises(ValidationError):
        PlanDict.model_validate(
            _base_plan_payload(),
            context={"breakthrough_mode_active": True},
        )
    # Right kind, bad reference.
    p = _base_plan_payload(prior_evidence={
        "kind": "domain_prior",
        "reference": "not-a-url",
        "summary": "x",
    })
    with pytest.raises(ValidationError):
        PlanDict.model_validate(p, context={"breakthrough_mode_active": True})
    # Right kind, arxiv URL.
    p = _base_plan_payload(prior_evidence={
        "kind": "domain_prior",
        "reference": "https://arxiv.org/abs/2305.12345",
        "summary": "focal loss for imbalanced classification",
    })
    PlanDict.model_validate(p, context={"breakthrough_mode_active": True})  # ok


def test_pillar2_plan_validator_rejects_doomed_fingerprint_collision():
    from lib.schemas.plan import PlanDict
    from pydantic import ValidationError
    p = PlanDict.model_validate(_base_plan_payload())
    fp = p.fingerprint()
    with pytest.raises(ValidationError):
        PlanDict.model_validate(
            _base_plan_payload(),
            context={"recent_fingerprints": [fp]},
        )
    # Different model → different fingerprint → accepted.
    PlanDict.model_validate(
        _base_plan_payload(model="logreg"),
        context={"recent_fingerprints": [fp]},
    )


# --- Pillar 4: registry expansion ---------------------------------------


def test_pillar4_registry_has_breakthrough_keys():
    from lib.registry import _MODELS
    expected = {
        "elasticnet", "logreg_l1", "ridge_classifier",
        "decision_tree", "extra_trees", "random_forest",
        "stacked_blend", "voting_soft", "bagging",
        "mlp_tabular", "lgbm_focal", "lgbm_weighted",
        "autoencoder_anomaly",
    }
    missing = expected - set(_MODELS.keys())
    assert not missing, f"missing keys: {missing}"


def test_pillar4_is_available_filters_by_capability():
    from lib.registry import is_available
    assert is_available("lgbm_binary", "tabular_classification") is True
    assert is_available("prophet", "tabular_classification") is False
    # forecasting capability accepts neither logreg nor lgbm_focal.
    assert is_available("logreg", "forecasting") is False


def test_pillar4_unknown_model_key_is_unavailable():
    from lib.registry import is_available
    assert is_available("nonexistent_model", "tabular_classification") is False


# --- Pillar 5: feature DSL ----------------------------------------------


def _toy_df():
    import pandas as pd
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=50),
        "b": rng.normal(size=50),
        "hour": np.arange(50) % 24,
        "category": ["x", "y", "z"] * 16 + ["x", "y"],
        "target": rng.integers(0, 2, 50).astype(int),
    })


def test_pillar5_cyclic_emits_sin_cos():
    from lib.features import _engineered_group
    df = _toy_df()
    out, cols = _engineered_group(df, "cyclic", forbidden_for_engineering={"target"})
    assert "X__hour_sin" in out.columns and "X__hour_cos" in out.columns


def test_pillar5_target_encoding_emits_placeholders():
    from lib.features import _engineered_group
    df = _toy_df()
    out, cols = _engineered_group(df, "target_encoding", forbidden_for_engineering={"target"})
    assert any("te_PLACEHOLDER" in c for c in cols)


def test_pillar5_binning_quantile_and_polynomial_3():
    from lib.features import _engineered_group
    df = _toy_df()
    out_b, cols_b = _engineered_group(df, "binning_quantile_5", forbidden_for_engineering={"target"})
    assert any(c.endswith("_qbin5") for c in cols_b)
    out_p, cols_p = _engineered_group(df, "polynomial_3", forbidden_for_engineering={"target"})
    assert any(c.endswith("_cube") for c in cols_p)


def test_pillar5_interactions_top_parametric():
    from lib.features import _engineered_group
    df = _toy_df()
    sketch = [{"col_a": "a", "col_b": "b", "mutual_info": 0.4}]
    out, cols = _engineered_group(df, "interactions_top10",
                                  sketch_top_interactions=sketch,
                                  forbidden_for_engineering={"target"})
    assert "X__a_x_b" in out.columns


# --- Pillar 3: wildcard + cross-project hydration ------------------------


def test_pillar3_family_keys_cover_all_bandit_families():
    from lib.schemas.plan import TechniqueFamily
    from lib.generate_hypotheses import _FAMILY_TO_KEYS
    for fam in TechniqueFamily.__args__:
        assert fam in _FAMILY_TO_KEYS, fam


def test_pillar3_wildcard_with_concrete_model_key():
    from lib.generate_hypotheses import _wildcard_hypothesis
    h = _wildcard_hypothesis("boosted_tree", iteration=11, model_key="xgboost_binary",
                             rationale_extra="Breakthrough.")
    assert h["model_hint"] == "xgboost_binary"
    assert "wild" in h["hypothesis_id"]
    assert h["source"] == "generator_wildcard"


def test_pillar3_hypothesis_library_entry_carries_dsl_and_params():
    from lib.schemas.knowledge import HypothesisLibraryEntry
    e = HypothesisLibraryEntry(
        entry_id="K-h-x", source_project="P", source_iteration=5, domain="d",
        capability_signature="X", pattern_summary="p", technique_family="t",
        info_gain=0.3, primary_metric="m", primary_metric_delta=0.1,
        model="lgbm_focal", feature_dsl=["+all_allowed"], params={"alpha": 0.25},
    )
    assert e.model == "lgbm_focal"
    assert e.feature_dsl == ["+all_allowed"]
    assert e.params == {"alpha": 0.25}


# --- Pillar 9: reviewer prose parser ------------------------------------


def test_pillar9_reviewer_directives_parse_and_persist(tmp_path: Path):
    from lib.synthesize import parse_and_persist_reviewer_notes
    prose = """## What is working
LGBM looks fine.

## What to try next
- area=causal family=linear: try L1 logreg restricted to causal neighbors
- try: stacked_blend: ensemble might help
- try: feature engineered:cyclic: hour-based cyclic
- free-form prose without parseable structure
"""
    parsed = parse_and_persist_reviewer_notes(tmp_path, 10, prose)
    assert len(parsed) == 4
    kinds = [d["kind"] for d in parsed]
    assert kinds == ["area_family", "model", "feature", "free"]
    # The colon-separated trailing rationale was stripped from the feature token.
    feat = next(d for d in parsed if d["kind"] == "feature")
    assert "engineered:cyclic" in feat["features_dsl"]
    assert "engineered:cyclic:" not in feat["features_dsl"]
    # File is appended-to.
    rows = (tmp_path / "memory" / "HYPOTHESES.jsonl").read_text().splitlines()
    assert len(rows) == 4


def test_pillar9_mark_directive_consumed_flips_consumed_flag(tmp_path: Path):
    from lib.synthesize import parse_and_persist_reviewer_notes, mark_directive_consumed
    prose = "## What to try next\n- try: stacked_blend: ensemble"
    parsed = parse_and_persist_reviewer_notes(tmp_path, 10, prose)
    rows = [json.loads(l) for l in (tmp_path / "memory" / "HYPOTHESES.jsonl").read_text().splitlines()]
    hid = rows[0]["hypothesis_id"]
    assert rows[0]["consumed"] is False
    assert mark_directive_consumed(tmp_path, hid) is True
    rows2 = [json.loads(l) for l in (tmp_path / "memory" / "HYPOTHESES.jsonl").read_text().splitlines()]
    assert rows2[0]["consumed"] is True


# --- Pillar 6: web_search query helpers ---------------------------------


def test_pillar6_build_sota_queries_capability_aware():
    from lib.web_search import build_sota_queries
    qs = build_sota_queries("tabular_classification", "boosted_tree", "imbalanced", year_min=2023, k=3)
    assert len(qs) == 3
    assert any("imbalanced classification" in q.lower() for q in qs)


def test_pillar6_parse_search_hits_normalizes_shape():
    from lib.web_search import parse_search_hits, shortlist_hits
    raw = [
        {"url": "https://arxiv.org/abs/2305.12345", "title": "Focal Loss",
         "snippet": "2024 paper", "score": 0.9},
        {"url": "https://example.com", "title": "blog", "snippet": "2020", "score": 0.1},
    ]
    hits = parse_search_hits(raw)
    assert len(hits) == 2
    short = shortlist_hits(hits, k=1, require_arxiv=True)
    assert len(short) == 1 and "arxiv.org" in short[0].url


# --- Pillar 7b: population evaluator ------------------------------------


def _toy_mission(operational_floor: "float | None" = None):
    from lib.schemas.mission import (
        Mission, MissionBudget, SuccessCriterion, CapabilityComposition,
    )
    return Mission(
        project_name="t",
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
        budget=MissionBudget(token_cap=1_000_000, operational_floor=operational_floor),
        business_question="Test.",
    )


def test_pillar7b_evaluate_at_population_no_joined_returns_empty(tmp_path: Path):
    """When the joined parquet doesn't exist, evaluate_at_population returns
    {} rather than raising — used in tests and bootstrap-incomplete states."""
    from lib.eval_population import evaluate_at_population, cache_clear
    cache_clear()
    out = evaluate_at_population(
        model=None, plan=None, mission=_toy_mission(),  # type: ignore[arg-type]
        project_dir=tmp_path,
        capability_key="tabular_classification",
    )
    assert out == {}


# --- Pillar 10: finalize blocks below floor -----------------------------


def test_pillar10_should_block_finalize_below_floor(tmp_path: Path):
    """When tier='low' AND best below operational_floor AND budget remains,
    `_should_block_finalize` returns block=True with the expected reason."""
    from lib.finalize import _should_block_finalize
    from lib.schemas.recommendation import Recommendation
    from lib.state import RunState, save_run_state

    state = RunState(project_name="t")
    state.best_primary_metric_value = 0.30
    state.current_iteration = 5
    state.breakthrough_entry_count = 0
    save_run_state(tmp_path, state)

    mission = _toy_mission(operational_floor=0.50)
    rec = Recommendation(
        project_name="t",
        recommendation_type="decision",
        decision="x",
        rationale="y",
        evidence_chain=[],
        causal_assumptions=[],
        ruled_out_failure_modes=[],
        what_would_change_it=[],
        model_card=[],
        confidence_tier="low",
    )
    info = _should_block_finalize(tmp_path, mission, rec)
    assert info["block"] is True
    assert info["reason"] == "below_operational_floor_with_budget_remaining"


# --- Pillar 11: agent files exist with correct frontmatter ---------------


def test_pillar11_agent_files_present():
    repo = Path(__file__).resolve().parents[2]
    agents_dir = repo / ".claude" / "agents"
    expected = {"planner", "researcher", "runner", "reviewer", "analyst",
                "literature", "novelty-check", "skeptic", "debate-arbiter"}
    present = {p.stem for p in agents_dir.glob("*.md")}
    missing = expected - present
    assert not missing, f"missing agent files: {missing}"


def test_pillar11_agent_frontmatter_has_model_and_tools():
    repo = Path(__file__).resolve().parents[2]
    for name in ["literature", "novelty-check", "skeptic", "debate-arbiter"]:
        text = (repo / ".claude" / "agents" / f"{name}.md").read_text(encoding="utf-8")
        assert "name:" in text and f"name: {name}" in text
        assert "model:" in text
        assert "allowed-tools:" in text


# --- Pillar 13: orchestrator skill description updated ------------------


def test_pillar13_run_md_describes_multi_agent_dispatch():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / ".claude" / "commands" / "run.md").read_text(encoding="utf-8")
    assert "Multi-agent" in text or "multi-agent" in text
    assert "novelty-check" in text
    assert "literature" in text
    assert "debate-arbiter" in text
