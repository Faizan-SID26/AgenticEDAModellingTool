"""Determinism + size budget for sketch builds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.project import create_project
from lib.schemas.mission import (
    CapabilityComposition,
    Mission,
    MissionBudget,
    SuccessCriterion,
)
from lib.sketch.builder import build_sketch


def _mission(name: str) -> Mission:
    return Mission(
        project_name=name,
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
        success_criterion=SuccessCriterion(metric="roc_auc", threshold=0.7, direction=">="),
        budget=MissionBudget(token_cap=10_000),
        business_question="Test.",
    )


def _df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(2, 0.5, n),
            "x3": rng.normal(-1, 0.3, n),
            "y": rng.binomial(1, 0.3, n),
        }
    )


def _normalize_manifest(d: dict) -> dict:
    """Drop the timestamp; everything else is deterministic."""
    d = dict(d)
    d.pop("created_at", None)
    return d


def test_build_is_deterministic(tmp_workspace: Path):
    proj_a = create_project(name="d_a", domain="manufacturing", token_budget=1, workspace=tmp_workspace)
    proj_b = create_project(name="d_b", domain="manufacturing", token_budget=1, workspace=tmp_workspace)
    df = _df()
    m_a = build_sketch(proj_a, df, _mission("d_a"), capability_keys=["temporal_classification"], seed=42)
    m_b = build_sketch(proj_b, df, _mission("d_b"), capability_keys=["temporal_classification"], seed=42)
    a = _normalize_manifest(m_a.model_dump())
    b = _normalize_manifest(m_b.model_dump())
    # project_name + paths differ; everything else should match.
    a.pop("project_name"); b.pop("project_name")
    a["l4_paths"] = [p.replace("d_a", "X") for p in a["l4_paths"]]
    b["l4_paths"] = [p.replace("d_b", "X") for p in b["l4_paths"]]
    assert a == b


def test_size_under_one_mb(tmp_workspace: Path):
    proj = create_project(name="sz", domain="manufacturing", token_budget=1, workspace=tmp_workspace)
    df = _df(n=5000)
    m = build_sketch(proj, df, _mission("sz"), capability_keys=["temporal_classification"], seed=0)
    # The L4 coreset parquet is ~150KB for 5k rows; everything else is JSON.
    # Budget: <1MB for L1+L2+L3+L5+L6+L7 (excluding L4 coresets).
    structural = (proj / "sketch" / "L1.json").stat().st_size
    structural += (proj / "sketch" / "L2.json").stat().st_size
    structural += (proj / "sketch" / "L3.json").stat().st_size
    structural += (proj / "sketch" / "L5.json").stat().st_size
    structural += (proj / "sketch" / "L6.json").stat().st_size
    structural += (proj / "sketch" / "L7.jsonl").stat().st_size
    assert structural < 1_000_000, f"structural sketch is {structural} bytes"
