"""Integration test for /init.

Creates a synthetic project under a tmp workspace, drops two synthetic
parquet files into data/, runs `lib.inspect.inspect_project`, and asserts:
    - INIT_PROFILE.json is well-formed.
    - results/init_report.md is non-empty.
    - Likely target / time / id columns are detected.
    - A join is proposed between tables sharing a key.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.inspect import inspect_project
from lib.project import create_project


@pytest.fixture()
def synthetic_project(tmp_workspace: Path) -> Path:
    """Build a project with synthetic process + qa parquet files."""
    proj = create_project(
        name="syn_init",
        domain="manufacturing",
        recipe="manufacturing_defect_classification",
        token_budget=10_000,
        workspace=tmp_workspace,
    )
    rng = np.random.default_rng(0)
    n = 1000
    process = pd.DataFrame(
        {
            "batch_id": np.arange(n),
            "batch_time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "reactor_temp": rng.normal(100, 5, n),
            "reactor_pressure": rng.normal(1.0, 0.05, n),
            "raw_grade": rng.choice(["A", "B", "C"], n),
        }
    )
    qa = pd.DataFrame(
        {
            "batch_id": np.arange(n),
            "downstream_qc_score": rng.normal(0, 1, n),
            "defect": rng.binomial(1, 0.1, n),
        }
    )
    process.to_parquet(proj / "data" / "process.parquet")
    qa.to_parquet(proj / "data" / "qa.parquet")
    return proj


def test_init_profile_well_formed(synthetic_project: Path):
    profile = inspect_project(synthetic_project)
    assert profile["n_files"] == 2
    files = {f["table_name"]: f for f in profile["files"]}
    assert "process" in files and "qa" in files
    # Likely target should include 'defect'.
    assert "defect" in files["qa"]["likely_target"]
    # Likely time should include 'batch_time'.
    assert "batch_time" in files["process"]["likely_time"]
    # Likely id should include 'batch_id'.
    assert any("batch_id" in f["likely_id"] for f in files.values())


def test_init_proposes_join_on_shared_key(synthetic_project: Path):
    profile = inspect_project(synthetic_project)
    joins = profile["proposed_joins"]
    assert any(
        set(j["on"]) == {"batch_id"}
        and {j["left_table"], j["right_table"]} == {"process", "qa"}
        for j in joins
    )


def test_init_writes_report_and_profile(synthetic_project: Path):
    inspect_project(synthetic_project)
    assert (synthetic_project / "memory" / "INIT_PROFILE.json").exists()
    report = (synthetic_project / "results" / "init_report.md").read_text(encoding="utf-8")
    assert "Files" in report
    assert "Proposed joins" in report
