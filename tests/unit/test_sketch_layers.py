"""Per-layer correctness tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.sketch.l1_distributions import build_l1, summarize_column
from lib.sketch.l2_joint import build_l2
from lib.sketch.l3_regimes import build_l3
from lib.sketch.l4_coresets import build_coreset
from lib.sketch.l5_timeseries import build_l5_for_column
from lib.sketch.l6_causal import build_l6
from lib.sketch.l7_failure_modes import empty_catalog, match_or_create


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    # Two regimes: variance shift in feature x_temp midway.
    x_temp = np.concatenate([rng.normal(100, 1.0, n // 2), rng.normal(105, 3.0, n - n // 2)])
    x_press = rng.normal(1.0, 0.05, n)
    cat = rng.choice(["A", "B", "C"], n)
    y = (x_temp > 102.5).astype(int)
    return pd.DataFrame(
        {"t": t, "x_temp": x_temp, "x_press": x_press, "cat": cat, "y": y}
    )


def test_l1_numeric_column(synthetic_df):
    s = summarize_column(synthetic_df["x_temp"])
    assert s.dtype == "numeric"
    assert "0.50" in s.quantiles
    assert s.mean is not None and s.stdev is not None
    assert s.n_total == len(synthetic_df)


def test_l1_categorical_column(synthetic_df):
    s = summarize_column(synthetic_df["cat"])
    assert s.dtype == "categorical"
    assert s.top_categories
    assert sum(c for _, c in s.top_categories) <= len(synthetic_df)


def test_l1_full(synthetic_df):
    summaries = build_l1(synthetic_df)
    by_name = {s.column: s for s in summaries}
    assert "y" in by_name and "x_temp" in by_name


def test_l2_top_interactions(synthetic_df):
    l2 = build_l2(synthetic_df, target="y", top_k_interactions=5, seed=0)
    assert l2.n_components >= 1
    assert l2.explained_variance_ratio
    assert any(it["col_a"] == "x_temp" or it["col_b"] == "x_temp" for it in l2.top_interactions)


def test_l3_detects_regime(synthetic_df):
    l3 = build_l3(synthetic_df, time_column="t", target="x_temp")
    # 2 regimes is the truth; we accept >=2 (PELT may over-segment slightly).
    assert l3.n_regimes >= 2


def test_l4_coreset_size(synthetic_df):
    coreset, summary = build_coreset(
        synthetic_df,
        capability_key="tabular_classification",
        target="y",
        n_rows=200,
        seed=0,
    )
    assert len(coreset) == 200
    assert "weight" in coreset.columns
    assert summary.n_rows == 200
    assert summary.weight_l2_norm > 0


def test_l5_for_column(synthetic_df):
    s = build_l5_for_column(synthetic_df["x_temp"], window=64, word_length=8, alphabet_size=8)
    assert s.column == "x_temp"
    assert s.matrix_profile_window == 64


def test_l6_returns_a_graph(synthetic_df):
    l6 = build_l6(synthetic_df, alpha=0.1, max_cond_set=1, max_columns=10, seed=0)
    assert l6.alpha == 0.1
    assert l6.nodes  # at least the numeric columns


def test_l7_match_or_create():
    catalog = empty_catalog()
    point = {"a": 1.0, "b": 2.0}
    catalog, cid1, created1 = match_or_create(catalog, point, iteration=1)
    assert created1 is True
    catalog, cid2, created2 = match_or_create(catalog, {"a": 1.05, "b": 2.05}, iteration=2)
    # Should match the first cluster (small distance).
    assert cid2 == cid1
    assert created2 is False
    catalog, cid3, created3 = match_or_create(catalog, {"a": 100.0, "b": -100.0}, iteration=3)
    assert created3 is True
    assert cid3 != cid1
