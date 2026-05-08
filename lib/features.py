"""Feature DSL expansion.

Plan dicts express features as a list of tokens. Tokens are:

- `+all_allowed` — expanded to MISSION.allowed_columns or "everything not
  in MISSION.forbidden_columns".
- `+lag_downstream` — include lagged values of immediately-downstream
  columns honoring MISSION's join-plan `lag_policy` (capability-agnostic;
  uses `_table_origin_map` written at bootstrap).
- `engineered:<GROUP>` — looked up in the engineered-feature catalog. The
  catalog now includes:
    * `interactions_top5`, `interactions_top<K>` (parametric, K ≤ 25)
    * `auto_l2`           — top-N pairs ranked by L2 mutual-info (default 10)
    * `ratios`            — pairwise division of top-N numerics
    * `polynomial_2`, `polynomial_3` — squares and cubes
    * `lag_<N>`           — shift-by-N
    * `lag_x_ratio`       — composition `(a/b).shift(k)` for top numeric pairs
    * `cyclic`            — sin/cos pairs for periodic-named/-flagged numerics
    * `frequency_encoding` — replace categoricals by their dense-rank value count
    * `binning_quantile_<k>` — quantile bins of numerics (default k=5)
    * `target_encoding`    — leave-one-out smoothed encoding for categoricals
                             (per-fold; runner wires this token specially)
    * `autoencoded`        — reconstruction error + bottleneck activations
                             (gated on breakthrough_mode_active)
- `sketch:top3_univariate` — pick the top-3 numeric columns by L2 mutual info.
- bare column name — used verbatim.

`expand_features(...)` returns the *concrete* column list a model will
see, with engineered columns materialized in the returned DataFrame.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from lib.schemas.mission import Mission

_log = logging.getLogger("eda.features")


_CYCLIC_NAME_RX = re.compile(
    r"(hour|minute|second|dayofweek|day_of_week|dow|month|doy|day_of_year|week|theta|angle|phase|degree)",
    flags=re.IGNORECASE,
)
"""Generic periodicity name patterns. Capability-agnostic — any column whose
name matches gets a sin/cos pair when `engineered:cyclic` fires."""

_CYCLIC_PERIODS_BY_HINT: dict[str, float] = {
    "hour": 24.0,
    "minute": 60.0,
    "second": 60.0,
    "dayofweek": 7.0,
    "day_of_week": 7.0,
    "dow": 7.0,
    "month": 12.0,
    "doy": 365.25,
    "day_of_year": 365.25,
    "week": 52.0,
    "theta": 360.0,
    "angle": 360.0,
    "phase": 360.0,
    "degree": 360.0,
}

_INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {
        # The L4 coreset adds a per-row sample weight as `weight`. It is a
        # framework-internal column, never a feature.
        "weight",
        "_weight",
        "__weight__",
        # Reserved engineered-feature prefix (X__a_x_b, X__a_div_b, ...);
        # we filter the prefix below, but include the bare token defensively.
        "X__",
    }
)


def _is_internal_column(name: str) -> bool:
    """Return True if the column is a framework-internal artifact and
    must not be treated as a feature."""
    if name in _INTERNAL_COLUMNS:
        return True
    if name.startswith("X__"):
        # Engineered features added during DSL expansion are re-emitted by
        # `engineered:<group>` tokens, not by `+all_allowed`.
        return True
    return False


def _resolve_all_allowed(mission: Mission, df_columns: Iterable[str]) -> list[str]:
    if mission.allowed_columns:
        return [c for c in mission.allowed_columns if not _is_internal_column(c)]
    forbidden = set(mission.forbidden_columns)
    forbidden.add(mission.target_column)
    if mission.time_column:
        forbidden.add(mission.time_column)
    if mission.group_column:
        forbidden.add(mission.group_column)
    return [c for c in df_columns if c not in forbidden and not _is_internal_column(c)]


def _add_top_interactions(df: pd.DataFrame, top_pairs: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    """Materialize engineered interaction features.

    For numeric pairs: produce the product column. For mixed pairs: skip.
    """
    new_cols: list[str] = []
    out = df.copy()
    for a, b in top_pairs:
        if a not in out.columns or b not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[a]) or not pd.api.types.is_numeric_dtype(out[b]):
            continue
        name = f"X__{a}_x_{b}"
        out[name] = out[a] * out[b]
        new_cols.append(name)
    return out, new_cols


def _engineered_group(
    df: pd.DataFrame,
    group: str,
    *,
    sketch_top_interactions: list[dict] | None = None,
    forbidden_for_engineering: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Materialize an engineered-group's columns. Returns (df_with_new, new_col_names).

    `forbidden_for_engineering` is a defense-in-depth filter: even if a
    bad pair somehow survived the L2 build, we refuse to materialize an
    interaction column that involves the target or any forbidden column.
    """
    forbidden = forbidden_for_engineering or set()
    if group == "interactions_top5" or group.startswith("interactions_top"):
        # Parametric: `interactions_top<K>` for K up to 25. Falls back to top-5
        # for the legacy alias `interactions_top5`.
        if group == "interactions_top5":
            top_k = 5
        else:
            try:
                top_k = max(1, min(25, int(group[len("interactions_top"):])))
            except ValueError:
                top_k = 5
        pairs: list[tuple[str, str]] = []
        for it in (sketch_top_interactions or [])[:top_k]:
            a = it.get("col_a")
            b = it.get("col_b")
            if not a or not b:
                continue
            if a in forbidden or b in forbidden or a == b:
                _log.warning(
                    "skipping leaky/degenerate interaction (%s, %s) at expand time",
                    a,
                    b,
                )
                continue
            pairs.append((a, b))
        return _add_top_interactions(df, pairs)
    if group == "auto_l2":
        # Same as interactions_top10 — explicit alias signaling
        # "use whatever the L2 sketch ranks highest, top 10".
        pairs = []
        for it in (sketch_top_interactions or [])[:10]:
            a = it.get("col_a")
            b = it.get("col_b")
            if a and b and a not in forbidden and b not in forbidden and a != b:
                pairs.append((a, b))
        return _add_top_interactions(df, pairs)
    if group == "ratios":
        # Ratios between every pair of positive numeric columns. Bounded.
        num = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        out = df.copy()
        new_cols: list[str] = []
        for i, a in enumerate(num):
            for b in num[i + 1 :]:
                col = f"X__{a}_div_{b}"
                denom = out[b].replace(0, np.nan)
                out[col] = out[a] / denom
                new_cols.append(col)
        return out, new_cols
    if group == "polynomial_2":
        num = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        out = df.copy()
        new_cols: list[str] = []
        for c in num:
            col = f"X__{c}_sq"
            out[col] = out[c] ** 2
            new_cols.append(col)
        return out, new_cols
    if group == "polynomial_3":
        num = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        out = df.copy()
        new_cols = []
        for c in num:
            col = f"X__{c}_cube"
            out[col] = out[c] ** 3
            new_cols.append(col)
        return out, new_cols
    if group.startswith("lag_x_ratio"):
        # Composition: lag of pairwise ratios. Uses the L2-ranked top pairs
        # if available; falls back to top numerics.
        try:
            lag = int(group.split("_")[-1]) if any(ch.isdigit() for ch in group) else 1
        except ValueError:
            lag = 1
        pairs = []
        if sketch_top_interactions:
            for it in sketch_top_interactions[:5]:
                a, b = it.get("col_a"), it.get("col_b")
                if a and b and a in df.columns and b in df.columns and a != b:
                    if a in forbidden or b in forbidden:
                        continue
                    pairs.append((a, b))
        if not pairs:
            num = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
            for i, a in enumerate(num):
                for b in num[i + 1:]:
                    pairs.append((a, b))
        out = df.copy()
        new_cols = []
        for a, b in pairs[:6]:
            denom = out[b].replace(0, np.nan)
            name = f"X__{a}_div_{b}_lag{lag}"
            out[name] = (out[a] / denom).shift(lag)
            new_cols.append(name)
        return out, new_cols
    if group.startswith("lag_"):
        try:
            lag = int(group.split("_", 1)[1])
        except ValueError:
            lag = 1
        num = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        out = df.copy()
        new_cols = []
        for c in num:
            name = f"X__{c}_lag{lag}"
            out[name] = out[c].shift(lag)
            new_cols.append(name)
        return out, new_cols
    if group == "cyclic":
        # sin/cos pairs for any numeric column whose name matches a generic
        # periodicity hint. Period inferred from the matched hint; falls
        # back to (max - min) of the column when unmatched.
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        out = df.copy()
        new_cols: list[str] = []
        for c in num:
            if c in forbidden:
                continue
            m = _CYCLIC_NAME_RX.search(str(c))
            if not m:
                continue
            hint = m.group(1).lower()
            period = _CYCLIC_PERIODS_BY_HINT.get(hint)
            if period is None:
                rng = float(out[c].max() - out[c].min())
                period = rng if rng > 1e-9 else 1.0
            theta = 2.0 * np.pi * out[c].astype(float) / period
            sin_name = f"X__{c}_sin"
            cos_name = f"X__{c}_cos"
            out[sin_name] = np.sin(theta)
            out[cos_name] = np.cos(theta)
            new_cols.extend([sin_name, cos_name])
        return out, new_cols
    if group == "frequency_encoding":
        # Replace each categorical/object column by its dense-rank value
        # count, normalized to (0, 1]. Scales gracefully across cardinalities.
        cat_cols = [
            c for c in df.columns
            if c not in forbidden and not _is_internal_column(c)
            and (pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object)
        ]
        out = df.copy()
        new_cols = []
        n = max(1, len(out))
        for c in cat_cols:
            vc = out[c].value_counts(dropna=False)
            mapping = (vc.rank(method="dense") / n).to_dict()
            name = f"X__{c}_freq"
            out[name] = out[c].map(mapping).astype(float)
            new_cols.append(name)
        return out, new_cols
    if group.startswith("binning_quantile_"):
        try:
            k = max(2, min(20, int(group.split("_")[-1])))
        except ValueError:
            k = 5
        num = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        out = df.copy()
        new_cols = []
        for c in num:
            if c in forbidden:
                continue
            try:
                binned = pd.qcut(out[c], q=k, labels=False, duplicates="drop")
            except Exception:  # noqa: BLE001
                continue
            name = f"X__{c}_qbin{k}"
            out[name] = binned.astype(float)
            new_cols.append(name)
        return out, new_cols
    if group == "target_encoding":
        # Per-fold leave-one-out target encoding requires CV-loop access to
        # the training rows only — it cannot be materialized here without
        # leakage. The runner's CV loop short-circuits this token: when it
        # sees a `target_encoding` placeholder column it re-fits the encoder
        # from training-fold rows only. Here we emit a sentinel column that
        # the runner will recognize and overwrite per-fold.
        cat_cols = [
            c for c in df.columns
            if c not in forbidden and not _is_internal_column(c)
            and (pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object)
        ][:10]
        out = df.copy()
        new_cols = []
        for c in cat_cols:
            name = f"X__{c}_te_PLACEHOLDER"
            # Filled with the global mean of the (encoded) target as a safe
            # default; replaced by the runner per-fold to avoid leakage.
            out[name] = float("nan")
            new_cols.append(name)
        return out, new_cols
    if group == "autoencoded":
        # Reconstruction-error + bottleneck activations from a small MLP
        # autoencoder. Trained here on the entire DataFrame's numerics —
        # this is acceptable because the autoencoder is unsupervised
        # (target_column is excluded). Keep training cheap so it's
        # affordable to fire each iteration when breakthrough mode activates.
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return df, []
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns.tolist()
            if c not in forbidden and not _is_internal_column(c)
        ][:30]
        if not num_cols:
            return df, []
        out = df.copy()
        Xn = out[num_cols].fillna(0.0).values.astype(float)
        scaler = StandardScaler().fit(Xn)
        Xs = scaler.transform(Xn)
        try:
            ae = MLPRegressor(
                hidden_layer_sizes=(min(32, max(4, Xs.shape[1])), 8, min(32, max(4, Xs.shape[1]))),
                random_state=0,
                early_stopping=True,
                max_iter=100,
            ).fit(Xs, Xs)
        except Exception as e:  # noqa: BLE001
            _log.debug("autoencoded fit failed: %s", e)
            return df, []
        recon = ae.predict(Xs)
        out["X__ae_recon_error"] = np.mean((Xs - recon) ** 2, axis=1)
        new_cols = ["X__ae_recon_error"]
        # Bottleneck activations — sklearn doesn't expose intermediate layers
        # directly; approximate with first-layer activations via coefs.
        try:
            W1 = ae.coefs_[0]
            b1 = ae.intercepts_[0]
            act = np.maximum(0, Xs @ W1 + b1)  # ReLU
            for j in range(min(8, act.shape[1])):
                name = f"X__ae_z{j}"
                out[name] = act[:, j]
                new_cols.append(name)
        except Exception as e:  # noqa: BLE001
            _log.debug("autoencoded bottleneck extraction failed: %s", e)
        return out, new_cols
    _log.warning("unknown engineered group: %s", group)
    return df, []


def _sketch_top3_univariate(top_interactions: list[dict] | None) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for it in (top_interactions or []):
        for k in ("col_a", "col_b"):
            v = it.get(k)
            if v and v not in seen:
                cols.append(v)
                seen.add(v)
            if len(cols) >= 3:
                break
        if len(cols) >= 3:
            break
    return cols


def expand_features(
    df: pd.DataFrame,
    feature_dsl: list[str],
    mission: Mission,
    *,
    sketch_top_interactions: list[dict] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Expand `feature_dsl` against `df`+MISSION+sketch context.

    Returns (augmented_df, concrete_feature_columns).
    """
    out = df
    concrete: list[str] = []

    # Forbidden set for engineered-feature defense in depth.
    forbidden_for_engineering: set[str] = set(mission.forbidden_columns or [])
    forbidden_for_engineering.add(mission.target_column)

    for tok in feature_dsl:
        tok = tok.strip()
        if not tok:
            continue
        if tok == "+all_allowed":
            concrete.extend(_resolve_all_allowed(mission, df.columns))
            continue
        if tok == "+lag_downstream":
            # Capability-agnostic: when MISSION has a time_column, materialize
            # lag-1 copies of all non-forbidden non-target numerics so the
            # model can use prior-row context. The audit gate catches actual
            # leakage. The lag amount is taken from the first non-asof join's
            # `lag_policy` if it's parseable as `use_window_<N>` (e.g.
            # `use_window_2`), else defaults to 1.
            if not mission.time_column:
                continue
            lag = 1
            for jp in (mission.join_plan or []):
                pol = (jp.lag_policy or "")
                m = re.search(r"use_window_(\d+)", pol)
                if m:
                    lag = max(1, int(m.group(1)))
                    break
            num = [
                c for c in out.select_dtypes(include=[np.number]).columns.tolist()
                if c not in forbidden_for_engineering
                and c != mission.time_column
                and not _is_internal_column(c)
            ][:25]
            for c in num:
                name = f"X__{c}_lagdown{lag}"
                out[name] = out[c].shift(lag)
                concrete.append(name)
            continue
        if tok == "+leak_canary":
            # The leakage probe deliberately includes one forbidden column.
            if mission.forbidden_columns:
                concrete.append(mission.forbidden_columns[0])
            continue
        if tok.startswith("engineered:"):
            group = tok.split(":", 1)[1]
            out, new = _engineered_group(
                out,
                group,
                sketch_top_interactions=sketch_top_interactions,
                forbidden_for_engineering=forbidden_for_engineering,
            )
            concrete.extend(new)
            continue
        if tok.startswith("sketch:"):
            sub = tok.split(":", 1)[1]
            if sub == "top3_univariate":
                # Filter target/forbidden out of the univariate top-3 too.
                top3 = [
                    c for c in _sketch_top3_univariate(sketch_top_interactions)
                    if c not in forbidden_for_engineering and not _is_internal_column(c)
                ]
                concrete.extend(top3)
            continue
        # Bare column.
        concrete.append(tok)

    # Dedupe preserving order.
    seen: set[str] = set()
    final: list[str] = []
    for c in concrete:
        if c not in seen and c in out.columns:
            final.append(c)
            seen.add(c)
    return out, final
