"""Feature DSL expansion.

Plan dicts express features as a list of tokens. Tokens are:

- `+all_allowed` — expanded to MISSION.allowed_columns or "everything not
  in MISSION.forbidden_columns".
- `+lag_downstream` — for manufacturing: include lagged values of
  immediately-downstream-of-target columns honoring the lag-join policy.
- `engineered:<GROUP>` — looked up in the engineered-feature catalog
  (interactions_top5, ratios, polynomial_2, ...).
- `sketch:top3_univariate` — pick the top-3 numeric columns by L2 mutual
  info.
- bare column name — used verbatim.

`expand_features(...)` returns the *concrete* column list a model will
see, with engineered columns materialized in the returned DataFrame.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from lib.schemas.mission import Mission

_log = logging.getLogger("eda.features")

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
    if group == "interactions_top5":
        pairs: list[tuple[str, str]] = []
        for it in (sketch_top_interactions or [])[:5]:
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
            # Manufacturing-style: skip silently if no time column or no
            # candidate downstream columns; the audit gate will catch real
            # leakage. v1 stub: no-op (downstream lagging requires raw
            # multi-table state).
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
