"""Data-loading + join execution.

Loads raw files under `<project>/data/`, executes the MISSION's
`join_plan`, and applies the lag-join policy from the domain (asof joins
default to `use_immediate_prior`).

The resulting joined parquet is persisted at
`<project>/sketch/raw_joined.parquet` (gitignored). The sketch builder
reads from this file. The agent never reads it directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from lib.schemas.mission import JoinSpec, Mission

_log = logging.getLogger("eda.data")


def _read_one(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suf == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported data file: {path}")


def load_tables(project_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every supported file under `<project>/data/` keyed by stem."""
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    out: dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        return out
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".csv", ".parquet", ".jsonl", ".ndjson", ".json"):
            continue
        out[p.stem] = _read_one(p)
    return out


def _execute_one_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    spec: JoinSpec,
    *,
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """Execute a single join according to its spec."""
    if spec.how == "asof":
        if not time_column:
            raise ValueError("asof join requires time_column on the MISSION")
        left_sorted = left.sort_values(time_column)
        right_sorted = right.sort_values(time_column)
        # use_immediate_prior → backward direction (right value is the most
        # recent at-or-before the left timestamp).
        return pd.merge_asof(
            left_sorted,
            right_sorted,
            on=time_column,
            by=spec.on if spec.on != [time_column] else None,
            direction="backward",
            suffixes=("", f"_{spec.right_table}"),
        )
    return left.merge(
        right,
        how=spec.how,
        on=spec.on,
        suffixes=("", f"_{spec.right_table}"),
    )


def execute_join_plan(
    tables: dict[str, pd.DataFrame],
    mission: Mission,
) -> pd.DataFrame:
    """Apply the MISSION's join plan and return a single joined DataFrame.

    If the join plan is empty, returns the first table (or raises if there
    are zero or more than one tables and no plan).
    """
    if not mission.join_plan:
        if len(tables) == 1:
            return next(iter(tables.values())).copy()
        if len(tables) == 0:
            raise FileNotFoundError("no data tables loaded")
        raise ValueError(
            f"multiple tables loaded but MISSION.join_plan is empty: {sorted(tables)}"
        )
    # Start from the first spec's left table.
    first = mission.join_plan[0]
    if first.left_table not in tables:
        raise KeyError(f"join_plan references unknown left_table: {first.left_table}")
    cur = tables[first.left_table].copy()
    for spec in mission.join_plan:
        right = tables.get(spec.right_table)
        if right is None:
            raise KeyError(f"join_plan references unknown right_table: {spec.right_table}")
        cur = _execute_one_join(cur, right, spec, time_column=mission.time_column)
    return cur


def write_joined(project_dir: Path, df: pd.DataFrame) -> Path:
    """Persist the joined frame to sketch/raw_joined.parquet."""
    p = Path(project_dir) / "sketch" / "raw_joined.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return p


def load_joined(project_dir: Path) -> pd.DataFrame:
    """Read back the persisted joined frame."""
    p = Path(project_dir) / "sketch" / "raw_joined.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)
