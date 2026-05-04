"""`build_sketch`: construct all sketch layers from a joined parquet.

Called once at /bootstrap. Subsequent updates are deterministic and
incremental (`lib.sketch.updaters`). The build is *deterministic* given a
seed: re-running on the same data with the same seed produces a
bit-identical manifest (modulo float-printing precision).
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from lib.schemas.mission import Mission
from lib.schemas.sketch import (
    L7FailureClusterSummary,
    SketchManifest,
)
from lib.sketch.l1_distributions import build_l1, save_l1
from lib.sketch.l2_joint import build_l2, save_l2
from lib.sketch.l3_regimes import build_l3, save_l3
from lib.sketch.l4_coresets import build_coreset, save_coreset
from lib.sketch.l5_timeseries import build_l5, save_l5
from lib.sketch.l6_causal import build_l6, save_l6
from lib.sketch.l7_failure_modes import save_l7
from lib.sketch.manifest import save_manifest
from lib.sketch.similarity import compute_similarity_vector

_log = logging.getLogger("eda.sketch.builder")


def _seed_everything(seed: int) -> None:
    """Seed all random sources we use during the build."""
    random.seed(seed)
    np.random.seed(seed)


def _file_size_bytes(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def build_sketch(
    project_dir: Path,
    df: pd.DataFrame,
    mission: Mission,
    *,
    capability_keys: Iterable[str],
    seed: int = 0,
    coreset_size: int = 5000,
) -> SketchManifest:
    """Build all 7 layers from `df` and persist them under `<project>/sketch/`.

    Returns the validated `SketchManifest`. Total binary size of L1..L7
    files is summed and stored on the manifest as a sanity check (target
    <1MB).
    """
    project_dir = Path(project_dir)
    sketch_dir = project_dir / "sketch"
    sketch_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(seed)
    _log.info("building sketch for %s (n_rows=%d, n_cols=%d)", project_dir.name, len(df), df.shape[1])

    # L1: per-column distributions.
    l1_summaries = build_l1(df)
    l1_path = sketch_dir / "L1.json"
    save_l1(l1_summaries, l1_path)

    # L2: joint structure.
    l2_summary = build_l2(df, target=mission.target_column, k=20, top_k_interactions=10, seed=seed)
    l2_path = sketch_dir / "L2.json"
    save_l2(l2_summary, l2_path)

    # L3: regimes (only meaningful if temporal_structure != none).
    l3_summary = build_l3(
        df,
        time_column=mission.time_column,
        target=mission.target_column,
    )
    l3_path = sketch_dir / "L3.json"
    save_l3(l3_summary, l3_path)

    # L4: per-capability coresets.
    l4_paths: list[str] = []
    for key in capability_keys:
        coreset, _summary = build_coreset(
            df,
            capability_key=key,
            target=mission.target_column,
            time_column=mission.time_column,
            n_rows=coreset_size,
            seed=seed,
        )
        cs_path = sketch_dir / f"L4_{key}.parquet"
        save_coreset(coreset, cs_path)
        l4_paths.append(str(cs_path.relative_to(project_dir)))

    # L5: per-time-series motifs/discords.
    l5_summary = build_l5(df, time_column=mission.time_column)
    l5_path = sketch_dir / "L5.json"
    save_l5(l5_summary, l5_path)

    # L6: causal hints.
    l6_summary = build_l6(df, alpha=0.05, max_cond_set=2, max_columns=20, seed=seed)
    l6_path = sketch_dir / "L6.json"
    save_l6(l6_summary, l6_path)

    # L7: starts empty; updaters fill it later.
    l7_path = sketch_dir / "L7.jsonl"
    save_l7([], l7_path)

    # Total size sanity check.
    total_bytes = sum(
        _file_size_bytes(p)
        for p in (l1_path, l2_path, l3_path, l5_path, l6_path, l7_path)
    )
    for cs in l4_paths:
        total_bytes += _file_size_bytes(project_dir / cs)

    sim_vec = compute_similarity_vector(l1_summaries, l2_summary, l3_summary, [])

    manifest = SketchManifest(
        project_name=mission.project_name,
        seed=seed,
        n_rows_source=int(len(df)),
        n_columns_source=int(df.shape[1]),
        capabilities=list(capability_keys),
        l1_path=str(l1_path.relative_to(project_dir)),
        l2_path=str(l2_path.relative_to(project_dir)),
        l3_path=str(l3_path.relative_to(project_dir)),
        l4_paths=l4_paths,
        l5_path=str(l5_path.relative_to(project_dir)),
        l6_path=str(l6_path.relative_to(project_dir)),
        l7_path=str(l7_path.relative_to(project_dir)),
        total_size_bytes=int(total_bytes),
        similarity_vector=list(map(float, sim_vec)),
    )
    save_manifest(project_dir, manifest)
    _log.info(
        "sketch built: %s layers, total ≈ %.0f KB",
        7,
        total_bytes / 1024,
    )
    return manifest
