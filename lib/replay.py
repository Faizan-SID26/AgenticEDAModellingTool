"""Deterministic project replay.

Reads `experiment_log.jsonl` + the framework version pinned in PROJECT.json
and reproduces all artifacts from the data:
    1. Re-load and re-join the source tables.
    2. Re-build the sketch with the same seed.
    3. Re-run each plan dict in order (the plan dict is reconstructed
       from the experiment row).
    4. Compare the replayed metrics to the recorded metrics; report any
       drift.

A clean replay is the gold-standard test of determinism.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from lib.data import execute_join_plan, load_tables, write_joined
from lib.project import open_project
from lib.run import execute_plan
from lib.schemas.experiment import ExperimentResult
from lib.schemas.mission import Mission
from lib.schemas.plan import PlanDict, PriorEvidence
from lib.sketch.builder import build_sketch
from lib.workspace import project_path, resolve_workspace
from lib.capabilities import validate_composition

_log = logging.getLogger("eda.replay")


def _reconstruct_plan(e: ExperimentResult) -> PlanDict:
    """Rebuild a PlanDict from a recorded ExperimentResult.

    `prior_evidence` and `expected_info_gain` are reconstructed using
    plausible defaults (the original is not stored on the experiment row;
    the plan was emitted by the agent and is implicit in the experiment).
    """
    return PlanDict(
        id=e.id,
        iteration=e.iteration,
        hypothesis_id=e.hypothesis_id,
        model=e.model,
        features=e.features_used or ["+all_allowed"],
        params=dict(e.params or {}),
        calibrate=e.calibrated,
        prior_evidence=PriorEvidence(
            kind="prior_experiment",
            reference=e.id,
            summary="reconstructed for replay",
        ),
        technique_family=e.technique_family,
        area=e.area,
        expected_info_gain=float(e.info_gain_actual or 0.0),
    )


def replay_project(
    project_name: str,
    *,
    workspace: Optional[Path] = None,
    up_to_iteration: Optional[int] = None,
) -> dict[str, Any]:
    """Replay a project's experiment log and return a drift report."""
    ws = resolve_workspace(workspace)
    proj = project_path(ws, project_name)
    meta = open_project(workspace, project_name)
    mission = Mission.model_validate_json((proj / "MISSION.json").read_text(encoding="utf-8"))

    # 1. Re-load + re-join.
    tables = load_tables(proj)
    df = execute_join_plan(tables, mission)
    write_joined(proj / "_replay", df) if False else write_joined(proj, df)

    # 2. Re-build sketch.
    cap_key = validate_composition(mission.capability).key
    build_sketch(proj, df, mission, capability_keys=[cap_key], seed=0)

    # 3. Replay each plan.
    log_path = proj / "experiment_log.jsonl"
    if not log_path.exists():
        return {"replayed": 0, "drift": [], "framework_version_pin": meta.framework_version_pin}
    drift: list[dict[str, Any]] = []
    n_replayed = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        original = ExperimentResult.model_validate_json(line)
        if up_to_iteration is not None and original.iteration > up_to_iteration:
            break
        plan = _reconstruct_plan(original)
        replayed = execute_plan(proj, mission, plan, seed=int(original.seeds.get("numpy", 0)))
        n_replayed += 1
        # Compare primary metric value.
        a = original.primary_metric_value
        b = replayed.primary_metric_value
        delta: Optional[float] = None
        if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
            delta = abs(a - b)
        drift.append(
            {
                "id": original.id,
                "primary_metric": original.primary_metric,
                "original": a,
                "replayed": b,
                "abs_delta": delta,
            }
        )
    return {
        "project_name": project_name,
        "framework_version_pin": meta.framework_version_pin,
        "replayed": n_replayed,
        "drift": drift,
    }
