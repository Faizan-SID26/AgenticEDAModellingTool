"""MISSION lock: validate completeness and consistency, write artifacts.

After `/plan` collects answers, the planner calls `lock_project()` which:
    1. Re-validates the assembled MISSION via the pydantic model.
    2. Validates the capability composition against the registered set.
    3. Writes MISSION.json, COLUMNS.json, JOIN_PLAN.json,
       memory/HYPOTHESES.jsonl (universal seeds + recipe seeds + domain seeds).
    4. Updates PROJECT.json status to "planned".
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from lib.capabilities import composition_signature, validate_composition
from lib.domains import get as get_domain
from lib.planning import write_mission
from lib.project import open_project, write_project_meta
from lib.schemas.mission import Mission
from lib.schemas.project_meta import ProjectMeta
from lib.workspace import project_path, resolve_workspace
from seeds import load_universal_seeds

_log = logging.getLogger("eda.lock")


def _seed_hypothesis_records(
    mission: Mission, recipe: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    """Materialize the seed-hypothesis records for memory/HYPOTHESES.jsonl.

    Combines universal seeds + recipe-specific seeds + domain seeds.
    Deduplicates by hypothesis_id with universal seeds first.
    """
    seeds: list[dict[str, Any]] = list(load_universal_seeds())
    seen = {h["hypothesis_id"] for h in seeds}

    domain_spec = get_domain(mission.domain)
    for key in domain_spec.seed_hypotheses:
        hid = f"H-domain-{mission.domain}-{key}"
        if hid in seen:
            continue
        seeds.append(
            {
                "hypothesis_id": hid,
                "name": key,
                "summary": f"Domain seed ({mission.domain}): {key}",
                "technique_family": "boosted_tree",
                "area": "features",
                "model_hint": "lgbm_default",
                "features_dsl": ["+all_allowed"],
                "expected_info_gain": 0.5,
                "rationale": f"From domain {mission.domain} seed_hypotheses list.",
                "source": "domain",
            }
        )
        seen.add(hid)
    if recipe:
        for key in recipe.get("default_seed_hypotheses", []) or []:
            hid = f"H-recipe-{recipe.get('recipe_key', '?')}-{key}"
            if hid in seen:
                continue
            seeds.append(
                {
                    "hypothesis_id": hid,
                    "name": key,
                    "summary": f"Recipe seed: {key}",
                    "technique_family": "boosted_tree",
                    "area": "features",
                    "model_hint": "lgbm_default",
                    "features_dsl": ["+all_allowed"],
                    "expected_info_gain": 0.5,
                    "rationale": f"From recipe {recipe.get('recipe_key')}.",
                    "source": "recipe",
                }
            )
            seen.add(hid)
    return seeds


def lock_project(
    project_name: str,
    mission: Mission,
    *,
    workspace: Optional[Path] = None,
    recipe: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    """Validate and write all locked artifacts; bump PROJECT status.

    Returns a dict of artifact_kind → path.
    """
    # Composition must resolve to at least one registered capability.
    spec = validate_composition(mission.capability)
    _log.info("locked composition resolves to capability '%s'", spec.key)

    ws = resolve_workspace(workspace)
    pdir = project_path(ws, project_name)
    if not pdir.exists():
        raise FileNotFoundError(f"project not found at {pdir}")
    memdir = pdir / "memory"
    memdir.mkdir(parents=True, exist_ok=True)

    # MISSION.json
    mission_path = write_mission(pdir, mission)

    # COLUMNS.json — light: target/time/group/allowed/forbidden + capability_signature
    columns_payload = {
        "target_column": mission.target_column,
        "time_column": mission.time_column,
        "group_column": mission.group_column,
        "allowed_columns": mission.allowed_columns,
        "forbidden_columns": mission.forbidden_columns,
        "capability_signature": composition_signature(mission.capability),
    }
    columns_path = memdir / "COLUMNS.json"
    columns_path.write_text(json.dumps(columns_payload, indent=2), encoding="utf-8")

    # JOIN_PLAN.json
    join_path = memdir / "JOIN_PLAN.json"
    join_path.write_text(
        json.dumps([j.model_dump() for j in mission.join_plan], indent=2, default=str),
        encoding="utf-8",
    )

    # HYPOTHESES.jsonl — seeded
    hyp_path = memdir / "HYPOTHESES.jsonl"
    seeds = _seed_hypothesis_records(mission, recipe=recipe)
    with hyp_path.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s) + "\n")

    # Bump project status to "planned".
    meta = open_project(workspace, project_name)
    meta = ProjectMeta(**{**meta.model_dump(), "status": "planned"})
    pj_path = write_project_meta(workspace, meta)

    return {
        "mission": mission_path,
        "columns": columns_path,
        "join_plan": join_path,
        "hypotheses": hyp_path,
        "project_meta": pj_path,
    }
