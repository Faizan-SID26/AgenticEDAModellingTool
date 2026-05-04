"""Process manufacturing domain (the v1 reference path)."""
from __future__ import annotations

from typing import Any, Optional

from lib.domains.base import (
    DomainSpec,
    HardBound,
    PhysicsRelation,
    infer_stage_from_keywords,
)
from lib.schemas.mission import Mission

# Stage keywords are searched in order; the first match wins. Defining
# upstream → downstream order matters: stage_frontier leakage is rejected
# only for stages strictly downstream of the target's stage.
_STAGE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("raw_material", ("raw_", "feedstock", "incoming", "supplier_")),
    ("preprocessing", ("preprocess", "mixing", "milling", "grinding")),
    ("reaction", ("reactor", "reaction", "kiln", "furnace", "synthesis")),
    ("refining", ("distillation", "filtration", "purify", "refine")),
    ("forming", ("press", "extrude", "mold", "form_", "casting")),
    ("finishing", ("anneal", "cure", "coat", "polish", "finish_")),
    ("packaging", ("pack_", "label_", "wrap_")),
    ("final_qa", ("final_qc", "final_qa", "release_test", "inspector_", "audit_", "downstream_qc")),
)

DEFAULT_FORBIDDEN: tuple[str, ...] = (
    "qc_",
    "audit_",
    "inspector_",
    "downstream_",
    "release_",
    "lab_result_",
    "final_yield",
    "final_grade",
    "post_",
)
"""Substrings that, if present in a column name, mark it as forbidden by
default under the stage_frontier leakage policy."""

PHYSICS_RELATIONS: tuple[PhysicsRelation, ...] = (
    PhysicsRelation(
        name="arrhenius_temp_dependence",
        description="Reaction rate ~ exp(-E/RT); higher temperature accelerates kinetics.",
        feature_roles=("<sensor:temperature>",),
        form="arrhenius",
    ),
    PhysicsRelation(
        name="linear_residence_time_yield",
        description="Yield typically increases roughly linearly with residence time within an operating window.",
        feature_roles=("<process:residence_time>", "<outcome:yield>"),
        form="linear",
    ),
    PhysicsRelation(
        name="pressure_temperature_coupling",
        description="In closed reactors, pressure and temperature are coupled (PV=nRT).",
        feature_roles=("<sensor:pressure>", "<sensor:temperature>"),
        form="linear",
    ),
)

EXPECTED_INTERACTIONS: tuple[tuple[str, ...], ...] = (
    ("<sensor:temperature>", "<sensor:pressure>"),
    ("<sensor:temperature>", "<process:residence_time>"),
    ("<process:flowrate>", "<sensor:concentration>"),
    ("<sensor:vibration>", "<asset:age_days>"),
)

SENSOR_FAILURE_PATTERNS: tuple[str, ...] = (
    "flatline_>=1h",
    "spike_outside_3sigma_for_>=10min",
    "physically_impossible_value",
    "stale_timestamp_repeat",
    "off_calibration_drift_>5%",
)

HARD_BOUNDS: tuple[HardBound, ...] = (
    HardBound(role="<sensor:temperature>", lower=-273.15, upper=2000.0, units="C"),
    HardBound(role="<sensor:pressure>", lower=0.0, upper=1000.0, units="bar"),
    HardBound(role="<sensor:relative_humidity>", lower=0.0, upper=100.0, units="%"),
    HardBound(role="<process:flowrate>", lower=0.0, upper=None, units="kg/s"),
    HardBound(role="<process:yield_rate>", lower=0.0, upper=1.0, units="fraction"),
)

SPEC = DomainSpec(
    key="manufacturing",
    description="Process manufacturing with stage-frontier leakage and lag-join asof semantics.",
    stage_keywords=_STAGE_KEYWORDS,
    default_forbidden=DEFAULT_FORBIDDEN,
    default_leak_frontier="final_qa",
    lag_join_default_policy="use_immediate_prior",
    physics_relations=PHYSICS_RELATIONS,
    expected_interactions=EXPECTED_INTERACTIONS,
    sensor_failure_patterns=SENSOR_FAILURE_PATTERNS,
    hard_bounds=HARD_BOUNDS,
    skeptic_extras=(
        "physical_bounds_check",
        "sensor_flatline_check",
        "stage_frontier_audit",
        "physics_relation_consistency",
    ),
    seed_hypotheses=(
        "regime_specific_submodel",
        "stage_frontier_baseline",
        "lag_join_with_immediate_prior",
        "interaction_temp_pressure",
    ),
)


def infer_stage(column_name: str) -> Optional[str]:
    """Return the manufacturing stage for a column name, or None."""
    return infer_stage_from_keywords(column_name, _STAGE_KEYWORDS)


def bundle_extras(mission: Mission) -> dict[str, Any]:
    """Return manufacturing-specific extras for the iteration brief.

    Includes the leak frontier, the inferred stage of the target column,
    and a copy of any expected interactions whose roles are likely
    populated.
    """
    target_stage = infer_stage(mission.target_column)
    return {
        "leak_frontier": SPEC.default_leak_frontier,
        "target_stage": target_stage,
        "expected_interactions": [list(i) for i in EXPECTED_INTERACTIONS],
        "lag_join_policy": SPEC.lag_join_default_policy,
    }
