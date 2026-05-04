"""Capability registry.

Every capability module exports a `SPEC: CapabilitySpec` and a
`make_splitter() -> ValidationSplitter`. The registry below maps the
capability key to the module, and `get(key)` returns the spec.

Composition validators check that a `CapabilityComposition` is consistent
with at least one registered capability.
"""
from __future__ import annotations

import importlib
from typing import Iterable

from lib.capabilities.base import CapabilitySpec
from lib.schemas.mission import CapabilityComposition

_CAPABILITY_MODULES = (
    "lib.capabilities.tabular_classification",
    "lib.capabilities.tabular_regression",
    "lib.capabilities.temporal_classification",
    "lib.capabilities.forecasting",
    "lib.capabilities.predictive_maintenance",
    "lib.capabilities.anomaly_detection",
    "lib.capabilities.root_cause_attribution",
)

_REGISTRY: dict[str, CapabilitySpec] = {}


def _load_all() -> None:
    """Lazily import all capability modules and register them."""
    if _REGISTRY:
        return
    for mod_name in _CAPABILITY_MODULES:
        mod = importlib.import_module(mod_name)
        spec: CapabilitySpec = getattr(mod, "SPEC")
        _REGISTRY[spec.key] = spec


def list_capabilities() -> list[str]:
    """Return all registered capability keys."""
    _load_all()
    return sorted(_REGISTRY.keys())


def get(key: str) -> CapabilitySpec:
    """Return the spec for a key, raising KeyError if unknown."""
    _load_all()
    return _REGISTRY[key]


def composition_signature(comp: CapabilityComposition) -> str:
    """Stable string fingerprint of a composition (used for retrieval keys)."""
    return (
        f"{comp.target_type}|"
        f"{comp.temporal_structure}|"
        f"{comp.leakage_model}|"
        f"{comp.validation_strategy}|"
        f"{comp.recommendation_type}"
    )


def find_compatible(comp: CapabilityComposition) -> list[CapabilitySpec]:
    """Return all registered capabilities consistent with this composition."""
    _load_all()
    out = []
    for spec in _REGISTRY.values():
        # Match on target_type and temporal_structure — these are the load-bearing fields.
        if spec.composition.target_type != comp.target_type:
            continue
        if spec.composition.temporal_structure != comp.temporal_structure:
            # Allow `none` capability to handle `regime_based` if no better fit exists.
            if spec.composition.temporal_structure != "none":
                continue
        out.append(spec)
    return out


def validate_composition(comp: CapabilityComposition) -> CapabilitySpec:
    """Pick the best matching capability or raise ValueError.

    Selection rule: among compatible capabilities, prefer the one whose
    own `composition` exactly matches `comp` on temporal_structure +
    leakage_model. If none, fall back to first compatible. If empty, raise.
    """
    matches = find_compatible(comp)
    if not matches:
        raise ValueError(
            f"no registered capability matches target_type={comp.target_type}, "
            f"temporal_structure={comp.temporal_structure}"
        )
    exact = [
        s
        for s in matches
        if s.composition.temporal_structure == comp.temporal_structure
        and s.composition.leakage_model == comp.leakage_model
    ]
    return (exact or matches)[0]


__all__ = [
    "CapabilitySpec",
    "list_capabilities",
    "get",
    "composition_signature",
    "find_compatible",
    "validate_composition",
]
