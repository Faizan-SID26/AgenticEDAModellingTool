"""Capability interface contract.

A capability module declares what the framework needs to know to handle a
particular shape of ML problem: required mission fields, default models &
metrics, the validation splitter, and any sketch-layer extras.

Modules dispatch on capability *fields* (target_type, temporal_structure,
etc.) rather than on a single problem-type label, so adding a new
capability is purely additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

from lib.schemas.mission import CapabilityComposition, Mission


@dataclass(frozen=True)
class CapabilitySpec:
    """Static specification of a capability.

    Concrete capability modules construct one of these and register it.
    """

    key: str
    """Stable identifier (e.g., 'tabular_classification')."""

    description: str
    """One-sentence human-friendly description."""

    composition: CapabilityComposition
    """The default capability composition this module assumes."""

    required_mission_fields: tuple[str, ...]
    """Mission attribute names that must be populated for this capability."""

    default_models: tuple[str, ...]
    """Registry keys (`lib.registry`) of default models tried during /run."""

    default_metrics: tuple[str, ...]
    """Metric names from `lib.eval` to compute on every experiment."""

    primary_metric: str
    """The metric that drives best-tracking and the success criterion."""

    primary_metric_direction: str
    """'>=' if higher is better, '<=' if lower is better."""

    sketch_extras_needed: tuple[str, ...] = field(default_factory=tuple)
    """Sketch layers / queries this capability specifically benefits from
    (e.g., 'L3_regimes', 'L5_timeseries', 'L6_causal')."""

    seed_hypothesis_recipe_keys: tuple[str, ...] = field(default_factory=tuple)
    """Recipe-specific hypothesis seeds to combine with the 5 universal
    seeds at /plan time."""


class ValidationSplitter(Protocol):
    """Callable producing (train_idx, val_idx, optional test_idx) tuples."""

    def __call__(
        self,
        n_rows: int,
        *,
        time: Optional[Iterable[Any]] = None,
        groups: Optional[Iterable[Any]] = None,
        seed: int = 0,
    ) -> list[tuple[Any, Any, Optional[Any]]]: ...


def assert_required_mission_fields(spec: CapabilitySpec, mission: Mission) -> None:
    """Raise ValueError if any required field on `mission` is missing/empty."""
    missing: list[str] = []
    for f in spec.required_mission_fields:
        cur: Any = mission
        for part in f.split("."):
            if cur is None:
                missing.append(f)
                break
            cur = getattr(cur, part, None)
        if cur in (None, "", []):
            missing.append(f)
    if missing:
        raise ValueError(
            f"capability '{spec.key}' requires fields: {missing}"
        )
