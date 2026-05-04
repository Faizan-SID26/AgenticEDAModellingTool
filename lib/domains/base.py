"""Domain interface contract.

A domain module supplies *priors* about the data: which keywords identify
process stages, which columns are typically downstream-of-target, default
join policies, expected interactions, sensor-failure patterns, hard
physical bounds, and a small set of domain-specific seed hypotheses to add
to the universal seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from lib.schemas.mission import Mission


@dataclass(frozen=True)
class PhysicsRelation:
    """A known physical / business relation between named feature roles.

    Used by the planner to seed domain-specific feature engineering and by
    the skeptic to flag results that contradict known physics.
    """

    name: str
    """Short identifier (e.g., 'arrhenius_temp_dependence')."""

    description: str
    """Human-friendly explanation."""

    feature_roles: tuple[str, ...]
    """Semantic role tags this relation involves."""

    form: str
    """Functional form, e.g., 'linear', 'arrhenius', 'power_law'."""


@dataclass(frozen=True)
class HardBound:
    """A hard bound on a column / role's value.

    Skeptic flags any prediction or input that violates these.
    """

    role: str
    """Semantic role this bound applies to."""

    lower: Optional[float]
    upper: Optional[float]
    units: str = ""


@dataclass(frozen=True)
class DomainSpec:
    """Static specification of a domain.

    Concrete domain modules expose a `SPEC: DomainSpec` and helper
    callables. The registry in `lib.domains.__init__` discovers them.
    """

    key: str
    """Stable identifier (e.g., 'manufacturing')."""

    description: str

    stage_keywords: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)
    """(stage_name, keyword_tuple) pairs. `infer_stage` returns the first
    stage whose keywords match (substring, lowercase). For
    `leakage_model=stage_frontier`, columns from a stage *after* the
    target's stage are forbidden."""

    default_forbidden: tuple[str, ...] = field(default_factory=tuple)
    """Column-name substrings that are always treated as forbidden when
    using the default leakage policy (e.g., 'qc_', 'audit_', 'inspector_')."""

    default_leak_frontier: Optional[str] = None
    """Default stage at which leakage is gated (e.g., 'final_qa')."""

    lag_join_default_policy: str = "use_immediate_prior"
    """Default lag policy for asof joins (used by lib.data)."""

    physics_relations: tuple[PhysicsRelation, ...] = field(default_factory=tuple)

    expected_interactions: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    """Expected feature-role pairs/triples. Researcher uses these to decide
    which interactions to surface from L2 first."""

    sensor_failure_patterns: tuple[str, ...] = field(default_factory=tuple)
    """Patterns that mark sensor failure (used by the skeptic / failure-mode
    catalog seeding). Examples: 'flatline_>= N hours', 'physically_impossible_value'."""

    hard_bounds: tuple[HardBound, ...] = field(default_factory=tuple)

    skeptic_extras: tuple[str, ...] = field(default_factory=tuple)
    """Extra skeptic check keys to enable for this domain
    (in addition to capability-defaults)."""

    seed_hypotheses: tuple[str, ...] = field(default_factory=tuple)
    """Recipe / hypothesis-template keys to seed alongside the universal seeds."""


def infer_stage_from_keywords(
    column_name: str, stage_keywords: tuple[tuple[str, tuple[str, ...]], ...]
) -> Optional[str]:
    """Return the first matching stage for a column name (substring, lowercase)."""
    name_lc = column_name.lower()
    for stage, keywords in stage_keywords:
        for kw in keywords:
            if kw.lower() in name_lc:
                return stage
    return None


def bundle_extras_default(mission: Mission) -> dict[str, Any]:
    """Return a domain-extras blob (default: empty)."""
    return {}
