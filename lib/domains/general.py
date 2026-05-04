"""No-domain fallback. Empty priors."""
from __future__ import annotations

from lib.domains.base import DomainSpec

SPEC = DomainSpec(
    key="general",
    description="No-domain fallback. No priors; the framework relies entirely on the sketch and universal seeds.",
    stage_keywords=(),
    default_forbidden=(),
    default_leak_frontier=None,
    lag_join_default_policy="use_immediate_prior",
    physics_relations=(),
    expected_interactions=(),
    sensor_failure_patterns=(),
    hard_bounds=(),
    skeptic_extras=(),
    seed_hypotheses=(),
)
