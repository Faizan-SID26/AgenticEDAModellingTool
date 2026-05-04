"""Skeleton domain module — copy this file when adding a new domain.

Steps:
    1. Copy this file to ``lib/domains/<your_domain>.py``.
    2. Fill in SPEC with domain priors.
    3. Register the new module in ``lib.domains.__init__._DOMAIN_MODULES``.
    4. Optionally add domain-specific seed-hypothesis recipes under ``recipes/``.
"""
from __future__ import annotations

from lib.domains.base import DomainSpec, HardBound, PhysicsRelation

SPEC = DomainSpec(
    key="_template",
    description="Skeleton; do not register. Use this as a starting point for new domains.",
    stage_keywords=(
        # ("stage_name", ("keyword1", "keyword2", ...)),
    ),
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
