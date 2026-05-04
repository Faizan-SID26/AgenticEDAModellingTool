"""Forecasting / demand domain (validates the domain abstraction)."""
from __future__ import annotations

from typing import Any

from lib.domains.base import DomainSpec, HardBound, PhysicsRelation
from lib.schemas.mission import Mission

_STAGE_KEYWORDS = (
    ("calendar", ("date", "day", "week", "month", "year", "holiday")),
    ("price", ("price", "promo", "discount", "list_")),
    ("inventory", ("stock", "inventory", "on_hand", "backorder")),
    ("demand_signal", ("clicks", "search", "trend", "lead_")),
    ("outcome", ("sales", "demand", "units", "orders")),
)

DEFAULT_FORBIDDEN = (
    "future_",
    "actuals_t_plus",
    "audit_",
    "post_event",
)

PHYSICS_RELATIONS = (
    PhysicsRelation(
        name="seasonality",
        description="Demand has yearly+weekly seasonality components.",
        feature_roles=("<calendar:dow>", "<calendar:month>", "<outcome:demand>"),
        form="additive_seasonal",
    ),
    PhysicsRelation(
        name="price_elasticity",
        description="Demand decreases with price (elasticity is negative).",
        feature_roles=("<price:list>", "<outcome:demand>"),
        form="log_log_linear",
    ),
)

EXPECTED_INTERACTIONS = (
    ("<calendar:holiday>", "<price:promo>"),
    ("<calendar:dow>", "<calendar:month>"),
    ("<inventory:stock>", "<outcome:demand>"),
)

SENSOR_FAILURE_PATTERNS = (
    "negative_demand",
    "calendar_gap",
    "duplicate_timestamps",
)

HARD_BOUNDS = (
    HardBound(role="<outcome:demand>", lower=0.0, upper=None, units="units"),
    HardBound(role="<price:list>", lower=0.0, upper=None, units="currency"),
    HardBound(role="<inventory:stock>", lower=0.0, upper=None, units="units"),
)

SPEC = DomainSpec(
    key="forecasting_demand",
    description="Demand forecasting with calendar/price/inventory features.",
    stage_keywords=_STAGE_KEYWORDS,
    default_forbidden=DEFAULT_FORBIDDEN,
    default_leak_frontier=None,
    lag_join_default_policy="use_immediate_prior",
    physics_relations=PHYSICS_RELATIONS,
    expected_interactions=EXPECTED_INTERACTIONS,
    sensor_failure_patterns=SENSOR_FAILURE_PATTERNS,
    hard_bounds=HARD_BOUNDS,
    skeptic_extras=("non_negative_demand_check", "calendar_continuity_check"),
    seed_hypotheses=(
        "naive_seasonal_baseline",
        "lagged_ridge",
        "stl_decompose_then_residual_model",
    ),
)


def bundle_extras(mission: Mission) -> dict[str, Any]:
    return {
        "expected_interactions": [list(i) for i in EXPECTED_INTERACTIONS],
        "lag_join_policy": SPEC.lag_join_default_policy,
    }
