"""Domain registry."""
from __future__ import annotations

import importlib
from typing import Iterable

from lib.domains.base import DomainSpec

_DOMAIN_MODULES = (
    "lib.domains.general",
    "lib.domains.manufacturing",
    "lib.domains.forecasting_demand",
)

_REGISTRY: dict[str, DomainSpec] = {}


def _load_all() -> None:
    if _REGISTRY:
        return
    for mod_name in _DOMAIN_MODULES:
        mod = importlib.import_module(mod_name)
        spec: DomainSpec = getattr(mod, "SPEC")
        _REGISTRY[spec.key] = spec


def list_domains() -> list[str]:
    _load_all()
    return sorted(_REGISTRY.keys())


def get(key: str) -> DomainSpec:
    """Return the spec for `key`. Returns the `general` spec if unknown
    (warning logged) so projects can still proceed without an exact match."""
    _load_all()
    if key in _REGISTRY:
        return _REGISTRY[key]
    return _REGISTRY["general"]


__all__ = ["DomainSpec", "list_domains", "get"]
