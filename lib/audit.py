"""Leakage gate.

Inputs:
- Mission (the source of truth on forbidden_columns and capability).
- The concrete feature list a plan dict expanded to.

Outputs:
- AuditResult with `ok`, `forbidden_used`, and a list of warnings.

The gate is *strict* by default: any forbidden column in the feature list
fails the audit and the runner halts the experiment unless the plan is
explicitly the leakage probe (area="leakage_probe").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from lib.schemas.mission import Mission


@dataclass
class AuditResult:
    ok: bool
    forbidden_used: list[str] = field(default_factory=list)
    target_used_as_feature: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: str = ""


def audit_features(
    mission: Mission,
    features: Iterable[str],
    *,
    plan_area: str = "",
) -> AuditResult:
    """Strict audit. Returns ok=False if any forbidden column is used.

    `plan_area` lets the leakage probe (area="leakage_probe") pass the
    audit on purpose to establish the empirical leakage ceiling.
    """
    feats = list(features)
    forbidden = set(mission.forbidden_columns)
    used_forbidden = [f for f in feats if f in forbidden]
    target_used = mission.target_column in feats
    warnings: list[str] = []

    ok = True
    if target_used:
        ok = False
        warnings.append("target_column appears in features")

    if used_forbidden and plan_area != "leakage_probe":
        ok = False
        warnings.append(
            f"forbidden columns used outside a leakage probe: {used_forbidden}"
        )
    elif used_forbidden and plan_area == "leakage_probe":
        warnings.append(
            f"leakage probe explicitly using forbidden columns: {used_forbidden}"
        )

    return AuditResult(
        ok=ok,
        forbidden_used=used_forbidden,
        target_used_as_feature=target_used,
        warnings=warnings,
    )
