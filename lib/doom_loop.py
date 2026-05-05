"""Doom-loop detector.

If the same plan-fingerprint is repeated across N consecutive iterations
without primary metric improvement, fire. The researcher should pick a
different arm / area on the next iteration; if not, /run halts on
catastrophic stagnation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from lib.schemas.experiment import ExperimentResult
from lib.schemas.plan import PlanDict


def _fingerprint(plan: PlanDict) -> str:
    """Hash the (model, technique_family, area, sorted_features) tuple."""
    payload = "|".join(
        [
            plan.model,
            plan.technique_family,
            plan.area,
            ",".join(sorted(plan.features)),
        ]
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


@dataclass
class DoomLoopVerdict:
    fired: bool
    fingerprint: str
    consecutive_repeats: int
    reason: str = ""


def check(
    recent_plans: Iterable[PlanDict],
    recent_experiments: Iterable[ExperimentResult],
    *,
    window: int = 3,
) -> DoomLoopVerdict:
    """Return whether the doom-loop has fired on the trailing window."""
    plans = list(recent_plans)[-window:]
    if len(plans) < window:
        return DoomLoopVerdict(False, fingerprint="", consecutive_repeats=0)

    fps = [_fingerprint(p) for p in plans]
    same = all(f == fps[0] for f in fps)

    exps = list(recent_experiments)[-window:]
    if len(exps) < window:
        return DoomLoopVerdict(False, fingerprint=fps[0], consecutive_repeats=len(plans))
    metrics = [e.primary_metric_value for e in exps if e.primary_metric_value is not None]
    if len(metrics) < window:
        # Cannot judge flatness when the metric was non-finite for any of
        # the trailing experiments. Doom-loop is *not* fired in that case.
        return DoomLoopVerdict(False, fingerprint=fps[0], consecutive_repeats=int(same) * window)
    flat = max(metrics) - min(metrics) < 1e-4
    if same and flat:
        return DoomLoopVerdict(
            fired=True,
            fingerprint=fps[0],
            consecutive_repeats=window,
            reason="same plan fingerprint repeated with no metric movement",
        )
    return DoomLoopVerdict(
        fired=False, fingerprint=fps[0], consecutive_repeats=int(same) * window
    )
