"""Doom-loop detector.

If the same plan-fingerprint is repeated across N consecutive iterations
without primary metric improvement, fire. The researcher should pick a
different arm / area on the next iteration; if not, /run halts on
catastrophic stagnation.

The orchestrator should call `check_from_disk(...)` rather than `check(...)`
so the trailing window is sourced from `memory/RECENT_PLANS.jsonl` (written
by `lib.state.record`) rather than an in-memory buffer the LLM session may
not preserve.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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


def check_from_disk(project_dir: Path, *, window: int = 3) -> DoomLoopVerdict:
    """Disk-backed doom-loop check.

    Reads the trailing `window` plan fingerprints from
    `memory/RECENT_PLANS.jsonl` and the trailing experiments from the
    project's experiment log. Fires under the same condition as `check`:
    same fingerprint repeated `window` times with metric range < 1e-4.

    Use this from the orchestrator instead of maintaining an in-memory
    `recent_plans` buffer — the LLM session does not reliably preserve
    that buffer across iterations.
    """
    project_dir = Path(project_dir)
    recent_path = project_dir / "memory" / "RECENT_PLANS.jsonl"
    if not recent_path.exists():
        return DoomLoopVerdict(False, fingerprint="", consecutive_repeats=0)

    rows: list[dict] = []
    for line in recent_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    fps = [r.get("fingerprint", "") for r in rows[-window:]]
    if len(fps) < window or not all(fps):
        return DoomLoopVerdict(False, fingerprint=fps[-1] if fps else "", consecutive_repeats=0)
    same = all(f == fps[0] for f in fps)

    # Read trailing experiments to test metric flatness.
    log_path = project_dir / "experiment_log.jsonl"
    if not log_path.exists():
        return DoomLoopVerdict(False, fingerprint=fps[0], consecutive_repeats=int(same) * window)
    metrics: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = obj.get("primary_metric_value")
        if isinstance(v, (int, float)):
            metrics.append(float(v))
    metrics = metrics[-window:]
    if len(metrics) < window:
        return DoomLoopVerdict(False, fingerprint=fps[0], consecutive_repeats=int(same) * window)
    flat = max(metrics) - min(metrics) < 1e-4
    if same and flat:
        return DoomLoopVerdict(
            fired=True,
            fingerprint=fps[0],
            consecutive_repeats=window,
            reason="same plan fingerprint repeated with no metric movement (disk)",
        )
    return DoomLoopVerdict(
        fired=False, fingerprint=fps[0], consecutive_repeats=int(same) * window
    )
