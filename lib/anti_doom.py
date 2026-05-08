"""Helpers for forcing structural diversification when the doom-loop fires.

The orchestrator can't reliably maintain an in-memory `recent_plans` buffer
across iterations (the LLM session is not guaranteed to preserve it), so the
plan fingerprint is persisted to `memory/RECENT_PLANS.jsonl` by
`lib.state.record(...)`. This module provides:

- `fingerprint_of(plan)` — re-export of `lib.doom_loop._fingerprint`.
- `load_recent_fingerprints(project_dir, *, window)` — read the trailing
  window of fingerprints from disk.
- `assert_distinct(plan_or_fingerprint, recent_fingerprints, *, window)` —
  raise ValueError if the proposed plan's fingerprint matches any of the
  last `window` fingerprints.

The orchestrator passes `recent_fingerprints` into `PlanDict` validation as
`validation_context["recent_fingerprints"]` so the validator can reject
collisions before the runner ever sees the plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from lib.doom_loop import _fingerprint
from lib.schemas.plan import PlanDict


_RECENT_PLANS_PATH = "memory/RECENT_PLANS.jsonl"
_DEFAULT_TRIM = 10


def fingerprint_of(plan: PlanDict) -> str:
    """Stable fingerprint of `(model, technique_family, area, sorted(features))`."""
    return _fingerprint(plan)


def recent_plans_path(project_dir: Path) -> Path:
    return Path(project_dir) / _RECENT_PLANS_PATH


def append_fingerprint(project_dir: Path, fp: str, *, iteration: int, trim: int = _DEFAULT_TRIM) -> Path:
    """Append `{fingerprint, iteration}` to RECENT_PLANS.jsonl and trim to
    the last `trim` entries. Cheap; called once per iteration from
    `lib.state.record(...)`."""
    p = recent_plans_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.append({"fingerprint": fp, "iteration": int(iteration)})
    rows = rows[-trim:]
    tmp = p.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def load_recent_fingerprints(project_dir: Path, *, window: int = 3) -> list[str]:
    """Return the trailing `window` plan fingerprints from disk, oldest first."""
    p = recent_plans_path(project_dir)
    if not p.exists():
        return []
    rows: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fp = r.get("fingerprint")
        if isinstance(fp, str):
            rows.append(fp)
    return rows[-window:]


def assert_distinct(
    plan_or_fingerprint: Union[PlanDict, str],
    recent_fingerprints: Iterable[str],
    *,
    window: int = 3,
) -> None:
    """Raise ValueError if the proposed plan/fingerprint matches any of the
    last `window` recent fingerprints. The orchestrator calls this after the
    doom-loop fires; outside of a doom-loop firing, callers should not enforce
    structural distinctness — the bandit + researcher prompt handle ordinary
    diversity."""
    if isinstance(plan_or_fingerprint, str):
        fp = plan_or_fingerprint
    else:
        fp = fingerprint_of(plan_or_fingerprint)
    recent = list(recent_fingerprints)[-window:]
    if fp in recent:
        raise ValueError(
            f"plan fingerprint {fp!r} matches one of the last {window} plans "
            f"({recent!r}); doom-loop active — pick a structurally different "
            "plan (different model OR technique_family OR area OR features)."
        )
