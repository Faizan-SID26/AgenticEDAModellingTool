"""Inter-agent communication channel for /run.

Subagents (researcher, runner, reviewer, analyst, literature, novelty-check,
skeptic, debate-arbiter) exchange small structured JSON payloads through
this directory rather than re-feeding context to each other. The orchestrator
polls these files between dispatches; it never re-reads transcripts.

Layout:

    <project>/memory/agent_inbox/iter_<N>/<role>_<message>.json

Message names by convention (the orchestrator and the agents agree on these):

    researcher_proposal.json   — PlanDict emitted by the researcher
    novelty_verdict.json       — {verdict: NOVEL|COLLAPSED, reason}
    skeptic_verdict.json       — {verdict: ACCEPT|REJECT, reason}
    literature_hits.json       — list of PaperHit dicts (only in breakthrough)
    arbiter_decision.json      — final PlanDict if debate fires
    runner_result.json         — ExperimentResult JSON
    reviewer_notes.json        — reviewer prose + parseable next-step bullets

Payloads are kept small (≤ a few hundred tokens) on purpose. Subagents
should write paths to large artifacts (plots, parquet) rather than the
artifacts themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_INBOX_DIRNAME = "agent_inbox"


def inbox_root(project_dir: Path) -> Path:
    return Path(project_dir) / "memory" / _INBOX_DIRNAME


def iter_dir(project_dir: Path, iteration: int) -> Path:
    return inbox_root(project_dir) / f"iter_{iteration:04d}"


def path_for(project_dir: Path, iteration: int, name: str) -> Path:
    """Return the canonical path for a message of name `name` at `iteration`.
    Creates parent dirs lazily — does not touch the file."""
    d = iter_dir(project_dir, iteration)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


def write(project_dir: Path, iteration: int, name: str, payload: Any) -> Path:
    """Atomically write a JSON payload to <inbox>/iter_<N>/<name>.json.

    `payload` may be any JSON-serializable structure or any object exposing
    `.model_dump()` (pydantic) — the latter is serialized via that method."""
    p = path_for(project_dir, iteration, name)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()  # type: ignore[attr-defined]
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def read(project_dir: Path, iteration: int, name: str) -> Any:
    """Read and JSON-parse <inbox>/iter_<N>/<name>.json. Raises
    FileNotFoundError if absent — callers should handle that explicitly to
    surface ordering bugs."""
    p = path_for(project_dir, iteration, name)
    if not p.exists():
        raise FileNotFoundError(f"agent_inbox message missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def exists(project_dir: Path, iteration: int, name: str) -> bool:
    return path_for(project_dir, iteration, name).exists()


def list_messages(project_dir: Path, iteration: int) -> list[str]:
    """Return the bare message names currently present at <iteration>."""
    d = iter_dir(project_dir, iteration)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
