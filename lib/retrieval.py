"""Cross-project retrieval.

Reads `knowledge/`:
- `knowledge/hypothesis_library.jsonl`
- `knowledge/failure_modes.jsonl`
- `knowledge/sketch_index.db` (SQLite of similarity vectors + project metadata)

Used by the planner / hypothesis generator at /plan and at the 5-iter
generation step.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from lib.schemas.knowledge import FailureModeEntry, HypothesisLibraryEntry
from lib.workspace import knowledge_dir, resolve_workspace

_log = logging.getLogger("eda.retrieval")


def _hypotheses_path(workspace: Path) -> Path:
    return knowledge_dir(workspace) / "hypothesis_library.jsonl"


def _failures_path(workspace: Path) -> Path:
    return knowledge_dir(workspace) / "failure_modes.jsonl"


def _index_path(workspace: Path) -> Path:
    return knowledge_dir(workspace) / "sketch_index.db"


def _ensure_index(p: Path) -> sqlite3.Connection:
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sketch_index (
            project_name TEXT PRIMARY KEY,
            domain TEXT,
            capability_signature TEXT,
            similarity_vector_json TEXT,
            confidence_tier TEXT
        )
        """
    )
    conn.commit()
    return conn


def upsert_sketch_index(
    workspace: Optional[Path],
    *,
    project_name: str,
    domain: str,
    capability_signature: str,
    similarity_vector: list[float],
    confidence_tier: str,
) -> None:
    ws = resolve_workspace(workspace)
    conn = _ensure_index(_index_path(ws))
    conn.execute(
        "INSERT OR REPLACE INTO sketch_index VALUES (?, ?, ?, ?, ?)",
        (
            project_name,
            domain,
            capability_signature,
            json.dumps(similarity_vector),
            confidence_tier,
        ),
    )
    conn.commit()
    conn.close()


def list_sketch_index(workspace: Optional[Path]) -> list[dict[str, Any]]:
    ws = resolve_workspace(workspace)
    p = _index_path(ws)
    if not p.exists():
        return []
    conn = sqlite3.connect(p)
    rows = conn.execute(
        "SELECT project_name, domain, capability_signature, similarity_vector_json, confidence_tier FROM sketch_index"
    ).fetchall()
    conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "project_name": r[0],
                "domain": r[1],
                "capability_signature": r[2],
                "similarity_vector": json.loads(r[3]),
                "confidence_tier": r[4],
            }
        )
    return out


def query_similar_projects(
    workspace: Optional[Path],
    similarity_vector: list[float],
    *,
    domain: Optional[str] = None,
    capability_signature: Optional[str] = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-k most similar past projects by cosine similarity.

    Confidence tier is used as a tie-breaker (high > medium > low).
    """
    from lib.sketch.similarity import cosine

    rows = list_sketch_index(workspace)
    if domain:
        rows = [r for r in rows if r["domain"] == domain]
    if capability_signature:
        rows = [r for r in rows if r["capability_signature"] == capability_signature]
    tier_rank = {"high": 3, "medium": 2, "low": 1, "no_signal": 0}
    scored: list[tuple[float, dict]] = []
    for r in rows:
        sim = cosine(similarity_vector, r["similarity_vector"])
        scored.append((sim + 0.01 * tier_rank.get(r["confidence_tier"], 0), r))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    out = []
    for sim, r in scored[:top_k]:
        out.append({**r, "score": float(sim)})
    return out


def load_hypothesis_library(workspace: Optional[Path]) -> list[HypothesisLibraryEntry]:
    ws = resolve_workspace(workspace)
    p = _hypotheses_path(ws)
    if not p.exists():
        return []
    out: list[HypothesisLibraryEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(HypothesisLibraryEntry.model_validate_json(line))
        except Exception as e:  # noqa: BLE001
            _log.debug("could not parse hypothesis row: %s", e)
    return out


def load_failure_modes(workspace: Optional[Path]) -> list[FailureModeEntry]:
    ws = resolve_workspace(workspace)
    p = _failures_path(ws)
    if not p.exists():
        return []
    out: list[FailureModeEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(FailureModeEntry.model_validate_json(line))
        except Exception as e:  # noqa: BLE001
            _log.debug("could not parse failure row: %s", e)
    return out


def query_hypotheses(
    workspace: Optional[Path],
    *,
    domain: Optional[str] = None,
    capability_signature: Optional[str] = None,
    min_info_gain: float = 0.1,
    top_k: int = 10,
) -> list[HypothesisLibraryEntry]:
    rows = load_hypothesis_library(workspace)
    if domain:
        rows = [r for r in rows if r.domain == domain]
    if capability_signature:
        rows = [r for r in rows if r.capability_signature == capability_signature]
    rows = [r for r in rows if r.info_gain >= min_info_gain]
    rows.sort(key=lambda r: r.info_gain, reverse=True)
    return rows[:top_k]


def summarize_library(
    *,
    workspace: Optional[Path] = None,
    domain: Optional[str] = None,
    capability: Optional[str] = None,
) -> dict[str, Any]:
    """Compact summary used by `eda library` CLI."""
    h = load_hypothesis_library(workspace)
    f = load_failure_modes(workspace)
    if domain:
        h = [r for r in h if r.domain == domain]
        f = [r for r in f if r.domain == domain]
    if capability:
        h = [r for r in h if r.capability_signature == capability]
        f = [r for r in f if r.capability_signature == capability]
    return {
        "n_hypotheses": len(h),
        "n_failure_modes": len(f),
        "top_hypothesis_pattern": (h[0].pattern_summary if h else None),
        "domains": sorted({r.domain for r in h} | {r.domain for r in f}),
        "capability_signatures": sorted({r.capability_signature for r in h} | {r.capability_signature for r in f}),
    }
