"""Token ledger, allocation, projection.

`budget.jsonl` is append-only. Each entry is a `BudgetLedgerEntry`.
`record_event(...)` writes the next entry, computing the running
`cumulative_total` and `fraction_consumed` against the project's cap.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from lib.schemas.budget import BudgetLedgerEntry

_log = logging.getLogger("eda.budget")


def ledger_path(project_dir: Path) -> Path:
    return Path(project_dir) / "budget.jsonl"


def _read_running_total(p: Path) -> tuple[int, int]:
    """Return (sequence_to_use, current_running_total)."""
    if not p.exists():
        return 0, 0
    last_seq = -1
    last_total = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        last_seq = max(last_seq, int(row.get("sequence", -1)))
        last_total = max(last_total, int(row.get("cumulative_total", 0)))
    return last_seq + 1, last_total


def record_event(
    project_dir: Path,
    *,
    iteration: int,
    event: str,
    role: str,
    cap: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    notes: Optional[str] = None,
) -> BudgetLedgerEntry:
    """Append a ledger row and return the validated entry."""
    p = ledger_path(project_dir)
    seq, running = _read_running_total(p)
    total = input_tokens + output_tokens + cache_read_tokens + cache_creation_tokens
    new_running = running + total
    fraction = (new_running / cap) if cap > 0 else 0.0
    entry = BudgetLedgerEntry(
        sequence=seq,
        iteration=iteration,
        event=event,  # type: ignore[arg-type]
        role=role,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        total_tokens=total,
        cumulative_total=new_running,
        cap=cap,
        fraction_consumed=fraction,
        notes=notes,
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
    return entry


def current_total(project_dir: Path) -> int:
    """Return the most recent cumulative_total."""
    _, total = _read_running_total(ledger_path(project_dir))
    return total


def fraction_consumed(project_dir: Path, cap: int) -> float:
    if cap <= 0:
        return 0.0
    return current_total(project_dir) / cap


def project_remaining_iterations(project_dir: Path, cap: int, avg_tokens_per_iter: int) -> int:
    """Crude projection: how many iterations fit in the remaining budget."""
    remaining = cap - current_total(project_dir)
    if avg_tokens_per_iter <= 0:
        return 0
    return max(0, remaining // avg_tokens_per_iter)
