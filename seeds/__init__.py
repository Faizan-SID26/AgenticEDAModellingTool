"""Universal hypothesis seeds (5 of them).

Loaded by `lib.lock` to populate every project's `memory/HYPOTHESES.jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SEED_PATH = Path(__file__).resolve().parent / "universal_seeds.jsonl"


def load_universal_seeds() -> list[dict[str, Any]]:
    """Return the 5 universal seed hypotheses as plain dicts."""
    if not _SEED_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in _SEED_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


__all__ = ["load_universal_seeds"]
