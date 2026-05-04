"""Schema migration helpers.

When `lib.SCHEMA_VERSION` bumps, write a migration here. v1 has only one
schema epoch (`"1"`); this module is the placeholder for future migrations.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

from lib import SCHEMA_VERSION

_log = logging.getLogger("eda.migrate_schema")


_MIGRATIONS: dict[tuple[str, str], Callable[[dict], dict]] = {
    # ("1", "2"): migrate_1_to_2,
}


def migrate_artifact(artifact_path: Path, target_version: str = SCHEMA_VERSION) -> bool:
    """Walk migrations from current → target. Returns True if changed."""
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    cur = raw.get("schema_version", "1")
    if cur == target_version:
        return False
    while cur != target_version:
        step = _MIGRATIONS.get((cur, _next_version(cur)))
        if step is None:
            raise NotImplementedError(f"no migration registered: {cur} → {target_version}")
        raw = step(raw)
        cur = raw.get("schema_version", _next_version(cur))
    artifact_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return True


def _next_version(v: str) -> str:
    try:
        return str(int(v) + 1)
    except ValueError:
        return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    args = p.parse_args(argv)
    changed = migrate_artifact(Path(args.path))
    print(f"changed={changed} target={SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
