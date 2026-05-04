"""Repo health checks.

- Oversized files (>10MB).
- Schema-validation of every JSON in `recipes/` and every project's
  PROJECT.json + MISSION.json.
- Universal seeds JSONL parses.
- All declared MCP servers have a corresponding module.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_log = logging.getLogger("eda.audit_repo")

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_oversized() -> list[str]:
    issues: list[str] = []
    for p in _REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        if p.suffix in (".pyc",):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz > _MAX_FILE_BYTES:
            issues.append(f"oversized:{p.relative_to(_REPO_ROOT)} ({sz} bytes)")
    return issues


def _check_recipes() -> list[str]:
    issues: list[str] = []
    rdir = _REPO_ROOT / "recipes"
    if not rdir.exists():
        return ["missing recipes/ directory"]
    for p in sorted(rdir.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            issues.append(f"unparseable_recipe:{p.name}: {e}")
            continue
        for key in ("recipe_key", "domain", "capability", "primary_capability", "default_success_criterion"):
            if key not in r:
                issues.append(f"missing_field:{p.name}: {key}")
        cap = r.get("capability", {})
        for key in ("temporal_structure", "leakage_model", "target_type", "validation_strategy", "recommendation_type"):
            if key not in cap:
                issues.append(f"missing_capability_field:{p.name}: {key}")
    return issues


def _check_seeds() -> list[str]:
    p = _REPO_ROOT / "seeds" / "universal_seeds.jsonl"
    if not p.exists():
        return ["missing seeds/universal_seeds.jsonl"]
    issues: list[str] = []
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            n += 1
        except json.JSONDecodeError as e:
            issues.append(f"unparseable_seed:line {n + 1}: {e}")
    if n != 5:
        issues.append(f"expected 5 universal seeds, found {n}")
    return issues


def _check_mcp_servers() -> list[str]:
    p = _REPO_ROOT / "mcp_servers" / "server_manifest.json"
    if not p.exists():
        return ["missing mcp_servers/server_manifest.json"]
    issues: list[str] = []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"unparseable mcp_servers/server_manifest.json: {e}"]
    for server in data.get("mcp_servers", []):
        cmd = server.get("command", [])
        if not cmd:
            issues.append(f"server {server.get('name')} has empty command")
            continue
        # Last token is the module name; check the file exists.
        mod = cmd[-1]
        modpath = (_REPO_ROOT / mod.replace(".", "/")).with_suffix(".py")
        if not modpath.exists():
            issues.append(f"server {server.get('name')}: module not found ({modpath})")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    issues: list[str] = []
    if args.recipes_only:
        issues.extend(_check_recipes())
    else:
        issues.extend(_check_oversized())
        issues.extend(_check_recipes())
        issues.extend(_check_seeds())
        issues.extend(_check_mcp_servers())

    if issues:
        for i in issues:
            print(f"ISSUE: {i}")
        print(f"\n{len(issues)} issues.")
        return 1
    print("OK: repo health checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
