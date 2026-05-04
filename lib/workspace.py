"""Workspace root resolution and project registry helpers.

The workspace is a directory containing `projects/`, `knowledge/`,
`recipes/`, etc. Most commands accept an explicit `--workspace` flag; if
omitted, the current working directory is walked upward looking for a
marker (`pyproject.toml` or a `.eda_workspace` sentinel file).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger("eda.workspace")

_MARKERS = ("pyproject.toml", ".eda_workspace")


def resolve_workspace(explicit: Optional[Path] = None, *, start: Optional[Path] = None) -> Path:
    """Return the workspace root directory.

    Precedence:
        1. ``explicit`` argument if given (and exists).
        2. Walk up from ``start`` (default: cwd) looking for a marker.
        3. Fall back to ``start`` itself.
    """
    if explicit is not None:
        ep = Path(explicit).resolve()
        if not ep.exists():
            raise FileNotFoundError(f"workspace path does not exist: {ep}")
        return ep
    base = (start or Path.cwd()).resolve()
    cur = base
    while True:
        for m in _MARKERS:
            if (cur / m).exists():
                return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    _log.debug("workspace marker not found from %s; using %s", base, base)
    return base


def projects_dir(workspace: Path) -> Path:
    """Return ``<workspace>/projects``, creating it if missing."""
    p = workspace / "projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def knowledge_dir(workspace: Path) -> Path:
    """Return ``<workspace>/knowledge``, creating it if missing."""
    p = workspace / "knowledge"
    p.mkdir(parents=True, exist_ok=True)
    return p


def recipes_dir(workspace: Path) -> Path:
    """Return ``<workspace>/recipes``."""
    return workspace / "recipes"


def project_path(workspace: Path, project_name: str) -> Path:
    """Return the directory for a named project."""
    return projects_dir(workspace) / project_name
