"""Sketch manifest read/write helpers."""
from __future__ import annotations

import json
from pathlib import Path

from lib.schemas.sketch import SketchManifest

_MANIFEST_RELATIVE = Path("sketch") / "manifest.json"


def manifest_path(project_dir: Path) -> Path:
    """Project-relative path to sketch/manifest.json."""
    return Path(project_dir) / _MANIFEST_RELATIVE


def save_manifest(project_dir: Path, manifest: SketchManifest) -> Path:
    """Persist manifest.json (committed)."""
    p = manifest_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_manifest(project_dir: Path) -> SketchManifest:
    """Load manifest.json or raise FileNotFoundError."""
    p = manifest_path(project_dir)
    if not p.exists():
        raise FileNotFoundError(p)
    return SketchManifest.model_validate_json(p.read_text(encoding="utf-8"))
