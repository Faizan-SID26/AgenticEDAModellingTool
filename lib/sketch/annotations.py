"""LLM-written sketch annotations.

Stored under `<project>/sketch/annotations/<kind>.jsonl`. Structural
updaters in `lib.sketch.updaters` never read these files. Annotations
are *commentary* — they augment but never modify the structural layers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from lib.schemas.sketch import SketchAnnotation


def annotation_dir(project_dir: Path) -> Path:
    return Path(project_dir) / "sketch" / "annotations"


def append_annotation(project_dir: Path, ann: SketchAnnotation) -> Path:
    """Append `ann` to the corresponding JSONL file."""
    d = annotation_dir(project_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{ann.kind}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(ann.model_dump_json() + "\n")
    return p


def list_annotations(project_dir: Path, kind: str) -> list[SketchAnnotation]:
    p = annotation_dir(project_dir) / f"{kind}.jsonl"
    if not p.exists():
        return []
    out: list[SketchAnnotation] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(SketchAnnotation.model_validate_json(line))
    return out
