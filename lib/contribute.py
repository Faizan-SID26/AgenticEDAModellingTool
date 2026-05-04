"""`/contribute`: prepare a contribution PR.

Stages everything the post-merge extractor will pick up:
    results/FINAL.md
    results/knowledge_bundle.json
    sketch/manifest.json
    sketch/annotations/*.jsonl
    memory/HYPOTHESES.jsonl (final state)

Writes a `CONTRIBUTION.md` summarizing what was learned.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from lib.project import open_project
from lib.schemas.knowledge import KnowledgeBundle
from lib.workspace import project_path, resolve_workspace

_log = logging.getLogger("eda.contribute")


def prepare(project_name: str, *, workspace: Optional[Path] = None) -> dict[str, str]:
    """Generate `CONTRIBUTION.md` summarizing the contribution.

    Does NOT touch git — the user runs the actual git/PR commands.
    """
    ws = resolve_workspace(workspace)
    proj = project_path(ws, project_name)
    if not proj.exists():
        raise FileNotFoundError(proj)
    meta = open_project(workspace, project_name)
    bundle_path = proj / "results" / "knowledge_bundle.json"
    final_path = proj / "results" / "FINAL.md"

    if not bundle_path.exists() or not final_path.exists():
        raise FileNotFoundError(
            f"missing FINAL.md or knowledge_bundle.json — run /finalize first ({proj})"
        )

    bundle = KnowledgeBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"# Contribution from project `{project_name}`\n")
    lines.append(f"_Domain_: `{meta.domain}` — _confidence_: **{meta.confidence_tier}**\n")
    lines.append(f"_Branch_: `{meta.branch}`\n")
    lines.append(f"_Capability signature_: `{bundle.capability_signature}`\n")
    lines.append(
        f"\n## Knowledge to be merged\n\n"
        f"- {len(bundle.hypothesis_entries)} hypothesis entries\n"
        f"- {len(bundle.failure_entries)} failure-mode entries\n"
        f"- 1 sketch similarity vector\n"
    )
    lines.append("\n## Files to commit\n")
    for rel in (
        "PROJECT.json",
        "MISSION.json",
        "memory/HYPOTHESES.jsonl",
        "memory/COURSE.md",
        "memory/COLUMNS.json",
        "memory/JOIN_PLAN.json",
        "experiment_log.jsonl",
        "budget.jsonl",
        "sketch/manifest.json",
        "sketch/annotations",
        "results/synthesis_*.md",
        "results/FINAL.md",
        "results/knowledge_bundle.json",
    ):
        lines.append(f"- `{rel}`")
    lines.append(
        "\n## Instructions\n\n"
        "1. Commit the listed files on the project branch.\n"
        "2. Open a PR to `main`.\n"
        "3. CI runs `tools/post_merge_extractor.py` after merge — it appends to `knowledge/`\n"
        "   and updates the sketch index. No manual edits to `knowledge/` required.\n"
    )
    out = proj / "CONTRIBUTION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    _log.info("wrote contribution scaffold at %s", out)
    return {"contribution_path": str(out), "branch": meta.branch}
