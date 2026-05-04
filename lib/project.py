"""Project lifecycle: create, open, finalize, archive.

Project files are laid out from `projects/.templates/_project_template/`.
Created projects are stamped with the current `lib.__version__` (replay
pin) and a status of `created`.
"""
from __future__ import annotations

import getpass
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from lib import __version__
from lib.schemas.project_meta import ProjectMeta
from lib.workspace import project_path, projects_dir, recipes_dir, resolve_workspace

_log = logging.getLogger("eda.project")

_TEMPLATE_RELATIVE = Path("projects") / ".templates" / "_project_template"


def _write_project_template(target: Path) -> None:
    """Materialize the standard per-project skeleton at `target`."""
    for sub in ("memory", "data", "sketch", "sketch/annotations", "results"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    # Empty append-only files.
    for f in ("experiment_log.jsonl", "budget.jsonl"):
        fp = target / f
        if not fp.exists():
            fp.touch()
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Project\n\n"
            "Created by `eda new-project`. Drop your data files into `data/`,\n"
            "then run `/init`, `/plan`, `/run`, `/contribute` from Claude Code.\n",
            encoding="utf-8",
        )


def create_project(
    name: str,
    domain: str,
    recipe: Optional[str] = None,
    token_budget: int = 100_000,
    iteration_budget: int = 100,
    branch: Optional[str] = None,
    workspace: Optional[Path] = None,
) -> Path:
    """Create a new project under projects/<name>/ and write PROJECT.json.

    Returns the project's absolute path. Refuses to overwrite an existing
    project.
    """
    ws = resolve_workspace(workspace)
    target = project_path(ws, name)
    if target.exists():
        raise FileExistsError(f"project already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    _write_project_template(target)

    if branch is None:
        try:
            user = getpass.getuser() or "anon"
        except Exception:  # noqa: BLE001 — fallback for restricted envs
            user = "anon"
        branch = f"project/{user}/{name}"

    # Validate domain key without importing the domain module fully (to keep
    # this file independent of capability imports during create-time).
    valid_domains = {"general", "manufacturing", "forecasting_demand"}
    if domain not in valid_domains:
        _log.warning("domain '%s' not in built-in registry %s", domain, sorted(valid_domains))

    if recipe is not None:
        rp = recipes_dir(ws) / f"{recipe}.json"
        if not rp.exists():
            _log.warning("recipe '%s' not found at %s — proceeding anyway", recipe, rp)

    meta = ProjectMeta(
        project_name=name,
        domain=domain,
        recipe=recipe,
        branch=branch,
        framework_version_pin=__version__,
        token_budget=token_budget,
        iteration_budget=iteration_budget,
    )
    (target / "PROJECT.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    _log.info("created project %s at %s (domain=%s, recipe=%s)", name, target, domain, recipe)
    return target


def open_project(workspace: Optional[Path], name: str) -> ProjectMeta:
    """Load and return a project's PROJECT.json as a validated ProjectMeta."""
    ws = resolve_workspace(workspace)
    target = project_path(ws, name)
    pj = target / "PROJECT.json"
    if not pj.exists():
        raise FileNotFoundError(f"no PROJECT.json at {pj}")
    return ProjectMeta.model_validate_json(pj.read_text(encoding="utf-8"))


def write_project_meta(workspace: Optional[Path], meta: ProjectMeta) -> Path:
    """Write a PROJECT.json (overwrites)."""
    ws = resolve_workspace(workspace)
    target = project_path(ws, meta.project_name)
    pj = target / "PROJECT.json"
    pj.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return pj


def list_projects(workspace: Path) -> list[dict]:
    """List every project's compact summary row from PROJECT.json files."""
    out: list[dict] = []
    pdir = projects_dir(workspace)
    if not pdir.exists():
        return out
    for child in sorted(pdir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        pj = child / "PROJECT.json"
        if not pj.exists():
            continue
        try:
            meta = ProjectMeta.model_validate_json(pj.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — list shouldn't fail on one bad project
            _log.warning("could not parse %s: %s", pj, e)
            continue
        out.append(
            {
                "project_name": meta.project_name,
                "domain": meta.domain,
                "status": meta.status,
                "confidence_tier": meta.confidence_tier,
                "branch": meta.branch,
            }
        )
    return out


def project_status(workspace: Path, project_name: str) -> dict:
    """One project's status, plus a count of experiment log rows + budget."""
    target = project_path(workspace, project_name)
    pj = target / "PROJECT.json"
    if not pj.exists():
        raise FileNotFoundError(f"no PROJECT.json at {pj}")
    meta = ProjectMeta.model_validate_json(pj.read_text(encoding="utf-8"))
    n_exp = 0
    log_path = target / "experiment_log.jsonl"
    if log_path.exists():
        n_exp = sum(1 for _ in log_path.read_text(encoding="utf-8").splitlines() if _.strip())
    budget_used = 0
    bp = target / "budget.jsonl"
    if bp.exists():
        for line in bp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                budget_used = max(budget_used, int(row.get("cumulative_total", 0)))
            except json.JSONDecodeError:
                continue
    return {
        "project_name": meta.project_name,
        "domain": meta.domain,
        "recipe": meta.recipe,
        "status": meta.status,
        "confidence_tier": meta.confidence_tier,
        "branch": meta.branch,
        "framework_version_pin": meta.framework_version_pin,
        "token_budget": meta.token_budget,
        "iteration_budget": meta.iteration_budget,
        "n_experiments": n_exp,
        "tokens_used": budget_used,
        "fraction_consumed": (budget_used / meta.token_budget) if meta.token_budget else 0.0,
    }


def archive_project(workspace: Path, project_name: str) -> Path:
    """Move project under projects/.archive/<project_name>."""
    target = project_path(workspace, project_name)
    if not target.exists():
        raise FileNotFoundError(target)
    archive_root = workspace / "projects" / ".archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / project_name
    if dest.exists():
        raise FileExistsError(dest)
    shutil.move(str(target), str(dest))
    return dest
