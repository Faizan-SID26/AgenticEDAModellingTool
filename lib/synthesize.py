"""Synthesis at every 10 iterations + vision checkpoint.

Selects 2 plots for vision review (residuals-vs-fitted of the current
best + a capability-specific diagnostic), produces `synthesis_NNN.md`,
updates `COURSE.md`, and adds reviewer-authored sketch annotations.

The reviewer role (Opus, vision-enabled) consumes this scaffold. The
scaffold is constructed deterministically here; the reviewer adds prose.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from lib.capabilities import validate_composition
from lib.schemas.experiment import ExperimentResult
from lib.schemas.mission import Mission
from lib.schemas.sketch import SketchAnnotation
from lib.sketch.annotations import append_annotation
from lib.state import read_experiments

_log = logging.getLogger("eda.synthesize")


def _pick_best(experiments: list[ExperimentResult], direction: str) -> Optional[ExperimentResult]:
    valid = [
        e for e in experiments if e.skeptic.verdict != "FAIL" and e.primary_metric_value == e.primary_metric_value
    ]
    if not valid:
        return None
    if direction == ">=":
        return max(valid, key=lambda e: e.primary_metric_value)
    return min(valid, key=lambda e: e.primary_metric_value)


def _select_plots(project_dir: Path, best: ExperimentResult) -> list[Path]:
    """Return up to 2 plot paths for vision review."""
    if not best.plot_paths:
        return []
    out = [project_dir / p for p in best.plot_paths]
    return out[:2]


def synthesis_path(project_dir: Path, iteration: int) -> Path:
    return Path(project_dir) / "results" / f"synthesis_{iteration:04d}.md"


def course_path(project_dir: Path) -> Path:
    return Path(project_dir) / "memory" / "COURSE.md"


def build_scaffold(
    project_dir: Path,
    mission: Mission,
    iteration: int,
) -> dict[str, object]:
    """Return a scaffold dict the reviewer (vision-enabled) fills in.

    The scaffold contains:
        - mission summary
        - capability key + signature
        - best experiment so far + selected plot paths
        - last-3 experiments
        - bandit posteriors snapshot
    Used by `.claude/skills/reviewer/SKILL.md`.
    """
    from lib.bandit import load as bandit_load, posterior_means

    cap_key = validate_composition(mission.capability).key
    experiments = read_experiments(project_dir)
    best = _pick_best(experiments, mission.success_criterion.direction)
    plots = _select_plots(project_dir, best) if best else []
    recent = [
        {
            "id": e.id,
            "iteration": e.iteration,
            "model": e.model,
            "area": e.area,
            "primary_metric_value": e.primary_metric_value,
            "verdict": e.skeptic.verdict,
            "info_gain_actual": e.info_gain_actual,
        }
        for e in experiments[-5:]
    ]
    posteriors = posterior_means(bandit_load(project_dir))
    scaffold = {
        "iteration": iteration,
        "capability_key": cap_key,
        "primary_metric": mission.success_criterion.metric,
        "direction": mission.success_criterion.direction,
        "threshold": mission.success_criterion.threshold,
        "best_so_far": (best.primary_metric_value if best else None),
        "best_iteration": (best.iteration if best else None),
        "plots_for_vision_review": [str(p.relative_to(project_dir)) for p in plots],
        "recent_experiments": recent,
        "bandit_posteriors": posteriors,
    }
    return scaffold


def render_synthesis_md(scaffold: dict[str, object], reviewer_notes: str = "") -> str:
    """Pretty-print the synthesis report (with optional reviewer notes)."""
    lines: list[str] = []
    lines.append(f"# Synthesis at iteration {scaffold['iteration']}\n")
    lines.append(f"_Capability_: `{scaffold['capability_key']}`")
    lines.append(
        f"_Primary metric_: `{scaffold['primary_metric']}` (target {scaffold['direction']} {scaffold['threshold']})"
    )
    if scaffold.get("best_so_far") is not None:
        lines.append(f"_Best so far_: `{scaffold['best_so_far']:.4f}` at iteration {scaffold.get('best_iteration')}")
    else:
        lines.append("_Best so far_: (none)")
    lines.append("\n## Plots reviewed")
    for p in scaffold.get("plots_for_vision_review", []):
        lines.append(f"- `{p}`")
    lines.append("\n## Recent experiments")
    for r in scaffold.get("recent_experiments", []):
        lines.append(
            f"- `{r['id']}` iter={r['iteration']} model={r['model']} area={r['area']} "
            f"value={r['primary_metric_value']:.4f} verdict={r['verdict']}"
        )
    lines.append("\n## Bandit posteriors")
    for k, v in sorted(scaffold.get("bandit_posteriors", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v:.3f}")
    if reviewer_notes:
        lines.append("\n## Reviewer notes\n")
        lines.append(reviewer_notes.strip() + "\n")
    return "\n".join(lines) + "\n"


def write_synthesis(
    project_dir: Path,
    mission: Mission,
    iteration: int,
    reviewer_notes: str = "",
) -> Path:
    """Write `results/synthesis_NNN.md` and append a reviewer annotation."""
    scaffold = build_scaffold(project_dir, mission, iteration)
    md = render_synthesis_md(scaffold, reviewer_notes=reviewer_notes)
    p = synthesis_path(project_dir, iteration)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    if reviewer_notes:
        append_annotation(
            project_dir,
            SketchAnnotation(
                kind="general_observation",
                target_id=f"synthesis_{iteration}",
                iteration=iteration,
                text=reviewer_notes[:2000],
                author_role="reviewer",
            ),
        )
    return p


def append_to_course(project_dir: Path, line: str) -> Path:
    """Append a one-line update to memory/COURSE.md."""
    p = course_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# Course log\n\nProject narrative, in order:\n\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- {line.strip()}\n")
    return p
