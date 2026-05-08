"""Synthesis at every 10 iterations + vision checkpoint.

Selects 2 plots for vision review (residuals-vs-fitted of the current
best + a capability-specific diagnostic), produces `synthesis_NNN.md`,
updates `COURSE.md`, and adds reviewer-authored sketch annotations.

The reviewer role (Opus, vision-enabled) consumes this scaffold. The
scaffold is constructed deterministically here; the reviewer adds prose.

Pillar 9 — reviewer prose binds the next batch. The reviewer SKILL/agent
must include a "What to try next" section. Each bullet there parses into
a `source="reviewer_directive"` hypothesis that gets first-class priority
in the next call to `lib.generate_hypotheses.generate(...)`.

Parseable forms (agreed with the reviewer agent):

    - area=<name> family=<name>: rationale...
    - try: <model_key>: rationale...
    - try: feature <token>: rationale...

Free-form bullets are persisted with `area="features"` and the bullet
text as `rationale` so reviewer prose is never silently dropped.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from lib.capabilities import validate_composition
from lib.schemas.experiment import ExperimentResult
from lib.schemas.mission import Mission
from lib.schemas.sketch import SketchAnnotation
from lib.sketch.annotations import append_annotation
from lib.state import read_experiments

_log = logging.getLogger("eda.synthesize")


def _pick_best(experiments: list[ExperimentResult], direction: str) -> Optional[ExperimentResult]:
    valid = [
        e
        for e in experiments
        if e.skeptic.verdict != "FAIL" and e.primary_metric_value is not None
    ]
    if not valid:
        return None
    if direction == ">=":
        return max(valid, key=lambda e: e.primary_metric_value)  # type: ignore[arg-type,return-value]
    return min(valid, key=lambda e: e.primary_metric_value)  # type: ignore[arg-type,return-value]


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
        v = r.get("primary_metric_value")
        v_str = f"{v:.4f}" if v is not None else "n/a"
        lines.append(
            f"- `{r['id']}` iter={r['iteration']} model={r['model']} area={r['area']} "
            f"value={v_str} verdict={r['verdict']}"
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
        # Pillar 9: parse "What to try next" bullets into reviewer directives
        # so the next iteration's researcher gets explicit, prioritized,
        # parseable next-step hypotheses.
        try:
            parse_and_persist_reviewer_notes(project_dir, iteration, reviewer_notes)
        except Exception as e:  # noqa: BLE001 — never fail synthesis on directive parsing
            _log.debug("reviewer-directive parsing failed: %s", e)
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


# --- Reviewer-directive parser (Pillar 9) -------------------------------


_NEXT_HEADER_RX = re.compile(
    r"^\s*##+\s*(what\s+to\s+try\s+next|next\s+steps|recommended\s+next\s+moves)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
_BULLET_RX = re.compile(r"^\s*[-*+]\s+(.*)$")
_AREA_FAMILY_RX = re.compile(
    r"area\s*=\s*([a-z_]+)\s+family\s*=\s*([a-z_]+)\s*[:\-]?\s*(.*)$",
    flags=re.IGNORECASE,
)
_TRY_MODEL_RX = re.compile(
    r"^try\s*:\s*([a-z_][a-z0-9_]*)\s*[:\-]?\s*(.*)$",
    flags=re.IGNORECASE,
)
_TRY_FEATURE_RX = re.compile(
    r"^try\s*:\s*feature\s+([\w:+]+)\s*[:\-]?\s*(.*)$",
    flags=re.IGNORECASE,
)


def _extract_next_section(prose: str) -> str:
    """Pull the body of the 'What to try next' section. Returns '' if not found."""
    m = _NEXT_HEADER_RX.search(prose)
    if not m:
        return ""
    rest = prose[m.end():]
    # Stop at the next markdown header.
    next_h = re.search(r"^\s*##+\s+", rest, flags=re.MULTILINE)
    if next_h:
        return rest[: next_h.start()]
    return rest


def _parse_directive_bullets(section_body: str) -> list[dict[str, Any]]:
    """Tokenize bullet lines into structured directives. Each returned dict
    has at least: `area`, `family` (or None), `model_hint`, `features_dsl`,
    `rationale`."""
    out: list[dict[str, Any]] = []
    for line in section_body.splitlines():
        m = _BULLET_RX.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue
        # try: feature <token>
        mf = _TRY_FEATURE_RX.match(body)
        if mf:
            tok, rest = mf.group(1).strip(), mf.group(2).strip()
            # The greedy `[\w:+]+` captures trailing `:` used as a rationale
            # separator (e.g. "engineered:cyclic: hour-based..."). Strip it.
            tok = tok.rstrip(":")
            out.append(
                {
                    "kind": "feature",
                    "area": "features",
                    "family": None,
                    "model_hint": "lgbm_default",
                    "features_dsl": ["+all_allowed", tok],
                    "rationale": rest or body,
                }
            )
            continue
        # try: <model_key>
        mm = _TRY_MODEL_RX.match(body)
        if mm:
            model_key, rest = mm.group(1).strip(), mm.group(2).strip()
            out.append(
                {
                    "kind": "model",
                    "area": "baseline",
                    "family": None,
                    "model_hint": model_key,
                    "features_dsl": ["+all_allowed"],
                    "rationale": rest or body,
                }
            )
            continue
        # area=X family=Y
        ma = _AREA_FAMILY_RX.search(body)
        if ma:
            area, family, rest = ma.group(1).strip().lower(), ma.group(2).strip().lower(), ma.group(3).strip()
            out.append(
                {
                    "kind": "area_family",
                    "area": area,
                    "family": family,
                    "model_hint": "lgbm_default",
                    "features_dsl": ["+all_allowed"],
                    "rationale": rest or body,
                }
            )
            continue
        # Free-form: keep so reviewer prose is never silently dropped.
        out.append(
            {
                "kind": "free",
                "area": "features",
                "family": None,
                "model_hint": "lgbm_default",
                "features_dsl": ["+all_allowed"],
                "rationale": body,
            }
        )
    return out


def _persist_reviewer_directives(
    project_dir: Path,
    iteration: int,
    parsed: list[dict[str, Any]],
) -> Path:
    """Append reviewer directives to memory/HYPOTHESES.jsonl with
    `source="reviewer_directive"`. The generator picks them up first on
    the next iteration and marks them `consumed=True` after selection."""
    p = Path(project_dir) / "memory" / "HYPOTHESES.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(parsed):
        family = d.get("family")
        if family is None:
            # Sensible default by area; researcher can refine.
            family = {
                "interactions": "boosted_tree",
                "regimes": "tree",
                "calibration": "boosted_tree",
                "robustness": "ensemble",
                "leakage_probe": "linear",
                "causal": "linear",
                "ensembling": "ensemble",
                "features": "boosted_tree",
                "baseline": "boosted_tree",
            }.get(d.get("area", "features"), "boosted_tree")
        rows.append(
            {
                "hypothesis_id": f"H-iter{iteration}-reviewer-{i}",
                "name": f"reviewer_directive_iter{iteration}_{i}",
                "summary": d.get("rationale", "")[:400],
                "technique_family": family,
                "area": d.get("area", "features"),
                "model_hint": d.get("model_hint", "lgbm_default"),
                "features_dsl": list(d.get("features_dsl", ["+all_allowed"])),
                "expected_info_gain": 0.7,
                "rationale": d.get("rationale", "Reviewer directive."),
                "source": "reviewer_directive",
                "kind": d.get("kind"),
                "iteration_emitted": int(iteration),
                "consumed": False,
            }
        )
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def parse_and_persist_reviewer_notes(
    project_dir: Path,
    iteration: int,
    reviewer_notes: str,
) -> list[dict[str, Any]]:
    """Public entry-point used by `write_synthesis(...)`. Returns the list
    of directives written (possibly empty)."""
    if not reviewer_notes:
        return []
    section = _extract_next_section(reviewer_notes)
    if not section:
        return []
    parsed = _parse_directive_bullets(section)
    if not parsed:
        return []
    _persist_reviewer_directives(project_dir, iteration, parsed)
    return parsed


def mark_directive_consumed(project_dir: Path, hypothesis_id: str) -> bool:
    """Flag a reviewer-directive hypothesis as consumed so it isn't picked
    twice. Rewrites HYPOTHESES.jsonl atomically. Returns True if a row was
    flagged."""
    p = Path(project_dir) / "memory" / "HYPOTHESES.jsonl"
    if not p.exists():
        return False
    rows: list[dict[str, Any]] = []
    changed = False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            r.get("source") == "reviewer_directive"
            and r.get("hypothesis_id") == hypothesis_id
            and not r.get("consumed")
        ):
            r["consumed"] = True
            changed = True
        rows.append(r)
    if changed:
        tmp = p.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        tmp.replace(p)
    return changed
