"""Iteration state machinery.

`state.next(...)`     — produce a *brief* (capped tokens) for the researcher.
`state.record(...)`   — append experiment + trigger updaters + bandit + budget.
`state.termination_check(...)` — evaluate all stop conditions.
`RUN_STATE.json`      — atomic per-project state for resumability.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lib import __version__
from lib.bandit import BanditState, load as bandit_load, save as bandit_save, posterior_means, update as bandit_update
from lib.budget import current_total, fraction_consumed, record_event
from lib.capabilities import composition_signature, get as get_capability, validate_composition
from lib.doom_loop import check as doom_check
from lib.schemas.experiment import ExperimentResult
from lib.schemas.mission import Mission
from lib.schemas.plan import PlanDict
from lib.sketch.updaters import update_after_experiment

_log = logging.getLogger("eda.state")


_LOG_PATH = "experiment_log.jsonl"
_RUN_STATE_PATH = "RUN_STATE.json"


# --- Run state (resumability) -------------------------------------------


@dataclass
class RunState:
    """Atomic per-project state file (`RUN_STATE.json`).

    Written after every step; replay reads it to resume.
    """

    project_name: str
    framework_version: str = __version__
    current_iteration: int = 0
    current_role: str = "researcher"
    last_completed_phase: str = "created"  # bootstrap|iter_<n>|synthesis_<n>|finalize
    best_primary_metric_value: float = float("-inf")
    best_iteration: int = -1
    iterations_since_improvement: int = 0
    last_regime_split_iteration: int = -100
    notes: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        # JSON cannot serialize -inf as a number; replace with a sentinel.
        if not math.isfinite(d.get("best_primary_metric_value", 0.0)):
            d["best_primary_metric_value"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        d = dict(d)
        v = d.get("best_primary_metric_value")
        if v is None:
            d["best_primary_metric_value"] = float("-inf")
        return cls(**d)


def run_state_path(project_dir: Path) -> Path:
    return Path(project_dir) / _RUN_STATE_PATH


def load_run_state(project_dir: Path) -> RunState:
    p = run_state_path(project_dir)
    if not p.exists():
        return RunState(project_name=Path(project_dir).name)
    return RunState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_run_state(project_dir: Path, state: RunState) -> Path:
    p = run_state_path(project_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


# --- Experiment log -----------------------------------------------------


def log_path(project_dir: Path) -> Path:
    return Path(project_dir) / _LOG_PATH


def append_experiment(project_dir: Path, experiment: ExperimentResult) -> Path:
    p = log_path(project_dir)
    with p.open("a", encoding="utf-8") as f:
        f.write(experiment.model_dump_json() + "\n")
    return p


def read_experiments(project_dir: Path) -> list[ExperimentResult]:
    p = log_path(project_dir)
    if not p.exists():
        return []
    out: list[ExperimentResult] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ExperimentResult.model_validate_json(line))
    return out


# --- Brief construction -------------------------------------------------


@dataclass
class IterationBrief:
    """Capped, JSON-able payload handed to the researcher each iteration."""

    iteration: int
    capability_signature: str
    primary_metric: str
    direction: str
    success_threshold: float
    on_split: str
    best_so_far: Optional[float]
    best_iteration: int
    iterations_since_improvement: int
    last_three_experiments: list[dict]
    bandit_posteriors: dict[str, float]
    budget_fraction_consumed: float
    suggested_sketch_queries: list[str]
    termination_imminent: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _suggested_queries_for(capability_key: str) -> list[str]:
    spec = get_capability(capability_key)
    suggestions = ["distribution(<top_features>)", "top_interactions(top_k=5)"]
    if "L3_regimes" in spec.sketch_extras_needed:
        suggestions.append("regimes()")
    if "L5_timeseries" in spec.sketch_extras_needed:
        suggestions.append("motifs(<column>)")
    if "L6_causal" in spec.sketch_extras_needed:
        suggestions.append("causal_neighbors(<target>)")
    if "L7_failure_modes" in spec.sketch_extras_needed:
        suggestions.append("failure_clusters(top_k=3)")
    return suggestions


def next(project_dir: Path, mission: Mission) -> IterationBrief:
    """Build the iteration brief for the researcher.

    Reads MISSION + RUN_STATE + last 3 experiments + bandit + budget. Does
    NOT write anything to disk.
    """
    state = load_run_state(project_dir)
    cap_key = validate_composition(mission.capability).key
    bandit = bandit_load(project_dir)
    posteriors = posterior_means(bandit)
    last_three = read_experiments(project_dir)[-3:]
    last_three_dicts = [
        {
            "id": e.id,
            "iteration": e.iteration,
            "model": e.model,
            "area": e.area,
            "primary_metric_value": e.primary_metric_value,
            "verdict": e.skeptic.verdict,
            "info_gain_actual": e.info_gain_actual,
        }
        for e in last_three
    ]
    cap = mission.budget.token_cap
    frac = fraction_consumed(project_dir, cap)
    termination_close = (
        frac >= 0.85
        or state.iterations_since_improvement >= mission.budget.stagnation_window - 1
        or state.current_iteration >= mission.budget.iteration_cap - 1
    )
    return IterationBrief(
        iteration=state.current_iteration + 1,
        capability_signature=composition_signature(mission.capability),
        primary_metric=mission.success_criterion.metric,
        direction=mission.success_criterion.direction,
        success_threshold=mission.success_criterion.threshold,
        on_split=mission.success_criterion.on_split,
        best_so_far=(state.best_primary_metric_value if math.isfinite(state.best_primary_metric_value) else None),
        best_iteration=state.best_iteration,
        iterations_since_improvement=state.iterations_since_improvement,
        last_three_experiments=last_three_dicts,
        bandit_posteriors=posteriors,
        budget_fraction_consumed=frac,
        suggested_sketch_queries=_suggested_queries_for(cap_key),
        termination_imminent=termination_close,
    )


# --- Recording ----------------------------------------------------------


def _is_improvement(value: Optional[float], best: float, direction: str) -> bool:
    if value is None or not math.isfinite(value):
        return False
    if not math.isfinite(best):
        return True
    return value > best if direction == ">=" else value < best


def record(
    project_dir: Path,
    mission: Mission,
    experiment: ExperimentResult,
    *,
    tokens: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Step 4 of the loop: append + update sketch + update bandit + log budget."""
    project_dir = Path(project_dir)
    state = load_run_state(project_dir)
    direction = mission.success_criterion.direction
    improved = _is_improvement(experiment.primary_metric_value, state.best_primary_metric_value, direction)
    experiment.is_best_so_far = improved
    if improved:
        # Compute info gain as the magnitude of improvement vs prior best,
        # bounded into [0,1] for the bandit.
        # `improved=True` already implies primary_metric_value is finite.
        prior = state.best_primary_metric_value if math.isfinite(state.best_primary_metric_value) else 0.0
        cur = float(experiment.primary_metric_value)  # type: ignore[arg-type]
        delta = abs(cur - prior)
        experiment.info_gain_actual = float(min(1.0, delta * 5.0))
        state.best_primary_metric_value = cur
        state.best_iteration = int(experiment.iteration)
        state.iterations_since_improvement = 0
    else:
        experiment.info_gain_actual = 0.0
        state.iterations_since_improvement += 1

    append_experiment(project_dir, experiment)

    # Update sketch deterministically.
    sketch_changes = update_after_experiment(
        project_dir,
        experiment,
        state_extras={"last_regime_split_iteration": state.last_regime_split_iteration},
    )
    if sketch_changes.get("l3") == "queued_for_resegmentation":
        state.last_regime_split_iteration = experiment.iteration

    # Update bandit.
    b = bandit_load(project_dir)
    b = bandit_update(b, experiment.technique_family, experiment.info_gain_actual)
    bandit_save(project_dir, b)

    # Budget ledger.
    tokens = tokens or {}
    record_event(
        project_dir,
        iteration=experiment.iteration,
        event="iter_end",
        role="researcher",
        cap=mission.budget.token_cap,
        input_tokens=int(tokens.get("input_tokens", 0)),
        output_tokens=int(tokens.get("output_tokens", 0)),
        cache_read_tokens=int(tokens.get("cache_read_tokens", 0)),
        cache_creation_tokens=int(tokens.get("cache_creation_tokens", 0)),
    )

    # Bump RUN_STATE.
    state.current_iteration = max(state.current_iteration, experiment.iteration)
    state.last_completed_phase = f"iter_{experiment.iteration}"
    save_run_state(project_dir, state)
    return {
        "improved": improved,
        "best_so_far": state.best_primary_metric_value,
        "iterations_since_improvement": state.iterations_since_improvement,
        "sketch_updates": sketch_changes,
    }


# --- Termination --------------------------------------------------------


@dataclass
class TerminationVerdict:
    halt: bool
    reasons: list[str] = field(default_factory=list)


_MIN_DISTINCT_FAMILIES_BEFORE_STAGNATION = 4
"""Anti-premature-convergence guard: stagnation alone never halts /run
until the project has actually tried at least this many distinct
technique families. The intent is to force the researcher to *explore*
before declaring 'no further improvement possible'."""


def termination_check(project_dir: Path, mission: Mission) -> TerminationVerdict:
    """Evaluate every stop condition and return the verdict.

    Stagnation is suppressed when the project has not yet tried at least
    `_MIN_DISTINCT_FAMILIES_BEFORE_STAGNATION` distinct technique
    families: 'we tried 3 boosted-tree configs and gave up' is not a
    valid project end state.
    """
    state = load_run_state(project_dir)
    reasons: list[str] = []
    exps = read_experiments(project_dir)

    # Goal met.
    sc = mission.success_criterion
    bv = state.best_primary_metric_value
    if math.isfinite(bv):
        ok = (bv >= sc.threshold) if sc.direction == ">=" else (bv <= sc.threshold)
        if ok:
            reasons.append("goal_met")

    # Budget.
    frac = fraction_consumed(project_dir, mission.budget.token_cap)
    if frac >= 1.0:
        reasons.append("budget_exhausted")

    # Stagnation — only fires if we've actually explored.
    if state.iterations_since_improvement >= mission.budget.stagnation_window:
        distinct_families = {
            e.technique_family for e in exps if e.skeptic.verdict != "FAIL"
        }
        if len(distinct_families) >= _MIN_DISTINCT_FAMILIES_BEFORE_STAGNATION:
            reasons.append("stagnation")
        else:
            # Otherwise, this is *premature* stagnation. Don't halt.
            # The orchestrator should escalate to wildcard / SOTA hypotheses.
            _log.info(
                "stagnation suppressed: only %d distinct families tried (min %d required)",
                len(distinct_families),
                _MIN_DISTINCT_FAMILIES_BEFORE_STAGNATION,
            )

    # Iteration cap.
    if state.current_iteration >= mission.budget.iteration_cap:
        reasons.append("iteration_cap")

    # Catastrophic skeptic failure: same FAIL skeptic key in the trailing N experiments.
    n_window = mission.budget.catastrophic_failure_window
    if len(exps) >= n_window:
        recent_fails = [e.skeptic.failed_checks for e in exps[-n_window:] if e.skeptic.verdict == "FAIL"]
        if len(recent_fails) == n_window:
            shared = set(recent_fails[0])
            for r in recent_fails[1:]:
                shared &= set(r)
            if shared:
                reasons.append(f"catastrophic_skeptic:{sorted(shared)}")

    return TerminationVerdict(halt=bool(reasons), reasons=reasons)
