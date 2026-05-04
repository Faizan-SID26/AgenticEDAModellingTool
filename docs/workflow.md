# Workflow reference

## Phases

| Phase         | Trigger     | Role          | Reads                                  | Writes                                                  |
|---------------|-------------|---------------|----------------------------------------|---------------------------------------------------------|
| Inspection    | `/init`     | Planner       | `data/*`                               | `memory/INIT_PROFILE.json`, `results/init_report.md`    |
| Planning      | `/plan`     | Planner       | INIT_PROFILE, recipe, domain priors    | `MISSION.json`, `memory/COLUMNS.json`, `JOIN_PLAN.json`, `HYPOTHESES.jsonl` |
| Bootstrap     | `/run` Phase A | Orchestrator | MISSION, data                          | `sketch/manifest.json`, `sketch/L*.{json,parquet,jsonl}` |
| Iterate       | `/run` Phase B | Researcher → Runner → Reviewer (every 10) | sketch + experiment log + bandit | one row in `experiment_log.jsonl`, plots in `results/iter_NNN/` |
| Finalize      | `/run` Phase C | Analyst    | full experiment log + sketch           | `results/FINAL.md`, `results/knowledge_bundle.json`     |
| Contribute    | `/contribute` | Planner     | FINAL + bundle                         | `CONTRIBUTION.md`                                       |
| Post-merge    | CI          | (deterministic Python) | merged project's bundle      | `knowledge/hypothesis_library.jsonl`, `knowledge/failure_modes.jsonl`, `knowledge/sketch_index.db` |

## Termination conditions for `/run`

- **Goal met** — MISSION.success_criterion satisfied on `on_split`.
- **Budget exhausted** — `cumulative_total >= budget.token_cap`.
- **Stagnation** — `iterations_since_improvement >= budget.stagnation_window`.
- **Catastrophic skeptic failure** — same skeptic key fails for
  `budget.catastrophic_failure_window` consecutive iterations.
- **Iteration cap** — `current_iteration >= budget.iteration_cap`.
- **User interrupt** — RUN_STATE preserved; `/resume` picks up.

## What gets committed per project

- `PROJECT.json`
- `MISSION.json`
- `memory/*` (INIT_PROFILE, COLUMNS, JOIN_PLAN, HYPOTHESES, COURSE.md, BANDIT)
- `experiment_log.jsonl` (append-only)
- `budget.jsonl` (append-only)
- `sketch/manifest.json` and `sketch/annotations/`
- `results/synthesis_NNN.md` and `results/FINAL.md`
- `results/knowledge_bundle.json`

What is **not** committed (regenerable):
- `data/` — raw user files
- `sketch/L*.{json,parquet,jsonl}` — built from data
- `sketch/raw_joined.parquet` — built from data
- `results/iter_NNN/` — per-iteration plots and artifacts

## Resumability

Every step writes `RUN_STATE.json` atomically. `/resume` reads it and
re-enters `/run` at the right phase. Replay (`eda replay <project>`) reads
the experiment log + framework version pin and reproduces all artifacts.
