---
description: Autonomous bootstrap → iterate → synthesize → finalize. Exploratory by default, web-search-enabled. Terminates only on goal-met, budget-exhausted, stagnation (after sufficient exploration), catastrophic skeptic failure, iteration cap, or user interrupt.
allowed-tools:
  - Read
  - Bash(python:*)
  - Agent
  - AskUserQuestion
  - WebSearch
  - WebFetch
---

# /run — autonomous execution

You are the **orchestrator**. /run is the single entry point that takes a
locked project (status `planned`) all the way to `completed` (or
`no_signal`). It is autonomous — the user is never required to type
`/continue`.

## Pre-flight (must succeed before iterating)

1. Confirm `MISSION.json` exists and parses.
2. Confirm `memory/HYPOTHESES.jsonl` exists and is non-empty.
3. Confirm `data/` has at least one supported file.

If any pre-flight fails, stop and report the gap to the user.

## Phase A — Bootstrap

If `sketch/manifest.json` already exists, skip bootstrap.

Otherwise:

```python
from pathlib import Path
import json
from lib.schemas.mission import Mission
from lib.data import load_tables, execute_join_plan, write_joined
from lib.sketch.builder import build_sketch
from lib.capabilities import validate_composition

proj = Path('.').resolve()
mission = Mission.model_validate_json((proj / 'MISSION.json').read_text(encoding='utf-8'))
tables = load_tables(proj)
df = execute_join_plan(tables, mission)
write_joined(proj, df)
cap_key = validate_composition(mission.capability).key
build_sketch(proj, df, mission, capability_keys=[cap_key], seed=0)
```

Tell the user: "Bootstrap complete. Beginning iteration."

## Phase B — Iteration loop

Repeat until the termination check halts:

### Step 1: brief

```python
from lib.state import next as state_next, load_run_state
brief = state_next(proj, mission)
```

If `brief.termination_imminent`, prefer hypotheses with high
expected_info_gain and small token cost.

### Step 2: plan

- If `brief.iteration % 5 == 0` (and not 0): replace the researcher's
  open-ended plan-picking with the hypothesis generator:

      from lib.generate_hypotheses import generate, append_to_log
      hyps = generate(proj, mission, iteration=brief.iteration)
      append_to_log(proj, hyps)

  Then pick the highest-`expected_info_gain` hypothesis and translate it
  into a plan dict.

- Otherwise: enter the **researcher** role (use the planner/researcher
  skills file at `.claude/skills/researcher/SKILL.md`). Emit a single
  validated plan dict.

Validate before handing off:

```python
from lib.schemas.plan import PlanDict
plan = PlanDict.model_validate_json(plan_json)
```

### Step 3: execute via runner sub-agent

Spawn the runner sub-agent (Haiku) with the plan JSON + the project dir
+ a seed. It will return the `ExperimentResult` JSON. Parse and validate:

```python
from lib.schemas.experiment import ExperimentResult
er = ExperimentResult.model_validate_json(runner_output)
```

### Step 4: record

```python
from lib.state import record
record(proj, mission, er)
```

This appends to the experiment log, deterministically updates L2/L3/L7,
updates the bandit, and writes a budget ledger entry.

### Step 5: synthesis (every 10 iters)

If `brief.iteration % 10 == 0` AND iteration > 0:

- Enter the **reviewer** role (vision-enabled).
- Build the scaffold:

      from lib.synthesize import build_scaffold, write_synthesis
      scaffold = build_scaffold(proj, mission, iteration=brief.iteration)

- Read the plot images at `scaffold.plots_for_vision_review` with
  `Read` (they are PNGs).
- Write reviewer prose, then:

      write_synthesis(proj, mission, iteration=brief.iteration, reviewer_notes=...)

### Step 6: termination check

```python
from lib.state import termination_check
verdict = termination_check(proj, mission)
if verdict.halt:
    break
```

Also check the doom-loop detector after 3+ iterations:

```python
from lib.doom_loop import check as doom_check
from lib.state import read_experiments
recent_exps = read_experiments(proj)[-3:]
# recent_plans is your in-memory buffer of the last 3 plan dicts.
doom = doom_check(recent_plans, recent_exps, window=3)
if doom.fired:
    # Force a different area / family on the next iteration.
    pass
```

## Phase C — Finalize

Triggered when the loop halts. Enter the **analyst** role and call:

```python
from lib.finalize import finalize
out = finalize(proj, mission)
print(out)
```

Tell the user the confidence tier + decision in one sentence; point to
`results/FINAL.md`. Suggest `/contribute`.

## Visible progress reporting

Between iterations, output one short line per iteration:

    iter 7 / 100 (12% budget) — area=interactions model=lgbm_binary roc_auc=0.81 ✓

After every synthesis (every 10 iters), output a 2-3 line summary.

After finalize, output the confidence tier + path to FINAL.md.

## Interrupt handling

If the user interrupts:

1. Save RUN_STATE.json with `last_completed_phase` set to the most
   recent phase (`bootstrap`, `iter_<n>`, `synthesis_<n>`, or
   `finalize`). `lib.state.save_run_state` is atomic.
2. Tell the user: "Interrupted at <phase>. Run `/resume` to continue."

## Constraints

- **Never read raw data.** All data interaction goes through the sketch
  tool surface and through `lib.run.execute_plan` (via the runner).
- **Never modify MISSION.json** during /run.
- **Never lower the budget cap.** If exceeded, halt.
- **Never skip the audit gate.** It is enforced by `lib.run.execute_plan`.
- **Plan dicts must validate.** No silent corrections.
