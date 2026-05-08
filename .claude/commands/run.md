---
description: Multi-agent autonomous bootstrap → iterate → synthesize → finalize. Each iteration dispatches researcher → novelty-check → skeptic → (debate-arbiter when needed) → runner via Agent. Halts only on goal-met, budget-exhausted, secondary stagnation, iteration cap, or interrupt.
allowed-tools:
  - Read
  - Bash(python:*)
  - Agent
  - AskUserQuestion
  - WebSearch
  - WebFetch
---

# /run — multi-agent execution

You are the **orchestrator**. /run is the single entry point that takes a
locked project (status `planned`) all the way to `completed` (or
`no_signal`). It is autonomous — the user is never required to type
`/continue`.

The orchestrator does **not** reason about plans. It only:

1. computes the iteration brief,
2. dispatches the right subagent,
3. routes structured JSON outputs through `memory/agent_inbox/`,
4. checks termination + the disk-backed doom-loop,
5. (when below operational floor) keeps the loop alive in breakthrough mode.

All scientific reasoning lives in subagents.

## Pre-flight

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

Tell the user: "Bootstrap complete. Beginning multi-agent iteration."

## Phase B — Iteration loop

Repeat until termination_check halts:

### Step 1: brief

```python
from lib.state import next as state_next
brief = state_next(proj, mission)
```

The brief now includes `breakthrough_mode_active`, `iterations_in_breakthrough`,
`operational_floor`, and `below_floor`. Persist it to the iteration inbox so
subagents can read it as a path:

```python
from lib.agent_inbox import write
write(proj, brief.iteration, "brief", brief.to_dict())
```

### Step 1b: literature scout (only in breakthrough mode)

```python
if brief.breakthrough_mode_active:
    Agent(
        subagent_type="literature",
        prompt=f"project_dir={proj} iteration={brief.iteration} "
               f"capability_key={brief.capability_signature.split('|')[0]} "
               f"technique_family={least_explored_family} "
               f"problem_signature={short_signature}",
    )
    # writes memory/agent_inbox/iter_<N>/literature_hits.json
```

### Step 2: researcher

Dispatch:

```python
Agent(
    subagent_type="researcher",
    prompt=(
        f"project_dir={proj} iteration={brief.iteration} "
        f"brief_path=memory/agent_inbox/iter_{brief.iteration:04d}/brief.json "
        + (f"literature_hits_path=memory/agent_inbox/iter_{brief.iteration:04d}/literature_hits.json"
           if brief.breakthrough_mode_active else "")
    ),
)
```

Read the proposal:

```python
from lib.agent_inbox import read
from lib.schemas.plan import PlanDict
from lib.anti_doom import load_recent_fingerprints

proposal = read(proj, brief.iteration, "researcher_proposal")
recent_fps = load_recent_fingerprints(proj, window=3)
plan = PlanDict.model_validate(
    proposal,
    context={
        "recent_fingerprints": recent_fps,
        "breakthrough_mode_active": brief.breakthrough_mode_active,
    },
)
```

If validation raises (collision or missing domain_prior), re-dispatch the
researcher with a corrective brief — do NOT silently fix the plan.

### Step 3: novelty-check + skeptic

```python
Agent(subagent_type="novelty-check", prompt=f"project_dir={proj} iteration={brief.iteration} window=3")
Agent(subagent_type="skeptic", prompt=f"project_dir={proj} iteration={brief.iteration} breakthrough_mode_active={brief.breakthrough_mode_active}")
novelty = read(proj, brief.iteration, "novelty_verdict")
skeptic = read(proj, brief.iteration, "skeptic_verdict")
```

Routing:

- If `novelty.verdict == "COLLAPSED"` → re-dispatch the researcher with
  the same brief but `recent_fingerprints` in the context (the validator
  will reject a second collision — repeat at most 2 times, else fall
  through to `skeptic`'s opinion).
- If `skeptic.verdict == "REJECT"`:
    * outside breakthrough mode → re-dispatch the researcher with the
      skeptic's reason; once.
    * inside breakthrough mode → dispatch `debate-arbiter`.

### Step 3b: debate-arbiter (only when triggered)

```python
if skeptic.get("verdict") == "REJECT" and brief.breakthrough_mode_active:
    Agent(
        subagent_type="debate-arbiter",
        prompt=f"project_dir={proj} iteration={brief.iteration} "
               f"researcher_proposal_path=... skeptic_verdict_path=... "
               f"literature_hits_path=...",
    )
    arbiter = read(proj, brief.iteration, "arbiter_decision")
    plan = PlanDict.model_validate(arbiter, context={"breakthrough_mode_active": True, "recent_fingerprints": recent_fps})
```

### Step 4: runner

```python
Agent(
    subagent_type="runner",
    prompt=f"project_dir={proj} plan_json={plan.model_dump_json()} seed=0",
)
runner_payload = read(proj, brief.iteration, "runner_result")
from lib.schemas.experiment import ExperimentResult
er = ExperimentResult.model_validate(runner_payload)
```

(If the existing runner agent prints the result instead of writing to
the inbox, adapt the orchestrator's invocation to capture stdout and
write it to `runner_result.json` itself — same observable behavior.)

### Step 5: record

```python
from lib.state import record
record(proj, mission, er, plan=plan)
```

`record(...)` now appends the plan fingerprint to
`memory/RECENT_PLANS.jsonl` so the disk-backed doom-loop check has a
persistent trailing window.

### Step 6: synthesis (every 10 iters)

If `brief.iteration % 10 == 0` AND `iteration > 0`:

```python
Agent(
    subagent_type="reviewer",
    prompt=f"project_dir={proj} iteration={brief.iteration}",
)
```

The reviewer calls `lib.synthesize.write_synthesis(...)` itself, which
now also parses "What to try next" bullets into reviewer-directive
hypotheses for the next iteration.

### Step 7: termination + disk-backed doom-loop

```python
from lib.state import termination_check
from lib.doom_loop import check_from_disk

verdict = termination_check(proj, mission)
doom = check_from_disk(proj, window=3)

if doom.fired:
    # Force the next iteration's PlanDict validation to use a non-empty
    # recent_fingerprints — a doom firing makes structural distinctness
    # mandatory next iter.
    pass  # already handled by the validator + load_recent_fingerprints

if verdict.halt:
    break
# `escalation_required` is non-halting: it tells us we just entered
# breakthrough mode. Keep iterating.
```

## Phase C — Finalize

Triggered when termination_check halts. Dispatch the analyst:

```python
Agent(subagent_type="analyst", prompt=f"project_dir={proj}")
```

The analyst calls `lib.finalize.finalize(...)`. Two outcomes:

- `{"final_path": "...", "confidence_tier": "...", ...}` — finalized.
  Tell the user the confidence tier + decision in one sentence; point to
  `results/FINAL.md`. Suggest `/contribute`.
- `{"requested_re_enter_loop": true, "reason": "...", ...}` — finalize
  refused because we're below operational_floor with budget remaining
  and breakthrough re-entry budget left. Re-enter Phase B; the next
  `state.next(...)` call will report `breakthrough_mode_active=True`.

After `breakthrough_max_entries` re-entries OR budget exhaustion, call
`lib.finalize.finalize(proj, mission, force=True)` to write FINAL.md
honestly even if the operational floor was never met.

## Visible progress reporting

Between iterations, output one short line per iteration:

    iter 7 / 100 (12% budget) — area=interactions model=lgbm_focal value=0.34 ✓ skeptic=ACCEPT

After every synthesis (every 10 iters), output a 2-3 line summary.

After finalize, output the confidence tier + path to FINAL.md.

## Interrupt handling

If the user interrupts:

1. Save RUN_STATE.json with `last_completed_phase` set to the most
   recent phase. `lib.state.save_run_state` is atomic.
2. Tell the user: "Interrupted at <phase>. Run `/resume` to continue."

## Constraints

- **Never read raw data.** All data interaction goes through the sketch
  tool surface and through `lib.run.execute_plan` (via the runner).
- **Never modify MISSION.json** during /run.
- **Never lower the budget cap.** If exceeded, halt.
- **Never skip the audit gate.** It is enforced by `lib.run.execute_plan`.
- **Plan dicts must validate** with the appropriate `validation_context`
  (anti-doom + breakthrough-grounding). No silent corrections.
- **Subagents communicate by file paths**, not by long prose pasted into
  dispatch prompts. The orchestrator's per-iteration token cost should
  be roughly: 1 brief (~150 tok) + 5-7 subagent dispatches × ~200 tok
  brief each + their structured outputs.
