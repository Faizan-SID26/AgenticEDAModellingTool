---
description: Resume an interrupted /run from RUN_STATE.json. Reads the last completed phase and re-enters /run from that point.
allowed-tools:
  - Read
  - Bash(python:*)
  - Agent
---

# /resume — pick up an interrupted /run

## Procedure

1. Load `RUN_STATE.json`:

       python -c "from lib.state import load_run_state; from pathlib import Path; print(load_run_state(Path('.').resolve()).to_dict() if hasattr(load_run_state(Path('.').resolve()), 'to_dict') else load_run_state(Path('.').resolve()))"

2. Inspect `last_completed_phase`:
   - `created` or `planned` → re-enter at `/run` Phase A (bootstrap).
   - `bootstrap` → enter Phase B at iteration 1.
   - `iter_N` → enter Phase B at iteration N+1.
   - `synthesis_N` → enter Phase B at iteration N+1 (synthesis already
     written for that block).
   - `finalize` → tell the user the project is already finalized; offer
     to run `/contribute`.

3. Re-read `MISSION.json` and confirm it still validates. If not, stop
   and report.

4. Hand control to `/run` from the resolved phase. Do not re-do the
   bootstrap if it has been completed.

## Constraints

- Never re-write the experiment log; it is append-only.
- Never re-bootstrap when `sketch/manifest.json` already exists.
- If `RUN_STATE.json` is missing entirely, treat it as a fresh `/run`.
