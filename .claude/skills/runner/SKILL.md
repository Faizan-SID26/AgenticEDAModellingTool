---
name: runner
description: Mechanical execution discipline (used both by the runner sub-agent and by humans debugging the runner).
---

# Runner discipline

The runner is a *mechanical* role. Its only job is to execute one
validated plan dict by calling `lib.run.execute_plan(...)`.

## Discipline

- **No scientific decisions.** No metric interpretation, no follow-up
  proposals.
- **No data reads.** All data interaction goes through `lib.run`, which
  loads the L4 coreset internally.
- **No file modifications** outside what `lib.run` writes (plots and
  artifacts under `results/iter_NNN/`).
- **Strict validation.** If the plan dict fails validation, return the
  error verbatim; do not "fix" the plan dict.
- **Strict reporting.** Output the `ExperimentResult` JSON exactly as
  produced by `lib.run.execute_plan`.

## Why this matters

The agent's scientific reasoning is supposed to be auditable from the
plan dict + experiment row alone. If the runner injected its own logic,
replay would diverge from what the researcher claimed to do.
