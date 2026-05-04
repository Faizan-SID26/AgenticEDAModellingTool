---
description: Profile every file in data/ and produce INIT_PROFILE.json + an init report. Pure inspection — ask no questions.
allowed-tools:
  - Read
  - Bash(python:*)
---

# /init — inspection only

You are running the **planner role in inspection mode**. Your only job is
to profile every file under `./data/` and emit two artifacts:

- `memory/INIT_PROFILE.json` — the structured profile.
- `results/init_report.md` — the human-readable summary.

## Procedure (deterministic)

1. Verify you are inside a project directory: confirm `PROJECT.json` exists.
   If not, stop and tell the user to `cd projects/<name>`.

2. Run the inspection tool:

       python -c "from lib.inspect import inspect_project; from pathlib import Path; inspect_project(Path('.'))"

3. Read back `results/init_report.md` and present a **concise** summary:
   - Number of files, total rows, column counts.
   - Likely target / time / id columns per file.
   - Proposed joins (if any).
   - Any files that failed to read.

4. Tell the user: "Run `/plan` next to lock the MISSION."

## Rules

- **Do not ask questions.** This command is purely informational.
- **Do not write Python that touches data.** `lib.inspect` is the only
  data-reading code allowed at this stage.
- **Do not modify MISSION.json or PROJECT.json.** That is `/plan`'s job.
- If `lib.inspect` raises, report the error verbatim and stop.

## Output

End with one line: `Init complete. Run /plan to continue.`
