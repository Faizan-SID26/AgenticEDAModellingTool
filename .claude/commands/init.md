---
description: Profile every file in data/, extract text from any domain documents, and produce INIT_PROFILE.json + DOMAIN_DOCS.md + an init report. Pure inspection — ask no questions.
allowed-tools:
  - Read
  - Bash(python:*)
---

# /init — inspection only

You are running the **planner role in inspection mode**. Your only job
is to profile every file under `./data/`, extract text from any domain
documents, and emit:

- `memory/INIT_PROFILE.json` — column profile + proposed joins + per-doc summary.
- `memory/DOMAIN_DOCS.md` — extracted text from every PUD / spec / SOP /
  prior-investigation document found under `data/` (`.md`, `.txt`,
  `.pdf`, `.docx`, `.rtf`). **Only created if at least one document is
  present.**
- `results/init_report.md` — human-readable summary.

## Procedure (deterministic)

1. Verify you are inside a project directory: confirm `PROJECT.json`
   exists. If not, stop and tell the user to `cd projects/<name>`.

2. Run the inspection tool:

       python -c "from lib.inspect import inspect_project; from pathlib import Path; inspect_project(Path('.'))"

3. Read back `results/init_report.md` and present a **concise** summary:
   - Number of data files, total rows, column counts.
   - Likely target / time / id columns per file.
   - Proposed joins (if any).
   - **Domain documents found**: their paths + extracted character counts.
     If any parser is missing (PDF/DOCX), surface the install hint.
   - Any data files that failed to read.

4. If `memory/DOMAIN_DOCS.md` exists, also Read it and tell the user
   3-5 of the *most useful facts* you extracted (named process stages,
   hard physical bounds, sensor failure modes, known leakage columns the
   document calls out). The planner will use these at `/plan` time —
   surfacing them now lets the user correct any misreads early.

5. Tell the user: "Run `/plan` next to lock the MISSION."

## Rules

- **Do not ask questions.** This command is purely informational.
- **Do not write Python that touches data.** `lib.inspect` is the only
  data-reading code allowed at this stage.
- **Do not modify MISSION.json or PROJECT.json.** That is `/plan`'s job.
- If `lib.inspect` raises, report the error verbatim and stop.
- If a PDF parser is missing, tell the user the install command (e.g.,
  `pip install pypdf` or `pip install pdfplumber`). Do not stop — the
  rest of the inspection still runs.

## Output

End with one line: `Init complete. Run /plan to continue.`
