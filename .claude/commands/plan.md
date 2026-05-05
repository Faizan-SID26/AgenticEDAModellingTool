---
description: Adaptive Q&A that locks the MISSION. Conversational, multi-batch, document-aware. Terminates only when MISSION passes consistency checks.
allowed-tools:
  - Read
  - Bash(python:*)
  - AskUserQuestion
---

# /plan — adaptive planning

You are running in the **planner role**. Use the planner skill at
`.claude/skills/planner/SKILL.md` for full procedural guidance.

## Inputs

- `memory/INIT_PROFILE.json` (must exist; if missing, stop and ask user
  to run `/init` first).
- `memory/DOMAIN_DOCS.md` (optional but **read it first** if it exists).
- `PROJECT.json` (recipe + domain + budget pin).
- The recipe at `recipes/<recipe>.json` if `PROJECT.json.recipe` is set.

## Procedure

1. **Read DOMAIN_DOCS.md if present.** Domain documents (PUDs, specs,
   SOPs) are first-class priors. If it exists, internalize the named
   process stages, hard physical bounds, sensor failure modes, and any
   columns the document calls out as "downstream of target" or
   "post-event". You will use these to:
   - Pre-fill / refine the planner's `forbidden_columns` inference,
   - Skip questions the document already answers,
   - Surface the named process for user confirmation.

2. Build the **initial question batch** (basics):

       python -c "from lib.planning import load_init_profile, load_recipe, build_initial_batch; ..."

3. Build **follow-up batches** in order. The planner now walks **three
   to four themed batches**, not just one:

   - **Batch 2 — process knowledge** (`build_followup_batch(..., iteration=1)`):
     process description, expected drivers, leakage pitfalls beyond the
     obvious, time interpretation, expected regimes, lag policy (for
     manufacturing), prior attempts that didn't work.
   - **Batch 3 — project context** (`iteration=2`): business question,
     deployment shape, FP/FN tradeoff (for binary targets), latency /
     interpretability constraints, extra forbidden columns, join
     confirmation.
   - **Batch 4 — domain-doc cross-check** (`iteration=3`, only if
     DOMAIN_DOCS.md exists): single open-ended question asking the user
     to correct anything you misread from the document.

   Loop: build batch → ask via `AskUserQuestion` → record answers →
   build next. Stop when `build_followup_batch` returns an empty batch.

4. **Don't be terse.** This is the one place the user pays attention to
   what you ask. Ask the long questions. Process knowledge questions
   should be *answered in full sentences*, not "yes / no".

5. Call `assemble_mission(...)` then `lock_project(...)`. Both validate
   strictly; if either raises, surface the error to the user, ask the
   focused follow-up that resolves it, and re-run.

6. On success, summarize the locked MISSION in 5-7 bullets:
   - Capability composition (the 5-tuple).
   - Target / time / group columns.
   - Forbidden column count + a sample.
   - Success criterion.
   - Budget cap.
   - **Process knowledge captured** (one line — confirms the deeper
     answers landed in MISSION.notes).
   - **Domain documents read** (count + paths, if any).

7. Tell the user: "Run `/run` to begin autonomous iteration."

## Rules

- **Never modify the MISSION after lock.** Lock once.
- **Never invent columns.** Every column referenced in MISSION must
  appear in INIT_PROFILE.
- **Never skip the lock validation step.** The pydantic models are the
  source of truth.
- **Never skip the deeper batches.** They feed MISSION.notes which the
  researcher consumes on every iteration brief.
- If the user types "skip" or "n/a" on a free-text question, that
  answer is valid — just don't fold an empty answer into MISSION.notes.
- If the user asks something off-topic (e.g., "explain the data"),
  defer with: "I can answer that after MISSION is locked — let's
  finish planning first."
