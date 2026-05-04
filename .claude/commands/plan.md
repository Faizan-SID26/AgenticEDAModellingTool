---
description: Adaptive Q&A that locks the MISSION. Conversational with the user; terminates only when MISSION passes consistency checks.
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
- `PROJECT.json` (recipe + domain + budget pin).
- The recipe at `recipes/<recipe>.json` if `PROJECT.json.recipe` is set.

## Procedure

1. Build the initial question batch by running:

       python -c "from lib.planning import load_init_profile, load_recipe, build_initial_batch; ..."

   (See the planner skill for the exact one-liner; the skill spells it
   out.)

2. **Ask the questions to the user.** Use `AskUserQuestion` for one
   batch at a time. For `confirm_inference` questions, present the
   inferred answer prominently — the user just confirms or corrects.

3. After each batch, run `build_followup_batch` until no more questions
   remain.

4. Call `assemble_mission(...)` then `lock_project(...)`. Both validate
   strictly; if either raises, surface the error to the user, ask the
   focused follow-up that resolves it, and re-run.

5. On success, summarize the locked MISSION in 5 bullets:
   - Capability composition (the 5-tuple).
   - Target / time / group columns.
   - Forbidden column count.
   - Success criterion.
   - Budget cap.

6. Tell the user: "Run `/run` to begin autonomous iteration."

## Rules

- **Never modify the MISSION after lock.** Lock once.
- **Never invent columns.** Every column referenced in MISSION must
  appear in INIT_PROFILE.
- **Never skip the lock validation step.** The pydantic models are the
  source of truth.
- If the user asks something off-topic (e.g., "explain the data"), defer
  with: "I can answer that after MISSION is locked — let's finish
  planning first."
