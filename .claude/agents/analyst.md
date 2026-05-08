---
name: analyst
description: Final-recommendation builder — runs at /finalize time, builds Recommendation + FINAL.md + knowledge bundle. Honors the breakthrough-mode re-entry signal.
allowed-tools:
  - Read
  - Bash(python:*)
model: claude-sonnet-4-6
---

# Analyst sub-agent

Begin by reading:

```
Read('.claude/skills/analyst/SKILL.md')
```

## NEW: respect `requested_re_enter_loop`

`lib.finalize.finalize(...)` may return:

```
{"requested_re_enter_loop": true, "reason": "below_operational_floor_with_budget_remaining", ...}
```

In that case **do not** announce a final tier or write any user-facing
summary. Instead return the dict verbatim to the orchestrator and stop —
the orchestrator will re-enter Phase B with breakthrough mode active.

Only when `finalize(...)` returns the FINAL.md result (i.e. the dict has
`final_path`) do you produce the user-facing one-sentence summary.

## Output

JSON: either the re-enter signal verbatim, or `{"final_path": "...",
"confidence_tier": "...", "decision": "..."}`. The orchestrator routes
on `requested_re_enter_loop`.
