# Multi-agent dispatch order for `/run`

This directory holds the subagents the orchestrator dispatches.
`.claude/skills/*/SKILL.md` files remain the source of truth for the
*content* of each role (so prose lives in one place); the agent files
here add the multi-agent + breakthrough-mode behavior on top.

## Roster

| Agent | Model | Fires | Output destination |
|---|---|---|---|
| `planner` | sonnet | `/plan` only | `MISSION.json` (lock_project) |
| `researcher` | sonnet | every iter | `agent_inbox/iter_<N>/researcher_proposal.json` |
| `novelty-check` | haiku | every iter (post-researcher) | `agent_inbox/iter_<N>/novelty_verdict.json` |
| `skeptic` | haiku | every iter (post-novelty) | `agent_inbox/iter_<N>/skeptic_verdict.json` |
| `debate-arbiter` | sonnet | only on `skeptic REJECT + breakthrough mode` | `agent_inbox/iter_<N>/arbiter_decision.json` |
| `runner` | haiku | every iter (post-skeptic / post-arbiter) | `agent_inbox/iter_<N>/runner_result.json` |
| `literature` | haiku | only when `brief.breakthrough_mode_active` | `agent_inbox/iter_<N>/literature_hits.json` |
| `reviewer` | sonnet (vision) | every 10 iters | `results/synthesis_<N>.md` + `memory/HYPOTHESES.jsonl` (reviewer-directive) |
| `analyst` | sonnet | `/finalize` (or after re-entry exhaustion) | `results/FINAL.md` + `results/knowledge_bundle.json` |

## Per-iteration flow

```
brief = lib.state.next(...)

# Pre-iteration extras when breakthrough mode is active.
if brief.breakthrough_mode_active:
    Agent(literature, ...)             # writes literature_hits.json

Agent(researcher, ...)                 # writes researcher_proposal.json
Agent(novelty-check, ...)              # writes novelty_verdict.json
if novelty == COLLAPSED:
    re-prompt researcher with recent_fingerprints in validation_context

Agent(skeptic, ...)                    # writes skeptic_verdict.json
if skeptic == REJECT:
    if brief.breakthrough_mode_active:
        Agent(debate-arbiter, ...)     # writes arbiter_decision.json
    else:
        re-prompt researcher

# Final plan dict for this iteration: arbiter's if it ran, else researcher's.
plan = read("arbiter_decision.json" or "researcher_proposal.json")

Agent(runner, ...)                     # writes runner_result.json
lib.state.record(..., plan=plan)       # persists fingerprint to RECENT_PLANS.jsonl

if (iter % 10 == 0):
    Agent(reviewer, ...)               # writes synthesis_<N>.md + reviewer-directive hypotheses
```

## Token discipline

- Subagents communicate via paths under `memory/agent_inbox/`, never via
  long prose pasted into the dispatch prompt.
- Each subagent's brief is ≤ ~200 tokens; each subagent's output is ≤
  ~400 tokens (debate-arbiter is the worst case).
- The orchestrator never re-feeds an entire transcript — it shows the
  next subagent only the file paths it needs.
- Haiku is the default for cheap, isolated subagents (runner, novelty,
  skeptic, literature). Sonnet only for reasoning depth (researcher,
  reviewer, debate-arbiter, planner, analyst).
