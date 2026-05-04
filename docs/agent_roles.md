# Agent roles

A single Claude operating in different roles per slash command and skill
file. State persists via files on disk.

| Role        | Model                | Trigger                     | Reads                              | Writes                                       |
|-------------|----------------------|-----------------------------|------------------------------------|----------------------------------------------|
| Planner     | Opus                 | `/init`, `/plan`            | INIT_PROFILE + recipe + priors     | MISSION.json + memory/COLUMNS.json + JOIN_PLAN.json + HYPOTHESES.jsonl |
| Researcher  | Sonnet (→ Opus)      | iter loop default           | iteration brief + sketch queries   | one plan dict per iteration                  |
| Reviewer    | Opus (vision)        | every 10 iters / synthesis  | plots + experiment log             | synthesis_NNN.md + sketch annotations + COURSE.md |
| Analyst     | Opus                 | `/finalize`                 | full experiment log + sketch       | results/FINAL.md + results/knowledge_bundle.json |
| Runner      | Haiku (sub-agent)    | per plan dict in step 3     | plan dict + sketch handle          | one row in experiment_log.jsonl + plots in results/iter_NNN/ |

## Why distinct roles

Each role is a contract. Splitting them keeps each role's prompt small
and auditable. Replay can re-run the runner deterministically from a
plan dict; replay cannot recover scientific judgment from a researcher
prompt, but it does not need to — the researcher's output (the plan
dict) is the only input replay needs.

## Skill files (the contracts)

- `.claude/skills/planner/SKILL.md`
- `.claude/skills/researcher/SKILL.md`
- `.claude/skills/reviewer/SKILL.md`
- `.claude/skills/analyst/SKILL.md`
- `.claude/skills/runner/SKILL.md`
- `.claude/agents/runner.md` (the sub-agent definition itself)
