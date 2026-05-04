---
description: Stage everything for a knowledge-merging PR. Writes CONTRIBUTION.md and reminds the user of the git commands.
allowed-tools:
  - Read
  - Bash(python:*)
---

# /contribute — prepare the merge PR

## Procedure

1. Confirm `results/FINAL.md` and `results/knowledge_bundle.json` exist.
   If not, stop and tell the user to run `/run` (which auto-finalizes).

2. Generate the contribution scaffold:

       python -c "
       from pathlib import Path
       from lib.contribute import prepare
       proj = Path('.').resolve()
       print(prepare(proj.name))
       "

3. Read back `CONTRIBUTION.md` and present a concise summary:
   - Confidence tier.
   - Number of hypothesis + failure entries to be merged.
   - Branch name + the files that will be committed.

4. Tell the user the exact git commands to run:

       git add PROJECT.json MISSION.json memory/ experiment_log.jsonl \
               budget.jsonl sketch/manifest.json sketch/annotations \
               results/synthesis_*.md results/FINAL.md results/knowledge_bundle.json \
               CONTRIBUTION.md
       git commit -m "<project>: contribute knowledge from completed run"
       git push origin <branch>

   And: open a PR to `main`. After merge, CI runs
   `tools/post_merge_extractor.py` to append to `knowledge/`.

## Constraints

- Do NOT run `git` commands yourself. The user must review + approve.
- Do NOT modify `knowledge/` directly — the post-merge extractor owns it.
