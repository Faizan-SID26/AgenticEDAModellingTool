# Contributing knowledge

Knowledge enters the framework only by completing a project and merging
the project's branch into `main`. The post-merge extractor (run by CI)
appends extracted entries to `knowledge/`.

## Flow

```
finish project on a branch
  → /finalize         → results/FINAL.md + results/knowledge_bundle.json
  → /contribute       → CONTRIBUTION.md
  → user opens PR     → reviewers approve
  → merge to main     → CI runs tools/post_merge_extractor.py <project>
  → knowledge/ grows  → next project's /plan + hypothesis generator query it
```

## What gets extracted

- Each non-failed experiment with `info_gain_actual >= 0.1` becomes a
  `HypothesisLibraryEntry` in `knowledge/hypothesis_library.jsonl`.
- Each `FAIL` skeptic verdict's `failed_checks` keys become
  `FailureModeEntry` rows in `knowledge/failure_modes.jsonl`.
- The project's `sketch/manifest.json#similarity_vector` is upserted into
  `knowledge/sketch_index.db` for retrieval.

## What does NOT get extracted

- Raw column names. They are anonymized to semantic role tags via
  `lib.extract_knowledge._anonymize_column`.
- Raw data values.
- Plot images.
- The full experiment log (stays in the project).

## Replay invariant

The post-merge extractor is *idempotent* per (project, entry_id). Re-running
it on the same project does not duplicate rows: `entry_id` is constructed
from project + experiment id + check key.
