# `knowledge/`

Cross-project knowledge accumulated by the framework. Populated **only**
by the post-merge extractor (`tools/post_merge_extractor.py`) — never by
hand.

## Files

- `hypothesis_library.jsonl` — successful patterns extracted from
  completed projects. Each row is a `HypothesisLibraryEntry`
  (`lib.schemas.knowledge.HypothesisLibraryEntry`). Column names are
  anonymized to *semantic role tags* via the source domain module
  (e.g., `temp_zone_a` → `<sensor:temperature>`).

- `failure_modes.jsonl` — cross-project failure mode catalog. Each row
  is a `FailureModeEntry`.

- `domain_learnings/<domain>.jsonl` — per-domain priors that strengthen
  with use. Same row schema as `hypothesis_library.jsonl`, filtered to
  one domain.

- `sketch_index.db` — SQLite of compact sketch fingerprints + project
  metadata. Used by `lib.retrieval.query_similar_projects`. **Regenerable**
  from `hypothesis_library.jsonl` + project sketch manifests, so it is
  gitignored.

## How knowledge grows

```
project finishes
  → /finalize writes results/knowledge_bundle.json
  → /contribute writes CONTRIBUTION.md
  → user opens PR to main
  → CI merges
  → CI runs tools/post_merge_extractor.py <project>
  → appends rows to knowledge/, upserts sketch_index.db
```

## How knowledge is queried

- `eda library` (CLI) — summary counts.
- `lib.retrieval.query_similar_projects(...)` — find similar past projects
  by sketch cosine similarity.
- `lib.retrieval.query_hypotheses(...)` — filter by domain / capability /
  min-info-gain.
- The retrieval MCP server (`mcp_servers/retrieval_server.py`) exposes
  the same surface to the agent at `/plan` and at hypothesis-generation
  steps.

## Anonymization invariants

- Raw column names never enter `knowledge/`. Anonymization happens at
  extraction time via `lib.extract_knowledge._anonymize_column`.
- No raw values, no row counts that could fingerprint a dataset, no
  free-form domain-specific identifiers.
