# Architecture

## Principles

1. **Vertical slices, not horizontal layers.** End-to-end correctness on one
   capability before adding another. Manufacturing-defect (binary
   classification with regime structure) is the v1 reference path.
2. **Determinism is sacred.** Every random source seeded; every dependency
   pinned in `requirements.txt`; framework version stamped on every artifact;
   sketch updates after experiments are deterministic Python (only the
   separate annotations layer is LLM-written).
3. **Schemas are the source of truth.** Every artifact is a pydantic v2
   model with a `schema_version` field. Every read validates; every write
   produces validated output.
4. **The agent reads structured state, writes structured outputs.** It never
   reads raw data and never writes Python that touches data. All data
   interaction goes through the sketch tool surface (MCP server). Plan
   execution is delegated to the Haiku runner sub-agent.
5. **Skill files are the agent's brain.** Each `.claude/commands/*.md` and
   `.claude/skills/*/SKILL.md` is a contract for role behavior: identity,
   procedure, output schema, available tools, constraints, next-step
   recommendation.
6. **Capability composition, not problem-type dispatch.** Modules check
   capabilities (`temporal_structure`, `leakage_model`, `target_type`,
   `validation_strategy`, `recommendation_type`), never dispatch on a single
   "problem type" enum.
7. **Honest failure is a valid outcome.** Reporting "no signal found" with
   evidence is shippable.
8. **The Process Data Sketch is the substrate.** Built once at `/bootstrap`,
   queried by the agent via tools, updated deterministically after every
   iteration. Layers L1–L7 are structural and Python-updated; a separate
   annotations layer is LLM-written at vision checkpoints. Total size <1MB
   for a 10GB / 1000-column dataset.
9. **Planning is manual; execution is autonomous.** `/plan` is a
   conversational adaptive question loop with the user. `/run` is fully
   autonomous from bootstrap through finalize.
10. **One repo, branches for projects, merge-to-main = enter organizational
    knowledge.** Project work happens on team branches. Completed projects
    raise PRs to main. CI extracts knowledge artifacts post-merge into
    `knowledge/` which future projects retrieve from.

## Agent roles

A single Claude (model picked per role) operating in different roles
governed by slash commands and skill files. State persists via files on disk
(MISSION.json, RUN_STATE.json, experiment_log.jsonl, etc.).

| Role        | Model                | Trigger                           | Input                              | Output                              |
|-------------|----------------------|-----------------------------------|------------------------------------|-------------------------------------|
| Planner     | Opus                 | `/plan`                           | INIT_PROFILE + recipe + priors     | MISSION.json, JOIN_PLAN, HYPOTHESES |
| Researcher  | Sonnet (→ Opus)      | iter loop (default)               | iteration brief + sketch queries   | plan dict                           |
| Reviewer    | Opus (vision)        | every 10 iters / on synthesis     | plots + experiment log             | synthesis_NNN.md, COURSE.md updates |
| Analyst     | Opus                 | `/finalize`                       | full experiment log + sketch       | FINAL.md (counterfactual rec.)      |
| Runner      | Haiku (sub-agent)    | per plan dict in iter step 3      | plan dict + sketch handle          | experiment result row + plots       |

The researcher escalates Sonnet → Opus on iterations flagged as
high-info-gain by the bandit + hypothesis generator.

## The 4-step iteration loop

1. `lib.state.next` — produce iteration brief (capped tokens) from
   MISSION + sketch + last 3 experiments + bandit posteriors + budget
   remaining.
2. **Researcher** — reads brief, queries sketch as needed, emits a plan
   dict that *must* contain `prior_evidence` referencing a sketch query
   result or prior experiment.
3. `lib.run` — executes plan via Haiku runner: load coreset, expand
   features, audit leakage, fit, score, run skeptic, save plots.
4. `lib.state.record` — append to experiment log; deterministically update
   sketch L2/L3/L7; update bandit posteriors; write budget ledger entry.

Inside the loop:
- Every 5 iterations: replace step 2 with the **hypothesis generator**.
- Every 10 iterations: also run **synthesis** + **vision checkpoint**
  (Reviewer role).

## Termination conditions for `/run`

- Goal met (MISSION.success_criterion evaluator returns true).
- Budget exhausted (≥ 100% of cap).
- Stagnation (N consecutive iterations no improvement).
- Catastrophic skeptic failure (same severe failure repeatedly).
- Iteration cap reached (default 100).
- User interrupt.

## The 5 universal seed hypotheses

1. Naive baseline on all allowed features.
2. Univariate champion using top-3 columns by predictive strength.
3. Regime-specific submodels scored on out-of-regime data.
4. Interaction-augmented baseline with top-5 interactions from L2.
5. Leakage probe (deliberately includes a downstream column to establish
   ceiling).

## Sketch tool surface (MCP)

Read: `quantile`, `distribution`, `cardinality`, `missingness`,
`top_interactions`, `conditional_dependence`, `principal_components`,
`regimes`, `regime_compare`, `motifs`, `discords`, `causal_neighbors`,
`confounder_candidates`, `failure_clusters`, `match_residuals`,
`fit_quick`, `cross_validate_quick`.

Write (deterministic, called by `lib.state.record`): `update_failure_catalog`,
`update_interactions`, `refine_regimes`.

## Plan dict required fields

`id`, `iteration`, `hypothesis_id`, `model`, `features` (DSL with `+all_allowed`,
`+lag_downstream`, `engineered:GROUP`), `params`, `calibrate`, `prior_evidence`
(mandatory; references sketch query result or prior experiment), `technique_family`,
`area`, `expected_info_gain`.

## Capability composition fields

| Field                 | Values                                                                |
|-----------------------|-----------------------------------------------------------------------|
| `temporal_structure`  | `regime_based`, `seasonal`, `none`                                    |
| `leakage_model`       | `stage_frontier`, `forecast_horizon`, `none`                          |
| `target_type`         | `binary`, `continuous`, `time_to_event`, `multi_horizon`, `rank`, `outlier_score` |
| `validation_strategy` | `time_split`, `rolling_origin`, `group_kfold`, `stratified`           |
| `recommendation_type` | `decision`, `forecast`, `ranked_factors`, `alert_policy`              |

Validators in `lib.schemas.mission` enforce composition consistency
(e.g., `target_type=time_to_event` requires `temporal_structure != none`).

## Per-project on-disk layout

```
projects/<name>/
├── PROJECT.json                  # status, budget, framework version, confidence_tier
├── MISSION.json                  # locked after /plan
├── memory/                       # INIT_PROFILE.json, COLUMNS.json, JOIN_PLAN.json, HYPOTHESES.jsonl, COURSE.md
├── data/                         # raw user files (gitignored)
├── sketch/                       # L1..L7 binaries (gitignored), manifest.json (committed), annotations/ (committed)
├── results/                      # iter_NNN/ (gitignored), synthesis_NNN.md (committed), FINAL.md (committed)
├── experiment_log.jsonl          # append-only (committed)
├── budget.jsonl                  # token ledger (committed)
├── RUN_STATE.json                # /run resumability
└── README.md                     # per-project notes
```

## Cross-project knowledge

Lives in `knowledge/`. Populated *post-merge* by
`tools/post_merge_extractor.py` (run by CI). Reads the merged project's
experiment log + sketch annotations, anonymizes column names to semantic
roles via the domain module, and appends to:

- `knowledge/hypothesis_library.jsonl`
- `knowledge/failure_modes.jsonl`
- `knowledge/domain_learnings/<domain>.jsonl`
- `knowledge/sketch_index.db` (regenerable SQLite)

`lib/retrieval.py` queries the index when seeding new projects.

## Replay

`lib.replay` reads `experiment_log.jsonl` plus the framework version pinned
in PROJECT.json and reproduces all artifacts deterministically.
