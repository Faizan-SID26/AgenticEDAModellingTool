# Build notes — v1

Recording the state of the v1 build, what was validated, and the small
list of items that are intentionally simple in v1.

## Validation matrix

| Step                                     | Result                                           |
|------------------------------------------|--------------------------------------------------|
| `pip install -e .` (deps available)      | Imports clean. Heavy deps are optional with sane fallbacks (lightgbm → GradientBoosting; stumpy → mean-window; dowhy → bootstrap regression; lifelines → ridge). |
| `pre-commit install && run --all-files`  | Hook config in place; not exercised in the build environment (no pre-commit binary installed locally). Run after `pip install -e .[dev]`. |
| `pytest tests/unit/`                     | **104 / 104 pass** (incl. eval suites collected from `tests/eval_suites/*.py` by path). |
| `pytest tests/integration/`              | **9 / 9 pass** (init, planning, bootstrap, iteration, run, replay, cross-project, contribute). All headless; no `requires_agent` marker needed. |
| `eda new-project demo_validation ...`    | Succeeds. Project skeleton matches the spec. |
| `python tools/audit_repo.py`             | OK — no oversized files, all recipes valid, 5 universal seeds, all MCP server modules present. |
| `python tests/fixtures/generate_fixtures.py` | Wrote 4 fixtures under `tests/fixtures/`. |

Total: **113 tests pass (unit + integration + eval suites).**

## Intentional simplifications in v1 (called out so they're easy to revisit)

- **t-digest quantiles → exact numpy quantiles.** L1 stores percentile→value
  pairs computed via `np.quantile`. Same surface; the t-digest's
  ε-bounded streaming property isn't needed because the sketch is built
  in one pass at /bootstrap and is not maintained streaming-style at
  query time. Schema is unchanged.
- **PC algorithm scope.** L6 runs the deletion phase + a v-structure
  orientation pass on the top-N highest-variance numeric columns. It
  does *not* run the full Meek orientation rules. Edges record
  `ci_test_pval` and a `weight = |partial correlation|`.
- **Domain skeptic extras (`physical_bounds_check`,
  `sensor_flatline_check`, etc.)** are recorded as advisory warnings
  ("`domain_extra_skipped:<key>`") because they require raw rows, not
  experiment summaries. Wiring these to the L4 coreset is a low-effort
  follow-up.
- **`fit_quick` / `cross_validate_quick` (sketch query surface).**
  Implemented and registered in the MCP server; they delegate to
  `lib.registry` + `lib.eval`. Not currently exercised by any integration
  test (the iteration loop uses `lib.run.execute_plan` directly), but
  they validate at import.
- **Survival LightGBM.** `lgbm_survival` falls back to `lgbm_regressor`
  predicting time-to-event. Replacing with a native survival objective
  (e.g., `coxnet_survival_analysis`) is additive in `lib.registry`.
- **Vision checkpoint.** The reviewer skill instructs the agent to read
  PNGs via `Read` and write reviewer prose. The synthesis scaffold is
  fully deterministic; only the prose layer requires a multimodal call.
  Without vision the loop still produces a synthesis report (just
  without prose).
- **`+lag_downstream` DSL token** is currently a no-op; the manufacturing
  lag-join machinery is wired through `execute_join_plan` (asof joins),
  but materializing per-stage lagged downstream features at expansion
  time requires a project-wide stage map that the v1 sketch does not
  retain. Left as a follow-up.
- **MCP servers.** `mcp_servers/{sketch,retrieval,budget}_server.py`
  prefer the official `mcp` package and fall back to a line-based
  JSON-RPC stdio protocol if not installed. Both modes are exercisable.

## Files / surfaces fully implemented

- All schemas in `lib/schemas/` (Mission with composition validators,
  PlanDict with prior_evidence, ExperimentResult, SketchManifest,
  Recommendation, KnowledgeBundle, BudgetLedgerEntry, Question, ProjectMeta).
- 7 capability modules + the registry + composition validator.
- 3 domain modules (`general`, `manufacturing`, `forecasting_demand`)
  with Arrhenius / pressure-temperature physics priors.
- 7 recipes covering each capability × representative domain.
- The full Process Data Sketch (L1..L7 + annotations + manifest +
  similarity).
- 4-step iteration loop (state.next → researcher → run → state.record)
  with deterministic L2/L3/L7 updaters.
- Hypothesis generator (cold start → universal seeds; warm start →
  sketch-derived).
- Synthesis (every 10 iters) + scaffold + sketch annotations writer.
- Counterfactual finalize (dowhy if available, else regression with
  bootstrap CI) + Recommendation + FINAL.md renderer.
- Autonomous /run command file + /resume + /status + /init + /plan +
  /contribute.
- Cross-project knowledge: post-merge extractor (column anonymization
  via domain stage keywords + role hints), retrieval MCP server,
  sketch index DB.
- Replay (`eda replay`) — deterministic re-execution with drift report.
- CLI: `eda new-project`, `list`, `status`, `library`, `replay`.
- Repo audit (`tools/audit_repo.py`): file-size, recipe schema, seed
  count, MCP server module presence.
- Fixture generator (`tests/fixtures/generate_fixtures.py`).

## How to run the final validation locally

```bash
pip install -e .[dev]
pre-commit install
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/eval_suites/planner_eval.py tests/eval_suites/researcher_eval.py \
       tests/eval_suites/reviewer_eval.py tests/eval_suites/analyst_eval.py -v
eda new-project demo_validation --domain manufacturing \
    --recipe manufacturing_defect_classification --budget 30
python tools/audit_repo.py
python tests/fixtures/generate_fixtures.py
```
