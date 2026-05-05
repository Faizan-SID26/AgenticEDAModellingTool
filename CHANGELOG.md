# Changelog

All notable changes to the EDA Framework. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-05-04

### Added

- Initial v1 build.
- Core schemas (`lib/schemas/`): MISSION, plan dict, experiment row, sketch
  manifest, recommendation, knowledge entries, project metadata, adaptive
  question, budget ledger.
- Capability modules: tabular_classification, tabular_regression,
  temporal_classification, forecasting, predictive_maintenance,
  anomaly_detection, root_cause_attribution.
- Domain modules: general, manufacturing, forecasting_demand.
- Recipes for the 7 capabilities × representative domains.
- Process Data Sketch (L1 distributions, L2 joint, L3 regimes, L4 coresets,
  L5 timeseries, L6 causal hints, L7 failure modes; separate annotations).
- 4-step iteration loop (state.next → researcher → run → state.record),
  with hypothesis generation every 5 iters and synthesis + vision
  checkpoint every 10.
- Autonomous `/run` with `RUN_STATE.json` resumability.
- Cross-project knowledge: post-merge extractor + retrieval MCP server.
- Deterministic replay (`lib.replay`).
- CLI: `eda new-project`, `list`, `status`, `library`, `replay`.

### Known limitations

- Some capability default-models are simple sklearn baselines; LightGBM is
  enabled but tuning ranges are conservative.
- Vision checkpoint integration depends on Claude Code multimodal support
  being available in the harness; otherwise text-only fallback is used.
