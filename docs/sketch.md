# The Process Data Sketch

A compact (<1MB structural), deterministically-built summary of every
dataset. The agent queries it via the MCP tool surface; it never reads
raw data directly.

## Layers

| Layer | Content                                                              | Build trigger | Update trigger                  |
|-------|----------------------------------------------------------------------|---------------|---------------------------------|
| L1    | Per-column distribution: quantiles + cardinality (HLL) + top categories | bootstrap     | (rebuilt only on next bootstrap) |
| L2    | Top-k PCA + sparse top-K interactions (mutual information)           | bootstrap     | `lib.sketch.updaters` on `area=interactions` is best |
| L3    | Regimes: change-points + per-regime mini-summary                     | bootstrap     | `queued_for_resegmentation` on high-info `area=regimes` |
| L4    | Per-capability importance-weighted coresets                          | bootstrap     | (rebuilt only on next bootstrap) |
| L5    | Per time-series column SAX + matrix profile motifs/discords          | bootstrap     | (rebuilt only on next bootstrap) |
| L6    | PC-algorithm DAG hints (partial-correlation tests)                   | bootstrap     | (rebuilt only on next bootstrap) |
| L7    | Failure-mode online cluster catalog (Mahalanobis + Welford)          | empty at bootstrap | every WARN/FAIL skeptic verdict |
| Annot. | LLM-written annotations (kind: regime_label, failure_cluster_label, …) | n/a       | reviewer at every 10 iters       |

L1, L2, L3, L5, L6 are *structural*: rebuilt by `lib.sketch.builder` and
never modified mid-run. L7 is online-updated per experiment. Annotations
are commentary that the structural updaters never read.

## MCP tool surface

`mcp_servers/sketch_server.py` exposes:

- Reads: `quantile`, `distribution`, `cardinality`, `missingness`,
  `top_interactions`, `conditional_dependence`, `principal_components`,
  `regimes`, `regime_compare`, `motifs`, `discords`, `causal_neighbors`,
  `confounder_candidates`, `failure_clusters`, `match_residuals`,
  `fit_quick`, `cross_validate_quick`.
- Writes (deterministic, called by `lib.state.record`):
  `update_failure_catalog`, `update_interactions`, `refine_regimes`.

## Determinism

`build_sketch(..., seed=N)` is deterministic given (data, seed, capability
list). Re-running on the same data with the same seed produces a
bit-identical manifest (modulo float-printing).

## Size budget

Target: <1MB total binary size for L1..L7 (excluding L4 coresets, which
are sample parquet files sized by capability and dataset).
