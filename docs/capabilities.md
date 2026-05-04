# Capabilities

Each capability is a module under `lib/capabilities/`. The framework
dispatches on individual fields of the `CapabilityComposition` 5-tuple —
not on a single problem-type label.

| Key                          | Target                  | Temporal       | Validation     | Recommendation     |
|------------------------------|-------------------------|----------------|----------------|--------------------|
| `tabular_classification`     | `binary`                | `none`         | `stratified`   | `decision`         |
| `tabular_regression`         | `continuous`            | `none`         | `group_kfold`  | `ranked_factors`   |
| `temporal_classification`    | `binary`                | `regime_based` | `time_split`   | `decision`         |
| `forecasting`                | `multi_horizon`         | `seasonal`     | `rolling_origin` | `forecast`       |
| `predictive_maintenance`     | `time_to_event`         | `regime_based` | `group_kfold`  | `alert_policy`     |
| `anomaly_detection`          | `outlier_score`         | `none`         | `stratified`   | `alert_policy`     |
| `root_cause_attribution`     | `rank`                  | `none`         | `stratified`   | `ranked_factors`   |

## When to use which

- **`tabular_classification`** — binary-target IID data, no time order
  matters.
- **`tabular_regression`** — continuous-target IID data; grouped CV is
  the default to avoid entity-level leakage.
- **`temporal_classification`** — binary target on time-ordered data with
  regime structure (manufacturing's reference shape).
- **`forecasting`** — predict at horizon h ≥ 1 with rolling-origin
  validation. `multi_horizon` means the target is the (per-horizon)
  vector, not a single value.
- **`predictive_maintenance`** — time-to-event per entity. Requires
  `group_column` (asset id).
- **`anomaly_detection`** — unsupervised; outputs an outlier score.
  Optionally semi-supervised when partial labels exist.
- **`root_cause_attribution`** — given a known defect, rank candidate
  causes by attribution. NDCG is the primary metric.

## Adding a new capability

See [`adding_a_capability.md`](adding_a_capability.md).
