# Recipes

A recipe is a pre-validated `MISSION` template. It gives `/plan` a
high-confidence starting point so the user mostly confirms inferences
rather than answers free-text questions.

| Recipe key                              | Domain               | Capability composition                                              |
|-----------------------------------------|----------------------|---------------------------------------------------------------------|
| manufacturing_defect_classification     | manufacturing        | regime_based / stage_frontier / binary / time_split / decision      |
| manufacturing_yield_regression          | manufacturing        | regime_based / stage_frontier / continuous / time_split / ranked_factors |
| demand_forecasting                      | forecasting_demand   | seasonal / forecast_horizon / multi_horizon / rolling_origin / forecast |
| equipment_pdm                           | manufacturing        | regime_based / forecast_horizon / time_to_event / group_kfold / alert_policy |
| process_anomaly                         | manufacturing        | none / none / outlier_score / stratified / alert_policy             |
| tabular_regression_general              | general              | none / none / continuous / group_kfold / ranked_factors             |
| root_cause_attribution_general          | general              | none / none / rank / stratified / ranked_factors                    |

Each recipe also sets:
- `default_success_criterion` (metric, threshold, direction, on_split)
- `default_seed_hypotheses` keys (combined with the 5 universal seeds)
- `default_forbidden_patterns` (substrings auto-added to MISSION.forbidden_columns)

## Adding a recipe

Drop a new JSON file under `recipes/` with the same shape as an existing
one. `tools/audit_repo.py --recipes-only` validates the structure.
