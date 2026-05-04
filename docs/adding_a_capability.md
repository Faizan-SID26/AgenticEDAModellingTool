# Adding a capability

A capability module declares what the framework needs to handle a
particular shape of ML problem. Modules dispatch on the
`CapabilityComposition` 5-tuple, never on a single problem-type label.

## Step 1 — file

Create `lib/capabilities/<your_key>.py`. Use one of the existing modules
(e.g., `tabular_classification.py`) as a reference.

## Step 2 — `SPEC`

Construct a `CapabilitySpec` with:

- `key` — stable identifier.
- `description` — one sentence.
- `composition` — the *default* composition this capability assumes.
- `required_mission_fields` — dotted-path attribute names the MISSION
  must populate (`target_column`, `time_column`, `success_criterion`, …).
- `default_models` — registry keys (`lib.registry`) for the iteration
  loop's starter models.
- `default_metrics` — metric names from `lib.eval`.
- `primary_metric` — the metric that drives best-tracking and the
  success criterion.
- `primary_metric_direction` — `">="` or `"<="`.
- `sketch_extras_needed` — list of sketch layers / queries this
  capability specifically benefits from.
- `seed_hypothesis_recipe_keys` — keys merged into seed hypotheses.

## Step 3 — splitter

Implement `make_splitter()` returning a `ValidationSplitter` that
produces `(train_idx, val_idx, optional_test_idx)` triples.

## Step 4 — register

Add the module path to `lib/capabilities/__init__._CAPABILITY_MODULES`.

## Step 5 — eval + registry

If the capability uses metrics not yet in `lib.eval._FN`, add them.
If it uses model keys not yet in `lib.registry._MODELS`, add them.

## Step 6 — recipe

Add a `recipes/<recipe_key>.json` that uses your capability.

## Step 7 — tests

Add unit tests under `tests/unit/test_capabilities.py` and an
integration test under `tests/integration/` that runs at least one
iteration on a synthetic dataset.

## Composition validators

`lib.schemas.mission.CapabilityComposition._check_consistency` enforces
the cross-field rules. If you add a new constraint (e.g., a target_type
that requires a specific validation_strategy), add it there.
