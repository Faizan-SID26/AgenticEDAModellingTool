# Adding a domain

A domain module supplies priors about a kind of data. Adding a new domain
is purely additive: you create one file, register it, and (optionally)
add a recipe.

## Step 1 — copy the template

```bash
cp lib/domains/_template.py lib/domains/<your_domain>.py
```

## Step 2 — fill in `SPEC`

Edit `lib/domains/<your_domain>.py` and populate the `DomainSpec`:

- `key` — stable identifier.
- `description` — one sentence.
- `stage_keywords` — ordered tuples of `(stage_name, keyword_tuple)`. The
  first match wins. Used for `leakage_model=stage_frontier` and for the
  knowledge-extractor's column-anonymization fallback.
- `default_forbidden` — substrings that should land in
  `MISSION.forbidden_columns` by default.
- `default_leak_frontier` — stage name at which leakage is gated.
- `lag_join_default_policy` — for asof joins.
- `physics_relations` — known relations between feature roles, used by
  the planner to seed feature engineering and by the skeptic to flag
  contradictions.
- `expected_interactions` — pairs/triples of feature roles you expect to
  carry signal.
- `sensor_failure_patterns` — strings used by the failure-mode catalog.
- `hard_bounds` — hard physical bounds (used by the skeptic).
- `skeptic_extras` — extra skeptic check keys for this domain.
- `seed_hypotheses` — recipe / hypothesis-template keys to seed.

## Step 3 — register

Edit `lib/domains/__init__.py` and add your module to
`_DOMAIN_MODULES`.

## Step 4 — recipe (optional)

Drop a JSON under `recipes/` referencing your domain. See
[`recipes.md`](recipes.md).

## Step 5 — tests

Add a unit test that:

```python
from lib.domains import get
spec = get("<your_domain>")
assert spec.key == "<your_domain>"
```

Optionally test `infer_stage_from_keywords` against representative column
names.
