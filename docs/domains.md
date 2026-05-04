# Domains

A *domain* supplies priors about a kind of data: which keywords identify
process stages, which columns are typically downstream of the target,
default join policies, expected interactions, sensor failure patterns,
hard physical bounds, and a small set of domain-specific seed
hypotheses to add to the universal seeds.

## Built-in domains

- `general` — no priors. Use it when no domain module fits.
- `manufacturing` — process manufacturing. Reference v1 path. Stage
  keywords from raw_material → final_qa, default forbidden patterns
  (`qc_`, `audit_`, `inspector_`, etc.), Arrhenius / pressure-temperature
  relations as physics priors, hard bounds (temperature, pressure, …).
- `forecasting_demand` — calendar / price / inventory features for
  demand forecasting. Validates that the domain abstraction generalizes.

## Adding a new domain

See [`adding_a_domain.md`](adding_a_domain.md). The fastest path:

1. Copy `lib/domains/_template.py` to `lib/domains/<your_domain>.py`.
2. Fill in `SPEC` (stage keywords, forbidden patterns, physics relations,
   expected interactions, sensor failure patterns, hard bounds, skeptic
   extras, seed hypothesis keys).
3. Register the module in `lib.domains.__init__._DOMAIN_MODULES`.
4. Add at least one recipe under `recipes/` that uses the domain.
