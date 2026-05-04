# Contributing

Three contribution flows live in this repo:

1. **Project work** — running `/init` → `/plan` → `/run` on a new dataset.
2. **Knowledge merges** — completing a project, raising a PR to `main`,
   letting CI extract knowledge into `knowledge/`.
3. **Framework PRs** — adding capabilities, domains, recipes, or core
   library improvements.

## Branching

- `main` — framework + accumulated cross-project knowledge. Protected.
- `project/<team>/<name>` — per-project work. Created automatically by
  `eda new-project` (the workspace is this repo, so projects are git
  branches inside it).
- `feature/<short-name>` — framework feature branches.

Project branches are merged to `main` only after `/finalize` has produced a
valid `results/FINAL.md` and `/contribute` has prepared the contribution
manifest. The post-merge extractor (CI) then writes to `knowledge/`.

## Project lifecycle

```
/init        Profile data → memory/INIT_PROFILE.json + results/init_report.md
/plan        Adaptive Q&A → MISSION.json + JOIN_PLAN.json + HYPOTHESES.jsonl
/run         Autonomous bootstrap → iterate → synthesize → finalize
/contribute  Stage knowledge artifacts and open PR to main
```

`/run` is autonomous from bootstrap through finalize. The user is never
required to type `/continue`. It terminates only on goal-met,
budget-exhausted, stagnation, catastrophic skeptic failure, iteration cap,
or interrupt.

## Adding a new capability

See [`docs/adding_a_capability.md`](docs/adding_a_capability.md). At minimum:

1. Add a module under `lib/capabilities/`.
2. Implement the capability interface (see `lib/capabilities/base.py`).
3. Register it in `lib/capabilities/__init__.py`.
4. Add unit tests under `tests/unit/test_capabilities.py`.
5. Add at least one recipe that uses it under `recipes/`.

## Adding a new domain

See [`docs/adding_a_domain.md`](docs/adding_a_domain.md). Copy
`lib/domains/_template.py`, fill in domain priors, register in
`lib/domains/__init__.py`.

## Knowledge contribution discipline

`/contribute` extracts only:

- Hypothesis patterns that produced ≥ a configured info gain threshold.
- Failure modes that fired and were resolved.
- Sketch-similarity vectors for retrieval.

Column names are anonymized to **semantic roles** via the domain module
(e.g., `temp_zone_a` → `<sensor:temperature>`). Raw data never enters
`knowledge/`.

## Style

- Type hints on every function signature.
- Docstrings on every module, class, and public function.
- `pathlib.Path` over string paths.
- `logging` over `print`.
- No bare `except` clauses.

Run `pre-commit install` after cloning. CI runs `ruff` + `pytest` + the
post-merge extractor (on merges to `main`).
