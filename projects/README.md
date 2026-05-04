# Projects

Each subdirectory is a project. Project work happens on dedicated git
branches (`project/<team>/<name>`). The ``.templates/_project_template/``
directory is the skeleton copied at creation time by `eda new-project`.

Per-project committed artifacts:

- `PROJECT.json`
- `MISSION.json` (after `/plan`)
- `memory/INIT_PROFILE.json`, `memory/COLUMNS.json`, `memory/JOIN_PLAN.json`,
  `memory/HYPOTHESES.jsonl`, `memory/COURSE.md`
- `experiment_log.jsonl`
- `budget.jsonl`
- `sketch/manifest.json`, `sketch/annotations/*.jsonl`
- `results/synthesis_NNN.md`, `results/FINAL.md`

Per-project gitignored artifacts (regenerable):

- `data/`
- `sketch/L*.bin`, `sketch/L*.parquet`, `sketch/raw_joined.parquet`
- `results/iter_NNN/`, `results/plots/`
