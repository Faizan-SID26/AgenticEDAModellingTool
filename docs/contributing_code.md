# Contributing code

## Setup

```bash
pip install -e .[dev]
pre-commit install
```

## Style

- Type hints on every function signature.
- Docstrings on every module, class, and public function.
- `pathlib.Path`, not strings.
- `logging` over `print`.
- No bare `except` clauses.
- Pydantic v2 throughout.
- PEP 621 metadata in `pyproject.toml`.

## Tests

- Unit tests under `tests/unit/` (no live LLM, no external services).
- Integration tests under `tests/integration/`. Tests that need a live
  Claude agent are marked `@pytest.mark.requires_agent` and skipped by
  default.
- Run `pytest tests/unit/ -v` before opening a PR.

## Branches

- `main` — protected; framework + accumulated knowledge.
- `feature/<name>` — framework feature branches.
- `project/<team>/<name>` — per-project work; auto-created by `eda new-project`.

## What goes where

- New schema → `lib/schemas/` + tests in `tests/unit/test_schemas.py`.
- New capability → `lib/capabilities/<name>.py` + a recipe.
- New domain → `lib/domains/<name>.py` (start from `_template.py`) + a recipe.
- New sketch query → `lib/sketch/queries.py` + register in
  `mcp_servers/sketch_server.py` + add a unit test.
- New skeptic check → `lib/skeptic.py` + tests.
