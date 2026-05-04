# Installation

## Requirements

- Python 3.10+ (3.11 recommended).
- pip ≥ 22.

## Steps

```bash
git clone <this repo>
cd eda-framework
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
pre-commit install
```

After install, `eda --help` should show all commands.

## Optional MCP integration

If you want the sketch / retrieval / budget tool surfaces wired into
Claude Code as MCP servers, install the optional `mcp` extra:

```bash
pip install -e .[mcp]
```

Without `mcp`, the servers fall back to a line-based JSON-RPC stdio
protocol that's still usable from tests and scripts.

## Heavy optional dependencies

- `lightgbm` is recommended for the boosted-tree models. If unavailable,
  the registry falls back to sklearn `GradientBoosting*`.
- `stumpy` provides matrix profile for L5; the fallback is a sliding-mean
  motif/discord detector.
- `dowhy` powers the counterfactual estimator; the fallback is a
  bootstrap-CI'd multivariate regression.
- `lifelines` provides Cox PH for `predictive_maintenance`; without it,
  PdM fits a ridge regressor on event times.

## Troubleshooting

- "Schema validation failed" on a project artifact means the artifact was
  written by an older framework version. Run
  `python tools/migrate_schema.py <path-to-artifact>` to migrate (currently
  a no-op, since v1 has one schema epoch).
- "no L4 coreset for capability X" means `/bootstrap` did not run with
  capability X. Re-bootstrap with the right capability list.
