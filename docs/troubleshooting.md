# Troubleshooting

## "no PROJECT.json at <path>"

You ran a slash command outside a project directory. `cd projects/<name>`.

## "no L4 coreset for capability <X>"

`/bootstrap` was run for a different capability list. Re-bootstrap with
the correct capability list, or pick one of the keys printed in
`sketch/manifest.json#capabilities`.

## "audit failed"

A plan dict tried to use a column in `MISSION.forbidden_columns` outside
of an `area=leakage_probe` plan. This is a feature, not a bug — the
audit gate enforces the leakage policy.

## "validation error: time_column is required when capability.temporal_structure != 'none'"

Set `MISSION.time_column` (via `/plan`) before unlocking.

## Plot rendering errors

The framework forces `matplotlib.use("Agg")` so plots write headless.
If you see "ImportError: No module named 'matplotlib'", install it
(`pip install matplotlib`) — it is a hard dependency.

## "schema_version mismatch"

Run `python tools/migrate_schema.py <path>`. v1 has a single schema
epoch, so this is a no-op until v2.

## Replay drift

If `eda replay` reports `abs_delta > 0`:

1. Check the framework version pin in `PROJECT.json` matches the current
   `lib.__version__`.
2. Check the experiment row's `seeds` field — replay uses these.
3. Run `pytest tests/unit/test_sketch_determinism.py` — drift in the
   sketch implies a non-deterministic dependency.

## Repo health

```bash
python tools/audit_repo.py
```

Reports oversized files, unparseable recipes, missing universal seeds,
or broken MCP server entries.
