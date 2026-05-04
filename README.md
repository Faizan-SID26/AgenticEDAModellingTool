# EDA Framework

A Claude-Code-native ML platform for end-to-end exploratory data analysis,
modeling, and hypothesis testing against arbitrary datasets. Completed
projects merge into the main branch and become organizational knowledge that
future projects automatically learn from.

## What it is

- **One-repo workspace.** This repo IS the workspace: framework code,
  cross-project knowledge, and individual projects all live here.
- **Slash-command UX.** `/init`, `/plan`, `/run`, `/contribute` orchestrate the
  agent through inspection, planning, autonomous iteration, and contribution.
- **Process Data Sketch.** A compact (<1MB), deterministically-built summary
  of every dataset that the agent queries via an MCP tool surface — the
  agent never reads raw data.
- **Capability composition, not problem-type dispatch.** Missions are
  declared as compositions over `temporal_structure`, `leakage_model`,
  `target_type`, `validation_strategy`, `recommendation_type`. New problem
  shapes don't require new dispatch tables.
- **Honest failure is shippable.** "No signal found, here's the evidence" is a
  valid project outcome.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
# 1. Create a project
eda new-project my_first --domain manufacturing \
    --recipe manufacturing_defect_classification --budget 30

# 2. In Claude Code, change to the project directory:
cd projects/my_first

# 3. Drop your data files into ./data/, then run:
#    /init       — profile data, propose joins
#    /plan       — adaptive Q&A to lock the MISSION
#    /run        — autonomous iteration through to a final recommendation
#    /contribute — prepare a PR that merges learnings into knowledge/
```

See `docs/quickstart.md` for a full walk-through and `docs/workflow.md` for
the phase-by-phase reference.

## Repository layout (top level)

```
eda-framework/
├── lib/                # Python core: schemas, sketch, iteration loop, ...
├── mcp_servers/        # Tool-surface MCP servers consumed by Claude Code
├── seeds/              # Universal hypothesis seeds (5 of them)
├── recipes/            # Pre-validated capability compositions
├── knowledge/          # Cross-project knowledge (grows on every merge)
├── projects/           # Per-project workspaces (team branches)
├── tools/              # Operational tooling (post-merge extractor, audit)
├── tests/              # Unit + integration + agent eval suites
├── docs/               # User and contributor docs
├── .claude/            # Slash commands, sub-agents, role skills
└── pyproject.toml
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the architectural principles
and agent role specifications.

## Status

This is **v1**. Some advanced features are stubbed with `NotImplementedError`
where the dependency surface is large enough that a clean install is more
important than feature completeness. See `BUILD_NOTES.md` for the running
list of partial / stubbed items.

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for branching conventions, the
project lifecycle (`/init` → `/plan` → `/run` → `/contribute`), and how
knowledge merges propagate into `knowledge/`.
