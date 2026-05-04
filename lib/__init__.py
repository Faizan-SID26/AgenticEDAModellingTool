"""EDA Framework — core library.

The agent never imports from `lib` directly: it uses slash commands and the
MCP tool surface. Code in `lib/` is invoked deterministically by the runner
sub-agent and by background machinery (`lib.run`, `lib.state`, etc.).
"""
from __future__ import annotations

__version__ = "0.1.0"
"""Framework version stamped into every artifact's `framework_version` field."""

SCHEMA_VERSION = "1"
"""Schema epoch. Bump on a breaking schema change. Per-model `schema_version`
fields default to this value."""
