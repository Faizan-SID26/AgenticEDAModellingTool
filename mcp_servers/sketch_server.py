"""MCP stdio server exposing the sketch tool surface to Claude Code.

Run as:

    python -m mcp_servers.sketch_server <project_dir>

Where `project_dir` defaults to the current working directory. The
server registers tools 1:1 with `lib.sketch.queries` functions.

Falls back to a no-op stdin/stdout JSON-RPC stub if the `mcp` package is
not installed (so `pip install -e .` succeeds without the optional
dependency).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from lib.sketch import queries

_log = logging.getLogger("eda.mcp.sketch")


_TOOL_REGISTRY: dict[str, Any] = {
    "quantile": queries.quantile,
    "distribution": queries.distribution,
    "cardinality": queries.cardinality,
    "missingness": queries.missingness,
    "top_interactions": queries.top_interactions,
    "conditional_dependence": queries.conditional_dependence,
    "principal_components": queries.principal_components,
    "regimes": queries.regimes,
    "regime_compare": queries.regime_compare,
    "motifs": queries.motifs,
    "discords": queries.discords,
    "causal_neighbors": queries.causal_neighbors,
    "confounder_candidates": queries.confounder_candidates,
    "failure_clusters": queries.failure_clusters,
    "match_residuals": queries.match_residuals,
    "fit_quick": queries.fit_quick,
    "cross_validate_quick": queries.cross_validate_quick,
}


def _resolve_project_dir() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).resolve()
    return Path.cwd().resolve()


def call(name: str, kwargs: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    """Dispatch one tool call. Used by both the MCP and the fallback paths."""
    if name not in _TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    fn = _TOOL_REGISTRY[name]
    return fn(project_dir, **kwargs)


def _serve_via_mcp() -> int:
    """Run the official MCP stdio server if the `mcp` package is installed."""
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import Tool, TextContent  # type: ignore
    except ImportError:
        return 2

    project_dir = _resolve_project_dir()
    server = Server("eda-sketch")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=fn.__doc__ or "",
                inputSchema={"type": "object", "additionalProperties": True},
            )
            for name, fn in _TOOL_REGISTRY.items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = call(name, arguments or {}, project_dir)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    import asyncio

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())
    return 0


def _serve_jsonrpc_fallback() -> int:
    """Trivial line-based JSON-RPC fallback when `mcp` is not installed.

    Each input line is `{"name": "<tool>", "args": {...}}`. Each output
    line is the JSON tool result. Useful for debugging / unit tests.
    """
    project_dir = _resolve_project_dir()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = call(req["name"], req.get("args", {}), project_dir)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(res, default=str) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    rc = _serve_via_mcp()
    if rc == 2:
        _log.warning("mcp package not installed; running JSON-RPC fallback over stdio")
        return _serve_jsonrpc_fallback()
    return rc


if __name__ == "__main__":
    sys.exit(main())
