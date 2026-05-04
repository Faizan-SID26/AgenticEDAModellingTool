"""Retrieval MCP server: exposes cross-project knowledge to the agent."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lib import retrieval

_TOOL_REGISTRY: dict[str, Any] = {
    "list_sketch_index": lambda workspace=None: retrieval.list_sketch_index(workspace),
    "query_similar_projects": retrieval.query_similar_projects,
    "load_hypothesis_library": lambda workspace=None: [
        h.model_dump() for h in retrieval.load_hypothesis_library(workspace)
    ],
    "load_failure_modes": lambda workspace=None: [
        f.model_dump() for f in retrieval.load_failure_modes(workspace)
    ],
    "query_hypotheses": lambda workspace=None, **kw: [
        h.model_dump() for h in retrieval.query_hypotheses(workspace, **kw)
    ],
    "summarize_library": retrieval.summarize_library,
}


def call(name: str, kwargs: dict[str, Any]) -> Any:
    if name not in _TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    return _TOOL_REGISTRY[name](**kwargs)


def _serve_via_mcp() -> int:
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import Tool, TextContent  # type: ignore
    except ImportError:
        return 2
    server = Server("eda-retrieval")

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
        result = call(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    import asyncio

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())
    return 0


def _serve_jsonrpc_fallback() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = call(req["name"], req.get("args", {}))
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(res, default=str) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    rc = _serve_via_mcp()
    if rc == 2:
        return _serve_jsonrpc_fallback()
    return rc


if __name__ == "__main__":
    sys.exit(main())
