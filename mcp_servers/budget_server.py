"""Budget ledger MCP server."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lib import budget


def _project() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).resolve()
    return Path.cwd().resolve()


_TOOL_REGISTRY: dict[str, Any] = {
    "current_total": lambda: {"current_total": budget.current_total(_project())},
    "fraction_consumed": lambda cap: {"fraction_consumed": budget.fraction_consumed(_project(), int(cap))},
    "project_remaining_iterations": lambda cap, avg: {
        "remaining": budget.project_remaining_iterations(_project(), int(cap), int(avg))
    },
}


def call(name: str, kwargs: dict[str, Any]) -> Any:
    if name not in _TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    return _TOOL_REGISTRY[name](**kwargs)


def main() -> int:
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import Tool, TextContent  # type: ignore

        server = Server("eda-budget")

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(name=k, description=fn.__doc__ or "", inputSchema={"type": "object", "additionalProperties": True})
                for k, fn in _TOOL_REGISTRY.items()
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            return [TextContent(type="text", text=json.dumps(call(name, arguments or {}), default=str))]

        import asyncio

        async def m():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(m())
        return 0
    except ImportError:
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


if __name__ == "__main__":
    sys.exit(main())
