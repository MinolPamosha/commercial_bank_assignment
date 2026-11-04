"""Async MCP client — the agent's bridge to the enterprise-data MCP server.

Speaks the Model Context Protocol over Streamable HTTP (official `mcp` SDK):
    initialize → tools/call(<tool>, arguments) → parse the JSON text content.

Every call is bounded by `MAX_TOOL_TIMEOUT_SEC` and every failure mode (server down, slow,
protocol error, tool-side validation error) is returned as a structured result instead of an
exception, so the tools node can degrade gracefully and the Activity Panel can show why.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langsmith import get_current_run_tree
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from backend.app.config import get_settings
from backend.app.models.schemas import Role
from backend.app.observability.logging import get_logger
from backend.app.observability.tracing import trace_tool
from backend.app.security.guardrails import validate_tool_params

logger = get_logger(__name__)

# Catalog tool name (RBAC / supervisor vocabulary) → MCP tool name on the server.
MCP_TOOL_NAMES = {
    "mcp_employee_directory": "employee_directory",
    "mcp_service_catalog": "service_catalog",
    "mcp_incident_records": "incident_records",
}


def _mcp_endpoint() -> str:
    # Trailing slash: the server mounts the transport at /mcp and would 307-redirect "/mcp".
    return f"{get_settings().mcp_base_url.rstrip('/')}/mcp/"


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """FastMCP returns dict results as one JSON text block; errors arrive as `isError=True`."""
    texts = [c.text for c in (result.content or []) if getattr(c, "type", "") == "text"]
    joined = "\n".join(texts)
    if getattr(result, "isError", False):
        return {"ok": False, "error": f"MCP tool error: {joined or 'unknown'}"}
    try:
        return {"ok": True, "data": json.loads(joined) if joined else {}}
    except json.JSONDecodeError:
        return {"ok": True, "data": {"text": joined}}


async def _call(mcp_tool: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    async with streamablehttp_client(_mcp_endpoint(), timeout=timeout, sse_read_timeout=timeout) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(mcp_tool, arguments=params)
    return _parse_tool_result(result)


@trace_tool("mcp_tool")
async def call_mcp_tool(tool_name: str, role: Role, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    run = get_current_run_tree()
    if run is not None:
        run.name = tool_name  # one span per concrete MCP tool in LangSmith
    check = validate_tool_params(tool_name, params, role)
    if not check.ok:
        return {"ok": False, "error": "; ".join(check.reasons)}

    mcp_tool = MCP_TOOL_NAMES.get(tool_name)
    if not mcp_tool:
        return {"ok": False, "error": f"Unknown MCP tool {tool_name}"}

    timeout = get_settings().max_tool_timeout_sec
    try:
        # Outer deadline covers connect + initialize + call; a hung server cannot stall the turn.
        return await asyncio.wait_for(_call(mcp_tool, params, timeout), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("mcp_timeout", tool=tool_name, timeout_sec=timeout)
        return {"ok": False, "error": "MCP tool timed out", "degraded": True}
    except Exception as exc:  # noqa: BLE001  (anyio raises ExceptionGroup for connection errors)
        reason = _root_cause(exc)
        logger.error("mcp_failed", tool=tool_name, error=reason)
        return {"ok": False, "error": f"MCP failure: {reason}", "degraded": True}


def _root_cause(exc: BaseException) -> str:
    """Unwrap anyio/asyncio ExceptionGroups so the log and the panel show the real cause."""
    seen = exc
    while isinstance(seen, BaseExceptionGroup) and seen.exceptions:
        seen = seen.exceptions[0]
    return f"{type(seen).__name__}: {seen}" if str(seen) else type(seen).__name__
