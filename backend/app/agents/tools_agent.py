"""Tools agent — executes the Supervisor's tool choice (MCP / analysis) under RBAC.

Selection order:
1. `selected_tool` + `tool_params` chosen by the Supervisor LLM (validated against the catalog)
2. keyword fallback when the LLM was unavailable or did not name a tool

Execution always goes through the tool gate (`validate_tool_params`) inside each tool, so a
viewer who is routed here is *denied* visibly rather than silently skipped.
"""

from __future__ import annotations

from typing import Any

from backend.app.agents.state import AgentState
from backend.app.models.schemas import AgentEvent, Role
from backend.app.tools.catalog import TOOL_NAMES
from backend.app.tools.mcp_client import call_mcp_tool
from backend.app.tools.python_analysis import python_analysis

KEYWORD_FALLBACK: tuple[tuple[tuple[str, ...], str], ...] = (
    (("employee", "directory", "who is", "on-call", "on call"), "mcp_employee_directory"),
    (("catalog", "service"), "mcp_service_catalog"),
    (("incident record", "open incident", "incidents"), "mcp_incident_records"),
)


def _fallback_tool(message: str) -> str:
    msg = message.lower()
    for keywords, tool in KEYWORD_FALLBACK:
        if any(k in msg for k in keywords):
            return tool
    return "mcp_service_catalog"


async def _execute(tool: str, params: dict[str, Any], role: Role) -> dict[str, Any]:
    if tool == "python_analysis":
        return await python_analysis(role=role, expression=str(params.get("expression") or "len([1,2,3])"))
    return await call_mcp_tool(tool, role, params=params)


async def tools_node(state: AgentState) -> dict:
    role = state["role"] if isinstance(state["role"], Role) else Role(state["role"])
    events = [AgentEvent(event_type="node", node="tools", message="Tools agent selecting enterprise tools")]

    tool = state.get("selected_tool")
    params: dict[str, Any] = dict(state.get("tool_params") or {})
    selected_by = "supervisor"
    if tool not in TOOL_NAMES:
        tool, params, selected_by = _fallback_tool(state["message"]), {}, "keyword_fallback"

    events.append(
        AgentEvent(
            event_type="tool_call",
            node="tools",
            message=f"Invoking {tool}",
            detail={"tool": tool, "params": params, "selected_by": selected_by, "role": role.value},
        )
    )
    result = await _execute(tool, params, role)
    results = [{"tool": tool, "result": result}]

    degraded = bool(result.get("degraded"))
    if not result.get("ok"):
        events.append(
            AgentEvent(
                event_type="guardrail" if "not permitted" in str(result.get("error", "")) else "error",
                node="tools",
                message=f"Tool blocked or failed: {tool}",
                detail=result,
            )
        )
    else:
        events.append(
            AgentEvent(
                event_type="tool_call",
                node="tools",
                message=f"{tool} returned data",
                detail={"keys": list((result.get("data") or result.get("analysis") or {}).keys())},
            )
        )

    return {
        "active_node": "tools",
        "tool_results": results,
        "events": events,
        "degraded": state.get("degraded", False) or degraded,
        "degradation_reason": "mcp_failure" if degraded else state.get("degradation_reason"),
        "route": "respond",
    }
