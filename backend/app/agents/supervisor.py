"""Supervisor Agent — intent understanding, task decomposition, routing.

Decision inputs (all shown to the LLM, all visible in the Activity Panel event):
- the user's message
- conversation memory (so follow-ups like "which theme was most common?" are answered
  from context instead of re-running the whole research pipeline)
- the tool catalog (so the LLM knows live MCP data exists and can pick a tool + params)
- the caller's role (informational only — RBAC is enforced later by the tool gate)

Every field the LLM returns is validated against a closed set before it touches graph
state; anything invalid falls back to the deterministic heuristic router.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.app.agents.llm import acomplete, heuristic_intent
from backend.app.agents.state import AgentState
from backend.app.models.schemas import AgentEvent, Role
from backend.app.observability.logging import get_logger
from backend.app.security.auth import role_allows_tool
from backend.app.tools.catalog import TOOL_NAMES, TOOLS_BY_NAME, catalog_for_prompt

logger = get_logger(__name__)

VALID_INTENTS = {"qa", "research", "tool", "admin", "chitchat"}
VALID_ROUTES = {"retrieve", "research", "tools", "respond"}

DEFAULT_PLANS = {
    "retrieve": ["hybrid search", "cite evidence", "answer"],
    "research": ["explore corpus", "batch docs", "analyze themes", "aggregate"],
    "tools": ["select MCP/analysis tool", "execute under RBAC", "answer"],
    "respond": ["compose reply from conversation context"],
}

SUPERVISOR_SYSTEM = """You are the Supervisor of a multi-agent assistant for bank employees.
Decide how to handle the request. Reply with ONLY a JSON object:
{{
  "intent": "qa" | "research" | "tool" | "admin" | "chitchat",
  "route": "retrieve" | "research" | "tools" | "respond",
  "tool": <tool name from the catalog or null>,
  "tool_params": <flat JSON object of params for that tool, or {{}}>,
  "plan": [<2-4 short steps>],
  "reason": <one sentence>
}}

Routing rules:
- "retrieve": a factual question answerable from internal documents (one topic, one or few docs).
- "research": broad synthesis across MANY documents — "summarize all…", recurring themes, trends,
  comparisons across incidents/policies. Expensive; only when genuinely multi-document.
- "tools": the user needs LIVE enterprise data (people, on-call, teams, service catalog/owners/SLOs,
  current/open incident tickets) or a calculation. Set "tool" and "tool_params".
- "respond": greetings/small talk, OR a follow-up that can be answered from the conversation
  memory below without new evidence (e.g. "which one was most common?", "say that shorter").
  Intent "chitchat" is only for greetings/thanks; a follow-up about earlier answers is "qa".
- Always pick "tools" when a catalog tool clearly fits, regardless of the user's role; authorization
  is enforced downstream and must stay visible.

Tool catalog:
{catalog}
"""


def _parse_llm_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def _validate_decision(data: dict[str, Any], fallback_intent: str, fallback_route: str) -> dict[str, Any]:
    """Coerce the LLM output into the closed vocabulary the graph understands."""
    intent = data.get("intent") if data.get("intent") in VALID_INTENTS else fallback_intent
    route = data.get("route") if data.get("route") in VALID_ROUTES else fallback_route

    tool = data.get("tool")
    if tool not in TOOL_NAMES:
        tool = None
    params = data.get("tool_params")
    if not isinstance(params, dict):
        params = {}
    # Only flat, primitive params are allowed through to the tool gate.
    params = {str(k): v for k, v in params.items() if isinstance(v, (str, int, float, bool))}

    # Keep route/tool consistent: a selected tool decides its own route.
    if tool:
        route = TOOLS_BY_NAME[tool].route
        if route == "tools" and intent not in {"tool", "admin"}:
            intent = "tool"
    else:
        params = {}  # params without a validated tool are meaningless; never forward them

    plan = [str(p) for p in (data.get("plan") or []) if str(p).strip()][:6]
    reason = str(data.get("reason") or "")
    return {"intent": intent, "route": route, "tool": tool, "tool_params": params, "plan": plan, "reason": reason}


async def supervisor_node(state: AgentState) -> dict:
    message = state["message"]
    role = state["role"] if isinstance(state["role"], Role) else Role(state["role"])
    memory = (state.get("memory_context") or "").strip()
    events = [
        AgentEvent(event_type="node", node="supervisor", message="Supervisor analyzing intent"),
        AgentEvent(event_type="state", node="supervisor", message="Active node: supervisor"),
    ]

    h_intent, h_route = heuristic_intent(message)
    decision = _validate_decision({}, h_intent, h_route)
    source = "heuristic"

    # Prefer LLM decomposition when available; fall back to the heuristic router on any failure.
    try:
        user_block = (
            f"Caller role: {role.value}\n\n"
            f"Conversation memory (most recent last):\n{memory or '(none — first turn)'}\n\n"
            f"New request:\n{message}"
        )
        raw = await acomplete(system=SUPERVISOR_SYSTEM.format(catalog=catalog_for_prompt()), user=user_block)
        data = _parse_llm_json(raw)
        if data:
            decision = _validate_decision(data, h_intent, h_route)
            source = "llm"
    except Exception as exc:  # noqa: BLE001
        events.append(
            AgentEvent(
                event_type="error",
                node="supervisor",
                message=f"LLM intent failed; using heuristic router ({exc})",
            )
        )

    if not decision["plan"]:
        decision["plan"] = DEFAULT_PLANS.get(decision["route"], ["answer"])

    detail = {
        "intent": decision["intent"],
        "route": decision["route"],
        "plan": decision["plan"],
        "decided_by": source,
        "reason": decision["reason"],
        "memory_used": bool(memory),
    }
    if decision["tool"]:
        detail["tool"] = decision["tool"]
        detail["tool_params"] = decision["tool_params"]
        detail["role_allowed"] = role_allows_tool(role, decision["tool"])  # informational; gate enforces

    events.append(
        AgentEvent(
            event_type="state",
            node="supervisor",
            message=f"Routed intent={decision['intent']} → {decision['route']}"
            + (f" (tool: {decision['tool']})" if decision["tool"] else ""),
            detail=detail,
        )
    )
    logger.info("supervisor_routed", intent=decision["intent"], route=decision["route"], tool=decision["tool"], by=source)
    return {
        "intent": decision["intent"],
        "route": decision["route"],
        "plan": decision["plan"],
        "selected_tool": decision["tool"],
        "tool_params": decision["tool_params"],
        "active_node": "supervisor",
        "events": events,
        "recursion_depth": state.get("recursion_depth", 0),
    }
