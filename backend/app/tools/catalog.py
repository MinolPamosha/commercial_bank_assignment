"""Tool catalog — the one place that says which tools exist and what they do.

Both the Supervisor prompt (so the LLM knows a tool exists) and the Tools agent (so a
selected tool maps to an executor) read from here. RBAC is *not* decided here: the
supervisor may route any role to a tool; the tool gate (`validate_tool_params`) denies it
and the denial is surfaced as a guardrail event. That is deliberate — the evaluator must be
able to see the agent attempt a tool and be refused, not have the LLM quietly avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    route: str  # graph route that executes this tool
    params: dict[str, str] = field(default_factory=dict)  # param name -> plain-English meaning


TOOL_CATALOG: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="knowledge_search",
        description="Hybrid dense+BM25 search over internal documents (policies, incidents, runbooks, architecture, specs, meeting notes).",
        route="retrieve",
    ),
    ToolSpec(
        name="mcp_employee_directory",
        description="Live enterprise directory: who someone is, their team, who is on-call.",
        route="tools",
        params={"q": "name or team to filter by (optional)"},
    ),
    ToolSpec(
        name="mcp_service_catalog",
        description="Live service catalog: systems, tier, owner, SLO.",
        route="tools",
        params={"tier": "service tier number to filter by (optional)"},
    ),
    ToolSpec(
        name="mcp_incident_records",
        description="Live incident ticket records (id, status open/closed, severity, service). Use for 'current/open incidents', not for narrative post-mortems.",
        route="tools",
        params={"status": "'open' or 'closed' (optional)"},
    ),
    ToolSpec(
        name="python_analysis",
        description="Safe arithmetic expression evaluator for standalone calculations. (Statistics over the documents are done by the research route, which calls this tool itself.)",
        route="tools",
        params={"expression": "safe Python expression to evaluate (optional)"},
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOL_CATALOG}
TOOL_NAMES: frozenset[str] = frozenset(TOOLS_BY_NAME)


def catalog_for_prompt() -> str:
    """Compact, LLM-readable listing used by the Supervisor."""
    lines = []
    for t in TOOL_CATALOG:
        params = ", ".join(f"{k} ({v})" for k, v in t.params.items()) or "none"
        lines.append(f"- {t.name} [route={t.route}] — {t.description} Params: {params}")
    return "\n".join(lines)
