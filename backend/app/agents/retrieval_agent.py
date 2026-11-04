"""Retrieval Agent — RAG / hybrid search operations."""

from __future__ import annotations

from backend.app.agents.state import AgentState
from backend.app.models.schemas import AgentEvent, Role
from backend.app.observability.logging import get_logger
from backend.app.tools.knowledge_search import knowledge_search

logger = get_logger(__name__)


async def retrieval_node(state: AgentState) -> dict:
    role = state["role"] if isinstance(state["role"], Role) else Role(state["role"])
    events = [
        AgentEvent(event_type="node", node="retrieval", message="Retrieval agent starting hybrid search"),
        AgentEvent(event_type="retrieval", node="retrieval", message="Dense + BM25 sparse search in progress"),
    ]

    # Optional metadata hints from natural language
    filters = {}
    msg = state["message"].lower()
    if "incident" in msg or "outage" in msg:
        filters["document_type"] = "incident"
    if "policy" in msg:
        filters["document_type"] = "policy"
    if "payments" in msg or "payment" in msg or "upi" in msg or "neft" in msg:
        filters["department"] = "payments"

    result = await knowledge_search(state["message"], role=role, filters=filters or None)
    events.append(
        AgentEvent(
            event_type="tool_call",
            node="retrieval",
            message="knowledge_search executed",
            detail={"ok": result.get("ok"), "hits": len(result.get("results") or []), "filters": filters},
        )
    )
    dropped = result.get("dropped_chunks") or {}
    if any(dropped.values()):
        events.append(
            AgentEvent(
                event_type="guardrail",
                node="retrieval",
                message="Retrieved-content guardrail removed chunks before they reached the LLM",
                detail=dropped,
            )
        )

    degraded = bool(result.get("degraded"))
    errors = list(state.get("errors") or [])
    if not result.get("ok"):
        errors.append(result.get("error", "retrieval failed"))
        degraded = True

    citations = [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "department": (r.get("metadata") or {}).get("department", ""),
            "document_type": (r.get("metadata") or {}).get("document_type", ""),
            "access_level": (r.get("metadata") or {}).get("access_level", ""),
            "score": float(r.get("score") or 0),
            "snippet": r.get("snippet", ""),
        }
        for r in result.get("results") or []
    ]

    events.append(
        AgentEvent(
            event_type="retrieval",
            node="retrieval",
            message=f"Retrieved {len(citations)} evidence chunks",
            detail={"doc_ids": [c["doc_id"] for c in citations]},
        )
    )

    return {
        "active_node": "retrieval",
        "retrieval_results": result.get("results") or [],
        "citations": citations,
        "events": events,
        "errors": errors,
        "degraded": state.get("degraded", False) or degraded,
        "degradation_reason": "retrieval_failure" if degraded else state.get("degradation_reason"),
        "route": "respond",
    }
