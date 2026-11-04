"""Retrieval Agent — RAG / hybrid search operations."""

from __future__ import annotations

import asyncio

from backend.app.agents.state import AgentState
from backend.app.models.schemas import AgentEvent, Role
from backend.app.observability.logging import get_logger
from backend.app.tools.knowledge_search import knowledge_search

logger = get_logger(__name__)

HINT_BOOST = 0.08  # added to the score of hits that match every guessed metadata hint
MAX_RESULTS = 6


def _metadata_hints(message: str) -> dict[str, str]:
    msg = message.lower()
    hints: dict[str, str] = {}
    if "incident" in msg or "outage" in msg:
        hints["document_type"] = "incident"
    if "policy" in msg:
        hints["document_type"] = "policy"
    if any(w in msg for w in ("payments", "payment", "upi", "neft", "imps")):
        hints["department"] = "payments"
    return hints


async def _search_with_soft_hints(query: str, role: Role, hints: dict[str, str]) -> dict:
    """Run the hinted and the unfiltered search side by side and merge them.

    The hinted search keeps precision when the guess is right; the unfiltered search keeps recall
    when it is wrong. A hit present in both (or matching the hints) gets a small boost.
    """
    if not hints:
        return await knowledge_search(query, role=role, filters=None)

    hinted, plain = await asyncio.gather(
        knowledge_search(query, role=role, filters=hints),
        knowledge_search(query, role=role, filters=None),
    )
    if not plain.get("ok"):
        return hinted if hinted.get("ok") else plain
    if not hinted.get("ok"):
        return plain

    merged: dict[str, dict] = {}
    for r in list(plain.get("results") or []) + list(hinted.get("results") or []):
        md = r.get("metadata") or {}
        matches_hints = all(str(md.get(k, "")) == v for k, v in hints.items())
        score = float(r.get("score") or 0.0) + (HINT_BOOST if matches_hints else 0.0)
        key = r.get("chunk_id") or r["doc_id"]
        if key not in merged or score > merged[key]["score"]:
            merged[key] = {**r, "score": score}
    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:MAX_RESULTS]

    dropped = {}
    for src in (plain, hinted):
        for k, v in (src.get("dropped_chunks") or {}).items():
            dropped[k] = max(dropped.get(k, 0), int(v or 0))
    return {"ok": True, "results": results, "dropped_chunks": dropped, "degraded": bool(plain.get("degraded") or hinted.get("degraded"))}


async def retrieval_node(state: AgentState) -> dict:
    role = state["role"] if isinstance(state["role"], Role) else Role(state["role"])
    events = [
        AgentEvent(event_type="node", node="retrieval", message="Retrieval agent starting hybrid search"),
        AgentEvent(
            event_type="retrieval",
            node="retrieval",
            message="Dense + BM25 sparse search, then lexical rerank",
            detail={"pipeline": ["dense_search", "bm25_search", "rrf_fuse", "rerank"]},
        ),
    ]

    # Metadata hints guessed from the wording. They are a *preference*, not a hard filter: a question
    # that says "payments" may still be best answered by a risk team document (e.g. meeting notes),
    # so we search with and without the hint concurrently and merge, boosting hinted matches.
    hints = _metadata_hints(state["message"])
    result = await _search_with_soft_hints(state["message"], role, hints)
    events.append(
        AgentEvent(
            event_type="tool_call",
            node="retrieval",
            message="knowledge_search executed" + (" (hinted + unfiltered, merged)" if hints else ""),
            detail={"ok": result.get("ok"), "hits": len(result.get("results") or []), "hints": hints},
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
            message=f"Retrieved {len(citations)} evidence chunks (reranked)",
            detail={
                "doc_ids": [c["doc_id"] for c in citations],
                "scores": [{"doc_id": c["doc_id"], "score": round(float(c.get("score") or 0), 4)} for c in citations],
            },
        )
    )

    return {
        "active_node": "retrieval",
        "retrieval_results": result.get("results") or [],
        "citations": citations,
        "withheld_chunks": {k: int(v or 0) for k, v in dropped.items()},
        "events": events,
        "errors": errors,
        "degraded": state.get("degraded", False) or degraded,
        "degradation_reason": "retrieval_failure" if degraded else state.get("degradation_reason"),
        "route": "respond",
    }
