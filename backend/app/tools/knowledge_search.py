"""Knowledge Search Tool — thin async wrapper over hybrid retriever."""

from __future__ import annotations

from typing import Any

from backend.app.models.schemas import Role
from backend.app.observability.logging import get_logger
from backend.app.observability.tracing import trace_tool
from backend.app.retrieval.hybrid import retriever
from backend.app.security.guardrails import screen_retrieved_content, validate_tool_params

logger = get_logger(__name__)


@trace_tool("knowledge_search")
async def knowledge_search(
    query: str,
    role: Role,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    check = validate_tool_params("knowledge_search", {"query": query, **(filters or {})}, role)
    if not check.ok:
        return {"ok": False, "error": "; ".join(check.reasons), "results": []}

    try:
        results = await retriever.search(query, filters=filters)
        results, dropped = screen_retrieved_content(results, role)
        logger.info("knowledge_search_ok", hits=len(results), role=role.value, dropped=dropped)
        return {
            "ok": True,
            # Chunks removed by the content guardrail (access level / embedded injection) — shown in the panel.
            "dropped_chunks": dropped,
            "results": [
                {
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "score": r.get("rerank_score", r.get("score", 0.0)),
                    "snippet": r["text"][:400],
                    "metadata": r.get("metadata", {}),
                    "chunk_id": r["chunk_id"],
                }
                for r in results
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("knowledge_search_failed", error=str(exc))
        return {"ok": False, "error": f"Vector/retrieval failure: {exc}", "results": [], "degraded": True}
