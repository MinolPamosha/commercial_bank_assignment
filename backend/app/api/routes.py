"""FastAPI routes — async chat, auth, feedback, health."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.agents.graph import NoPendingApproval, run_agent, stream_agent
from backend.app.config import get_settings
from backend.app.memory.store import memory_store
from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationSaveRequest,
    ConversationSummary,
    FeedbackRequest,
    HealthResponse,
    HitlDecision,
    LoginRequest,
    TokenResponse,
)
from backend.app.observability.logging import get_logger
from backend.app.retrieval.hybrid import retriever
from backend.app.security.auth import (
    authenticate,
    create_access_token,
    create_hitl_token,
    decode_hitl_token,
    get_current_user,
)
from backend.app.security.rate_limit import rate_limiter

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        bank=settings.bank_name,
        retrieval_mode=retriever.mode if retriever._ready else "not_initialized",
        langsmith_enabled=settings.tracing_enabled,
        llm_provider=settings.llm_provider,
        llm_model=settings.resolved_llm_model,
        llm_enabled=settings.llm_enabled,
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    user = authenticate(body.username, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(user["username"], user["role"], user["display_name"])
    return TokenResponse(access_token=token, role=user["role"], display_name=user["display_name"])


async def _enforce_rate_limit(user: dict[str, Any]) -> None:
    """Per-user token bucket; raises 429 with Retry-After for graceful client handling."""
    allowed, retry_after = await rate_limiter.allow(user["username"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def _finalize_turn(result: dict[str, Any], body: ChatRequest, user: dict[str, Any]) -> ChatResponse:
    """Issue a signed HITL token (if the graph paused) and build the API response."""
    hitl_token = None
    if result.get("hitl_required"):
        hitl_token = create_hitl_token(username=user["username"], session_id=body.session_id, message=body.message)

    citations = [Citation(**c) if not isinstance(c, Citation) else c for c in result.get("citations") or []]
    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        session_id=body.session_id,
        events=result.get("events") or [],
        hitl_required=bool(result.get("hitl_required")),
        hitl_token=hitl_token,
        degraded=bool(result.get("degraded")),
        degradation_reason=result.get("degradation_reason"),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, user=Depends(get_current_user)):
    """Blocking chat turn — returns the answer plus the full activity-event log."""
    await _enforce_rate_limit(user)
    try:
        result = await run_agent(
            message=body.message,
            session_id=body.session_id,
            user_id=user["username"],
            role=user["role"],
            require_hitl=body.require_hitl,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("chat_unhandled", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Agent failure: {exc}") from exc
    return _finalize_turn(result, body, user)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, user=Depends(get_current_user)):
    """Real-time chat turn as newline-delimited JSON (NDJSON).

    Each line is one of:
      {"type": "event",  ...AgentEvent}      — activity panel update, sent as the node completes
      {"type": "token",  "text": "..."}      — streamed answer token from the Response agent
      {"type": "result", ...ChatResponse}    — final payload (answer, citations, hitl_token, …)
      {"type": "error",  "detail": "..."}    — unhandled failure after the stream already started
    """
    await _enforce_rate_limit(user)

    async def ndjson() -> AsyncIterator[str]:
        try:
            async for chunk in stream_agent(
                message=body.message,
                session_id=body.session_id,
                user_id=user["username"],
                role=user["role"],
                require_hitl=body.require_hitl,
            ):
                if chunk["type"] == "event":
                    line = {"type": "event", **chunk["event"].model_dump(mode="json")}
                elif chunk["type"] == "token":
                    line = {"type": "token", "text": chunk["text"]}
                else:
                    response = _finalize_turn(chunk["result"], body, user)
                    line = {"type": "result", **response.model_dump(mode="json")}
                yield json.dumps(line, default=str) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("chat_stream_unhandled", error=str(exc))
            yield json.dumps({"type": "error", "detail": f"Agent failure: {exc}"}) + "\n"

    return StreamingResponse(
        ndjson(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/hitl", response_model=ChatResponse)
async def hitl_decide(body: HitlDecision, user=Depends(get_current_user)):
    """Resolve a paused research answer by resuming the LangGraph thread at its `hitl` interrupt.

    Approve → the checkpointed draft is released as-is (no re-generation).
    Send back → the Response agent revises using the reviewer's note and the same evidence;
    the revised draft pauses at HITL again (`hitl_required=True` + a fresh token).
    If the API restarted since the pause (in-memory checkpointer) → 410, ask again.
    """
    pending = decode_hitl_token(body.hitl_token)
    if pending["sub"] != user["username"] and user["role"].value != "administrator":
        raise HTTPException(status_code=403, detail="Not allowed to resolve this approval request")

    try:
        result = await run_agent(
            message=pending["msg"],
            session_id=pending["sid"],
            user_id=user["username"],
            role=user["role"],
            resume={"approved": body.approved, "note": body.note, "decided_by": user["username"]},
        )
    except NoPendingApproval as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="No draft is waiting for approval in this session (it may have expired or the service restarted) — ask the question again.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("hitl_unhandled", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Agent failure: {exc}") from exc

    hitl_token = None
    if result.get("hitl_required"):  # send-back produced a revised draft waiting again
        hitl_token = create_hitl_token(username=user["username"], session_id=pending["sid"], message=pending["msg"])

    logger.info(
        "hitl_decided",
        user=user["username"],
        approved=body.approved,
        session_id=pending["sid"],
        revised=bool(hitl_token),
    )
    citations = [Citation(**c) if not isinstance(c, Citation) else c for c in result.get("citations") or []]
    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        session_id=pending["sid"],
        events=result.get("events") or [],
        hitl_required=bool(result.get("hitl_required")),
        hitl_token=hitl_token,
        degraded=bool(result.get("degraded")),
        degradation_reason=result.get("degradation_reason"),
    )


@router.post("/feedback")
async def feedback(body: FeedbackRequest, user=Depends(get_current_user)):
    memory_store.record_feedback(
        user["username"],
        {
            "session_id": body.session_id,
            "message_id": body.message_id,
            "rating": body.rating,
            "comment": body.comment,
        },
    )
    return {"ok": True}


# --- Conversation history: a user's chats outlive the browser tab and sign-out ----------------
MAX_CONVERSATION_BYTES = 2_000_000  # a research turn with full panel events is ~100 KB


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(user=Depends(get_current_user)):
    return memory_store.list_conversations(user["username"])


@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str, user=Depends(get_current_user)):
    saved = memory_store.get_conversation(user["username"], session_id)
    if not saved:  # also the answer for someone else's session id: nothing leaks across users
        raise HTTPException(status_code=404, detail="No such conversation for this user")
    return saved


@router.put("/conversations/{session_id}")
async def save_conversation(session_id: str, body: ConversationSaveRequest, user=Depends(get_current_user)):
    if len(session_id) > 64 or not session_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid session id")
    if len(json.dumps(body.state)) > MAX_CONVERSATION_BYTES:
        raise HTTPException(status_code=413, detail="Conversation too large to save")
    memory_store.save_conversation(user["username"], session_id, body.title, body.state)
    return {"ok": True}


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str, user=Depends(get_current_user)):
    """Remove one chat: transcript, session working memory, and any paused graph checkpoint."""
    from backend.app.agents.graph import forget_thread

    existed = memory_store.delete_conversation(user["username"], session_id)
    await forget_thread(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail="No such conversation for this user")
    return {"ok": True, "deleted": session_id}


@router.delete("/conversations")
async def delete_all_conversations(
    clear_long_term: bool = False,
    user=Depends(get_current_user),
):
    """Remove every chat for this user. Pass clear_long_term=true to also forget preferences/feedback."""
    from backend.app.agents.graph import forget_thread

    # Snapshot ids before delete so we can drop their graph checkpoints too.
    ids = [c["session_id"] for c in memory_store.list_conversations(user["username"])]
    count = memory_store.delete_all_conversations(user["username"], clear_long_term=clear_long_term)
    for sid in ids:
        await forget_thread(sid)
    return {"ok": True, "deleted": count, "clear_long_term": clear_long_term}


@router.post("/admin/reindex")
async def reindex(user=Depends(get_current_user)):
    if user["role"].value != "administrator":
        raise HTTPException(status_code=403, detail="Administrator only")
    await retriever.initialize()
    return {"ok": True, "chunks": len(retriever.chunks), "mode": retriever.mode}


@router.get("/admin/feedback")
async def export_feedback(user=Depends(get_current_user)):
    """Bonus feedback-loop companion: administrators can inspect every user's ratings,
    durable preferences derived from thumbs-down comments, and recurring research topics."""
    if user["role"].value != "administrator":
        raise HTTPException(status_code=403, detail="Administrator only")
    return memory_store.export_feedback()
