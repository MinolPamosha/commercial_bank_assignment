"""LLM factory with graceful degradation when keys are missing."""

from __future__ import annotations

from typing import Any, Optional

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger
from backend.app.security.guardrails import brand_system_preamble

logger = get_logger(__name__)


def get_chat_model():
    """Build the chat model for the configured provider.

    - openai   → api.openai.com, model e.g. 'gpt-4o-mini'
    - requesty → router.requesty.ai (OpenAI-compatible), model e.g. 'openai/gpt-4o-mini'
    Both go through langchain_openai.ChatOpenAI so LangSmith tracing is identical.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        logger.warning(
            "llm_offline_mode",
            reason=f"{settings.llm_provider.upper()}_API_KEY missing",
        )
        return None
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": settings.resolved_llm_model,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_sec,
        "api_key": settings.llm_api_key,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
        # Optional analytics headers Requesty recognises
        kwargs["default_headers"] = {"X-Title": settings.assistant_name}
    return ChatOpenAI(**kwargs)


async def acomplete(system: str, user: str) -> str:
    """Async completion; falls back to deterministic stub for offline demos."""
    model = get_chat_model()
    if model is None:
        settings = get_settings()
        return (
            f"[Offline stub — configure {settings.llm_provider.upper()}_API_KEY for live answers]\n"
            f"Based on available evidence context, here is a structured reply for Apex Commercial Bank.\n\n{user[:1500]}"
        )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = await model.ainvoke(
            [
                SystemMessage(content=f"{brand_system_preamble()}\n{system}"),
                HumanMessage(content=user),
            ]
        )
        return str(resp.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("llm_failed", error=str(exc))
        raise


def heuristic_intent(message: str) -> tuple[str, str]:
    """Deterministic router used when LLM is unavailable or as a fast path."""
    m = message.lower()
    if any(k in m for k in ("summarize all", "recurring", "root cause", "across", "last year", "batch")):
        return "research", "research"
    # Counting / averaging over the written reports is document analysis, not a live ticket lookup.
    if any(k in m for k in ("how many", "average", "count", "calculate", "statistics")) and any(
        k in m for k in ("incident", "outage", "report", "document", "polic")
    ):
        return "research", "research"
    if any(k in m for k in ("employee", "who is", "on call", "on-call", "directory", "service catalog", "open incident", "incident record")):
        return "tool", "tools"
    if any(k in m for k in ("reindex", "admin", "export feedback")):
        return "admin", "tools"
    if len(m.split()) < 3 and any(k in m for k in ("hi", "hello", "thanks")):
        return "chitchat", "respond"
    return "qa", "retrieve"
