"""LangSmith tracing — mandatory observability requirement.

What ends up in a trace for one chat turn:

    chat_turn (chain)  ← root; metadata: user_id, role, session_id, intent
    ├── input_guardrail (chain)
    ├── LangGraph (chain)
    │   ├── supervisor (chain) → ChatOpenAI (llm)
    │   ├── research / retrieval / tools (chain)
    │   │   ├── knowledge_search (tool)
    │   │   │   └── hybrid_search (retriever) → dense_search, bm25_search, rerank
    │   │   ├── python_analysis (tool)
    │   │   └── mcp_* (tool)
    │   ├── response (chain) → ChatOpenAI (llm)
    │   └── validate / hitl (chain)
    └── memory_update (chain)

LangGraph nodes and ChatOpenAI calls are traced automatically by LangChain callbacks.
Tools and the retriever are plain async functions, so they are wrapped with
`langsmith.traceable` (`trace_tool`, `trace_retriever` below) — without that, tool calls and
retrieval operations would be invisible in LangSmith, which the assignment explicitly requires.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langsmith import traceable
from langsmith import utils as ls_utils

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)


def configure_langsmith() -> bool:
    """Propagate LangSmith env vars so LangChain auto-traces graph runs.

    Sets both LANGCHAIN_* (legacy, still used by LangChain/LangGraph SDKs) and
    LANGSMITH_* (current LangSmith UI / agent addressing) from the same key.
    """
    settings = get_settings()
    if not settings.tracing_enabled:
        logger.warning("langsmith_disabled", reason="LANGCHAIN_API_KEY missing or LANGCHAIN_TRACING_V2=false")
        return False

    # Legacy names — required for langchain/langgraph auto-tracing today
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    # Current LangSmith names (shown in the Connect-an-agent wizard)
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langchain_tracing_v2 else "false"
    os.environ["LANGSMITH_API_KEY"] = settings.langchain_api_key
    os.environ["LANGSMITH_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langchain_project
    os.environ.setdefault("LANGSMITH_AGENT_ID", settings.langchain_project)
    os.environ.setdefault("LANGSMITH_AGENT_ENVIRONMENT", "local")

    # langsmith memoises env lookups (`get_env_var` is lru_cached). If anything evaluated
    # `tracing_is_enabled()` before this ran, it cached "disabled" and every `@traceable`
    # span would be silently dropped — clear the cache so the new values are seen.
    ls_utils.get_env_var.cache_clear()

    logger.info("langsmith_enabled", project=settings.langchain_project)
    return True


# ---- Decorators for non-LangChain code paths ---------------------------------------------

_SKIP_INPUT_KEYS = {"self"}
_MAX_INPUT_CHARS = 2000


def _compact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep traces readable: drop `self`, truncate long strings (e.g. snippet lists)."""
    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if k in _SKIP_INPUT_KEYS:
            continue
        if isinstance(v, str) and len(v) > _MAX_INPUT_CHARS:
            v = v[:_MAX_INPUT_CHARS] + f"… (+{len(v) - _MAX_INPUT_CHARS} chars)"
        elif isinstance(v, list) and len(v) > 20:
            v = v[:20] + [f"… (+{len(v) - 20} more)"]
        out[k] = v
    return out


def trace_tool(name: str) -> Callable:
    """Mark an async tool function as a LangSmith `tool` run (nests under the calling node)."""
    return traceable(run_type="tool", name=name, process_inputs=_compact_inputs)


def trace_retriever(name: str) -> Callable:
    """Mark a retrieval function as a LangSmith `retriever` run."""
    return traceable(run_type="retriever", name=name, process_inputs=_compact_inputs)


def trace_step(name: str) -> Callable:
    """Generic chain-type span for pre/post-graph steps (guardrails, memory)."""
    return traceable(run_type="chain", name=name, process_inputs=_compact_inputs)
