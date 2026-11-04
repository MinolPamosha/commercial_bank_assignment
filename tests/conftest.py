"""Test configuration — everything runs OFFLINE.

Environment variables are set *before* the app is imported so that `Settings` (lru_cached)
resolves to: no LLM key (deterministic stubs), no LangSmith, local hybrid retrieval over the
real corpus, and an unreachable MCP server (exercises the degraded path).
Environment variables take precedence over `.env`, so a developer's real keys are never used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

_OFFLINE_ENV = {
    "LLM_PROVIDER": "openai",
    "OPENAI_API_KEY": "",
    "REQUESTY_API_KEY": "",
    "PINECONE_API_KEY": "",
    "USE_LOCAL_RETRIEVAL": "true",
    "LANGCHAIN_API_KEY": "",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "MCP_BASE_URL": "http://127.0.0.1:9",  # nothing listens here → MCP failure path
    "MAX_TOOL_TIMEOUT_SEC": "2",
    "RATE_LIMIT_TOKENS": "3",
    "RATE_LIMIT_REFILL_PER_SEC": "0.001",
    "APP_SECRET_KEY": "test-secret",
    "LOCAL_INDEX_PATH": "data/index_test",
    "LOG_LEVEL": "WARNING",
}
for k, v in _OFFLINE_ENV.items():
    os.environ[k] = v

import pytest  # noqa: E402

from backend.app.models.schemas import Role  # noqa: E402
from backend.app.retrieval.hybrid import retriever  # noqa: E402


@pytest.fixture(scope="session")
def roles() -> dict[str, Role]:
    return {"viewer": Role.VIEWER, "analyst": Role.ANALYST, "admin": Role.ADMINISTRATOR}


@pytest.fixture(scope="session")
async def ready_retriever():
    """Index the real sample corpus once (hash embeddings + BM25, no network)."""
    if not retriever._ready:
        await retriever.initialize()
    return retriever
