"""Async embedding helper with OpenAI; deterministic hash fallback for offline demos."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)


def _hash_embed(text: str, dim: int = 384) -> list[float]:
    """Offline-safe pseudo-embedding so hybrid search still works without API keys."""
    vec = [0.0] * dim
    tokens = text.lower().split()
    for tok in tokens:
        digest = hashlib.sha256(tok.encode()).digest()
        for i in range(0, min(len(digest), 32)):
            idx = digest[i] % dim
            vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class EmbeddingClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        # Same provider switch as chat: direct OpenAI or Requesty gateway (/v1/embeddings).
        if self.settings.llm_enabled:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.settings.llm_api_key,
                    base_url=self.settings.llm_base_url,  # None → api.openai.com
                )
                logger.info(
                    "embedding_client_ready",
                    provider=self.settings.llm_provider,
                    model=self.settings.resolved_embedding_model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("embed_client_init_failed", error=str(exc))

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None:
            return [_hash_embed(t) for t in texts]
        try:
            resp = await self._client.embeddings.create(
                model=self.settings.resolved_embedding_model,
                input=list(texts),
            )
            return [item.embedding for item in resp.data]
        except Exception as exc:  # noqa: BLE001
            logger.error("embed_failed_fallback_hash", error=str(exc))
            return [_hash_embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
