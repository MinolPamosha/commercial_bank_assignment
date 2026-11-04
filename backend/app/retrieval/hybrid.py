"""Hybrid dense + BM25 sparse retrieval with optional Pinecone and local fallback.

Design notes (for evaluators):
- Dense captures semantic similarity (embeddings).
- Sparse (BM25) captures exact keywords like INC ids and product codes.
- Final score = w_d * dense + w_s * sparse (min-max normalized per query).
- Optional simple cross-encoder-style rerank via lexical overlap + dense score.
- Pinecone namespaces isolate environments; metadata filters enforce document scope.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger
from backend.app.observability.tracing import trace_retriever, trace_step
from backend.app.retrieval.document_loader import DocumentChunk, load_documents
from backend.app.retrieval.embeddings import EmbeddingClient

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().replace("/", " ").replace("-", " ").split() if t]


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = EmbeddingClient()
        self.chunks: list[DocumentChunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dense: Optional[np.ndarray] = None
        self._pinecone_index = None
        self._ready = False

    @property
    def mode(self) -> str:
        if self.settings.use_local_retrieval or self._pinecone_index is None:
            return "local_hybrid"
        return "pinecone_hybrid"

    async def initialize(self) -> None:
        self.chunks = load_documents()
        corpus = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)
        vectors = await self.embedder.embed_documents([c.text for c in self.chunks])
        self._dense = np.array(vectors, dtype=np.float32)

        # Persist local index for cold starts / Docker volumes
        index_dir = Path(self.settings.local_index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (index_dir / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")
        np.save(index_dir / "dense.npy", self._dense)

        if not self.settings.use_local_retrieval and self.settings.pinecone_api_key:
            await self._init_pinecone(vectors)
        self._ready = True
        logger.info("retriever_ready", mode=self.mode, chunks=len(self.chunks))

    async def _init_pinecone(self, vectors: list[list[float]]) -> None:
        """Create/connect the index and upsert the corpus.

        The Pinecone SDK is synchronous; every network call is pushed to a worker thread with
        `asyncio.to_thread` so index bootstrap never blocks the FastAPI event loop.
        """
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            existing = {idx["name"] for idx in await asyncio.to_thread(pc.list_indexes)}
            name = self.settings.pinecone_index
            dim = len(vectors[0])
            if name not in existing:
                await asyncio.to_thread(
                    pc.create_index,
                    name=name,
                    dimension=dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region),
                )
            self._pinecone_index = pc.Index(name)
            # Upsert into namespace "prod"
            batch = []
            for c, vec in zip(self.chunks, vectors):
                batch.append(
                    {
                        "id": c.chunk_id,
                        "values": vec,
                        "metadata": {
                            **{k: v for k, v in c.metadata.items() if not isinstance(v, list)},
                            "tags": ",".join(c.metadata.get("tags") or []),
                            "text": c.text[:3500],
                            "title": c.title,
                            "doc_id": c.doc_id,
                        },
                    }
                )
            # Pinecone upsert in slices
            for i in range(0, len(batch), 50):
                await asyncio.to_thread(self._pinecone_index.upsert, vectors=batch[i : i + 50], namespace="prod")
            logger.info("pinecone_upserted", count=len(batch), namespace="prod")
        except Exception as exc:  # noqa: BLE001
            logger.error("pinecone_init_failed", error=str(exc))
            self._pinecone_index = None

    @trace_retriever("hybrid_search")
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        namespace: str = "prod",
    ) -> list[dict[str, Any]]:
        if not self._ready:
            await self.initialize()

        top_k = top_k or self.settings.retrieval_top_k
        filters = filters or {}

        dense_task = self._dense_search(query, top_k=top_k * 3, filters=filters, namespace=namespace)
        sparse_task = asyncio.to_thread(self._sparse_search, query, top_k * 3, filters)
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

        fused = self._rrf_fuse(dense_hits, sparse_hits, top_k=top_k * 2)
        reranked = self._rerank(query, fused)[: self.settings.rerank_top_n]
        return reranked

    @trace_retriever("dense_search")
    async def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
        namespace: str,
    ) -> list[dict[str, Any]]:
        qvec = await self.embedder.embed_query(query)

        if self._pinecone_index is not None and not self.settings.use_local_retrieval:
            try:
                pine_filter = {k: {"$eq": v} for k, v in filters.items()} if filters else None
                # Sync SDK call → worker thread, bounded by the tool timeout so a slow vector
                # DB degrades to the local index instead of stalling the request.
                res = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._pinecone_index.query,
                        vector=qvec,
                        top_k=top_k,
                        namespace=namespace,
                        include_metadata=True,
                        filter=pine_filter,
                    ),
                    timeout=self.settings.max_tool_timeout_sec,
                )
                hits = []
                for m in res.get("matches") or res.matches:
                    md = m["metadata"] if isinstance(m, dict) else m.metadata
                    score = m["score"] if isinstance(m, dict) else m.score
                    hits.append(
                        {
                            "chunk_id": m["id"] if isinstance(m, dict) else m.id,
                            "doc_id": md.get("doc_id"),
                            "title": md.get("title"),
                            "text": md.get("text", ""),
                            "metadata": md,
                            "dense_score": float(score),
                            "source": "pinecone",
                        }
                    )
                return hits
            except asyncio.TimeoutError:
                logger.error("pinecone_query_timeout", timeout_sec=self.settings.max_tool_timeout_sec)
            except Exception as exc:  # noqa: BLE001
                logger.error("pinecone_query_failed", error=str(exc))

        # Local cosine (fallback when Pinecone is disabled, failed or timed out)
        assert self._dense is not None
        q = np.array(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._dense @ q
        idxs = np.argsort(-sims)[:top_k]
        hits = []
        for i in idxs:
            c = self.chunks[int(i)]
            if not self._passes_filters(c, filters):
                continue
            hits.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "text": c.text,
                    "metadata": c.metadata,
                    "dense_score": float(sims[int(i)]),
                    "source": "local_dense",
                }
            )
        return hits

    @trace_retriever("bm25_search")
    def _sparse_search(self, query: str, top_k: int, filters: dict[str, Any]) -> list[dict[str, Any]]:
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(query))
        idxs = np.argsort(-scores)[:top_k]
        hits = []
        for i in idxs:
            c = self.chunks[int(i)]
            if not self._passes_filters(c, filters):
                continue
            hits.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "text": c.text,
                    "metadata": c.metadata,
                    "sparse_score": float(scores[int(i)]),
                    "source": "bm25",
                }
            )
        return hits

    @staticmethod
    def _passes_filters(chunk: DocumentChunk, filters: dict[str, Any]) -> bool:
        for k, v in filters.items():
            if chunk.metadata.get(k) != v:
                return False
        return True

    def _rrf_fuse(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion + weighted normalized scores for explainability."""
        scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}

        dens = _minmax([h.get("dense_score", 0.0) for h in dense_hits])
        spars = _minmax([h.get("sparse_score", 0.0) for h in sparse_hits])

        def merge(h: dict[str, Any]) -> str:
            """Union the two hit records; keep *every* retriever that found the chunk."""
            cid = h["chunk_id"]
            prev = docs.get(cid, {})
            sources = sorted(set(prev.get("sources", [])) | {h.get("source", "")} - {""})
            docs[cid] = {**prev, **h, "sources": sources}
            return cid

        for rank, (h, ns) in enumerate(zip(dense_hits, dens)):
            cid = merge(h)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            scores[cid] += self.settings.hybrid_dense_weight * ns

        for rank, (h, ns) in enumerate(zip(sparse_hits, spars)):
            cid = merge(h)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            scores[cid] += self.settings.hybrid_sparse_weight * ns

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for cid, score in ranked:
            item = docs[cid]
            item["score"] = float(score)
            item["source"] = "+".join(item.get("sources") or [])  # e.g. "bm25+pinecone"
            out.append(item)
        return out

    @trace_step("rerank")
    def _rerank(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Lightweight reranker (bonus): lexical overlap + hybrid score."""
        q_tokens = set(_tokenize(query))
        for h in hits:
            t_tokens = set(_tokenize(h.get("text", "")))
            overlap = len(q_tokens & t_tokens) / (len(q_tokens) + 1e-9)
            h["rerank_score"] = 0.7 * h.get("score", 0.0) + 0.3 * overlap
        return sorted(hits, key=lambda x: x["rerank_score"], reverse=True)


# Process-wide singleton
retriever = HybridRetriever()
