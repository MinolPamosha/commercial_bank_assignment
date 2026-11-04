"""Hybrid retrieval: score fusion, metadata filters, attribution, access-level enforcement."""

from backend.app.models.schemas import Role
from backend.app.retrieval.hybrid import _minmax
from backend.app.tools.knowledge_search import knowledge_search


def test_minmax_normalisation():
    assert _minmax([]) == []
    assert _minmax([5.0, 5.0]) == [0.0, 0.0]
    assert _minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_rerank_boosts_lexical_overlap_over_hybrid_score_alone(ready_retriever):
    """Bonus rerank layer: a higher hybrid score without query words loses to a hit that matches them."""
    hits = [
        {"chunk_id": "weak_words", "score": 0.80, "text": "unrelated ledger reconciliation notes"},
        {"chunk_id": "strong_words", "score": 0.45, "text": "payment gateway timeout during UPI settlement lag"},
    ]
    ranked = ready_retriever._rerank("payment gateway timeout UPI settlement", hits)
    assert ranked[0]["chunk_id"] == "strong_words"
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_rrf_fusion_unions_sources_and_keeps_both_scores(ready_retriever):
    dense = [
        {"chunk_id": "a", "dense_score": 0.9, "source": "local_dense"},
        {"chunk_id": "b", "dense_score": 0.8, "source": "local_dense"},
        {"chunk_id": "d", "dense_score": 0.1, "source": "local_dense"},
    ]
    sparse = [
        {"chunk_id": "b", "sparse_score": 7.0, "source": "bm25"},
        {"chunk_id": "c", "sparse_score": 2.0, "source": "bm25"},
        {"chunk_id": "d", "sparse_score": 1.0, "source": "bm25"},
    ]
    fused = ready_retriever._rrf_fuse(dense, sparse, top_k=10)
    by_id = {h["chunk_id"]: h for h in fused}
    # "b" ranks high in both channels → fused to the top, both raw scores preserved
    assert fused[0]["chunk_id"] == "b"
    assert by_id["b"]["source"] == "bm25+local_dense"
    assert "dense_score" in by_id["b"] and "sparse_score" in by_id["b"]
    assert by_id["a"]["source"] == "local_dense" and by_id["c"]["source"] == "bm25"
    assert all("score" in h for h in fused)
    assert fused[-1]["chunk_id"] == "d"  # weak in both channels → last


async def test_exact_document_id_is_found_via_sparse_channel(ready_retriever):
    hits = await ready_retriever.search("INC-2025-0142")
    assert hits and hits[0]["doc_id"] == "INC-2025-0142"


async def test_metadata_filters_restrict_results(ready_retriever):
    hits = await ready_retriever.search("payment outage", filters={"document_type": "policy"})
    assert hits
    assert all(h["metadata"]["document_type"] == "policy" for h in hits)

    hits = await ready_retriever.search("payment", filters={"department": "cards"})
    assert all(h["metadata"]["department"] == "cards" for h in hits)


async def test_knowledge_search_applies_access_level_by_role(ready_retriever):
    q = "UPI settlement lag redis connection pool INC-2025-0142"
    analyst = await knowledge_search(q, role=Role.ANALYST)
    viewer = await knowledge_search(q, role=Role.VIEWER)
    assert analyst["ok"] and viewer["ok"]
    assert "INC-2025-0142" in {r["doc_id"] for r in analyst["results"]}  # confidential → analyst sees it
    assert "INC-2025-0142" not in {r["doc_id"] for r in viewer["results"]}  # viewer never does
    assert all(r["metadata"]["access_level"] in {"public", "internal"} for r in viewer["results"])
