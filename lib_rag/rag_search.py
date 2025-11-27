from typing import List, Dict, Optional
import time

from .config import get_config, get_logger
from .embeddings import generate_dense, generate_hybrid
from .milvus_client import search_dense as milvus_search_dense, search_hybrid as milvus_search_hybrid


def dense_search(
    query: str,
    top_k: int = 5,
    correlation_id: Optional[str] = None,
) -> List[Dict]:
    """
    Retorna estrutura:

    [
      {
        "doc_id": str,
        "chunk_id": int,
        "text": str,
        "score": float,
        "metadata": {...},
        "latency_ms": float
      }
    ]
    """
    cfg = get_config()
    logger = get_logger(__name__)

    start_emb = time.perf_counter()
    q_vec = generate_dense([query])[0]
    emb_latency = (time.perf_counter() - start_emb) * 1000

    hits, search_latency = milvus_search_dense(q_vec, top_k=top_k, correlation_id=correlation_id)
    total_latency = emb_latency + search_latency

    results: List[Dict] = []
    for h in hits:
        results.append(
            {
                "doc_id": h["doc_id"],
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "score": h["score"],
                "metadata": h["metadata"],
                "latency_ms": total_latency,
            }
        )

    logger.info(
        f"dense_search retornou {len(results)} resultados",
        extra={
            "event": "rag_dense_search",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
        },
    )
    return results


def hybrid_search(
    query: str,
    top_k: int = 5,
    correlation_id: Optional[str] = None,
) -> List[Dict]:
    cfg = get_config()
    logger = get_logger(__name__)

    start_emb = time.perf_counter()
    dense, sparse = generate_hybrid([query])
    q_dense = dense[0]
    q_sparse = sparse[0]
    emb_latency = (time.perf_counter() - start_emb) * 1000

    hits, search_latency = milvus_search_hybrid(q_dense, q_sparse, top_k=top_k, correlation_id=correlation_id)
    total_latency = emb_latency + search_latency

    results: List[Dict] = []
    for h in hits:
        results.append(
            {
                "doc_id": h["doc_id"],
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "score": h["score"],
                "metadata": h["metadata"],
                "latency_ms": total_latency,
            }
        )

    logger.info(
        f"hybrid_search retornou {len(results)} resultados",
        extra={
            "event": "rag_hybrid_search",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
        },
    )
    return results
