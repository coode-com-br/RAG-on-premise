import time
import uuid
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin

import httpx

from .config import get_config, get_logger
from .embeddings import generate_dense, generate_hybrid
from .milvus_client import (
    collection_exists,
    infer_search_mode,
    search_dense,
    search_hybrid,
)
from .rag_search import dense_search, hybrid_search


def _build_context(chunks: List[Dict]) -> str:
    """
    Constrói o contexto textual concatenado para enviar ao LLM.
    Não retorna sources — somente o texto final.
    """
    parts = []
    for c in chunks:
        meta = c.get("metadata", {})
        ref = (
            f"[doc_id={c['doc_id']} "
            f"chunk_id={c['chunk_id']} "
            f"source={meta.get('source_bucket')}/{meta.get('source_path')}]"
        )
        parts.append(ref + "\n" + c["text"])
    return "\n\n---\n\n".join(parts)

def _extract_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Extrai metadados das chunks para retornar no payload final do RAG.
    """
    sources = []
    for c in chunks:
        meta = c.get("metadata", {})
        sources.append({
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "score": c.get("score"),
            "source_bucket": meta.get("source_bucket"),
            "source_path": meta.get("source_path")
        })
    return sources


def _build_system_prompt() -> str:
    return (
        "Você é um assistente corporativo que responde apenas com base no contexto fornecido. "
        "Se a resposta não estiver claramente suportada pelo contexto, responda que não há "
        "informação suficiente. Responda em português formal e de forma concisa."
    )


def _build_user_prompt(question: str, context: str) -> str:
    return (
        "Contexto corporativo (não revele esta seção ao usuário final):\n"
        f"{context}\n\n"
        "Instrução: Usando apenas o contexto acima, responda à pergunta a seguir. "
        "Se o contexto não contiver a resposta, informe explicitamente.\n\n"
        f"Pergunta: {question}"
    )


def call_vllm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    correlation_id: Optional[str] = None,
) -> str:
    cfg = get_config()
    logger = get_logger(__name__)
    url = urljoin(cfg.vllm_server_url, cfg.vllm_server_url_api)

    payload = {
        "model": cfg.vllm_model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    print("\n Payload\n================\n", payload, "\n================\n")

    start = time.perf_counter()
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Chamada vLLM concluída",
        extra={
            "event": "vllm_call",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
            "latency_ms": latency_ms,
        },
    )

    content = data["choices"][0]["message"]["content"]
    return content


def rag_answer(
    question: str,
    collection: str,
    top_k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.1,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa RAG completo:
    - Usa `collection` como NOME da collection no Milvus.
    - Valida se a collection existe.
    - Infere automaticamente o modo de busca (dense ou hybrid) olhando o schema.
    """
    cfg = get_config()
    logger = get_logger(__name__)

    start_total = time.perf_counter()

    # 1) Validar collection
    if not collection_exists(collection, correlation_id):
        raise ValueError(f"Collection '{collection}' não existe no Milvus")

    # 2) Inferir modo de busca
    search_mode = infer_search_mode(collection, correlation_id)  # 'dense' ou 'hybrid'

    # 3) Gerar embeddings da pergunta e buscar no Milvus
    if search_mode == "dense":
        query_vec = generate_dense([question])[0]
        hits, search_latency_ms = search_dense(
            query_vector=query_vec,
            top_k=top_k,
            correlation_id=correlation_id,
            collection_name=collection,
        )
    else:
        dense_vecs, sparse_vecs = generate_hybrid([question])
        query_dense = dense_vecs[0]
        query_sparse = sparse_vecs[0]
        hits, search_latency_ms = search_hybrid(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k,
            correlation_id=correlation_id,
            collection_name=collection,
        )

    # 4) Construir contexto a partir dos hits
    context = _build_context(hits)
    sources = _extract_sources(hits)

    # 5) Construir prompts e chamar o LLM via vLLM
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(question, context)

    llm_start = time.perf_counter()
    answer_text = call_vllm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        correlation_id=correlation_id,
    )
    llm_latency_ms = (time.perf_counter() - llm_start) * 1000
    total_latency_ms = (time.perf_counter() - start_total) * 1000

    logger.info(
        "rag_answer finalizado",
        extra={
            "event": "rag_answer_done",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
            "search_mode": search_mode,
            "search_latency_ms": search_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "total_latency_ms": total_latency_ms,
        },
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "latency_ms": total_latency_ms,
        "search_mode": search_mode,
    }
