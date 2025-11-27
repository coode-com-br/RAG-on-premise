from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np
from pymilvus import model

from .config import get_config, get_logger


@lru_cache()
def get_embedding_model():
    """
    Usa BGEM3EmbeddingFunction (BAAI/bge-m3) do pymilvus.model.hybrid.
    Requer instalação de 'pymilvus[model]' + FlagEmbedding.
    """
    cfg = get_config()
    logger = get_logger(__name__)
    logger.info(
        "Inicializando BGEM3EmbeddingFunction",
        extra={"event": "embedding_model_init", "tenant_id": cfg.tenant_id, "correlation_id": None, "doc_id": None},
    )
    ef = model.hybrid.BGEM3EmbeddingFunction(
        model_name=cfg.embedding_model_name,
        batch_size=cfg.embedding_batch_size,
        device=cfg.embedding_device,
        use_fp16=True,
        normalize_embeddings=True,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    # Validação de dimensão
    dense_dim = ef.dim["dense"] if isinstance(ef.dim, dict) else ef.dim
    if dense_dim != cfg.embedding_model_dim:
        raise RuntimeError(
            f"Dimensão de embedding divergente: modelo {dense_dim}, configuração EMBEDDING_MODEL_DIM={cfg.embedding_model_dim}"
        )
    return ef


def _truncate_texts(texts: List[str]) -> List[str]:
    """
    Trunca textos para EMBEDDING_MAX_LENGTH caracteres (aproximação de tokens).
    """
    cfg = get_config()
    max_len = cfg.embedding_max_length
    return [t[:max_len] for t in texts]


def generate_dense(texts: List[str]) -> List[List[float]]:
    """
    Gera apenas vetores densos.
    """
    ef = get_embedding_model()
    truncated = _truncate_texts(texts)
    out = ef(truncated)
    dense = out["dense"]
    # Garantir listas nativas Python
    return [np.asarray(v, dtype="float32").tolist() for v in dense]


def generate_hybrid(texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
    """
    Gera vetores densos e esparsos no formato esperado pelo Milvus:

    - dense_vectors: List[List[float]]
    - sparse_vectors: List[Dict[int, float]]   # um dicionário por linha (token_idx -> weight)
    """
    ef = get_embedding_model()
    truncated = _truncate_texts(texts)

    # Use a chamada que você já vinha usando para o BGEM3EmbeddingFunction.
    # Se antes era ef(truncated), mantenha assim:
    out = ef(truncated)
    # (Se você estiver usando ef.encode_documents(truncated), troque a linha acima por ela.)

    # Dense: lista de vetores numpy -> lista de listas de float32
    dense_vectors = [np.asarray(v, dtype="float32").tolist() for v in out["dense"]]

    raw_sparse = out["sparse"]
    sparse_vectors: List[Dict[int, float]] = []

    # Caso 1: já é lista de dicts (algumas versões do wrapper fazem isso)
    if isinstance(raw_sparse, list) and raw_sparse and isinstance(raw_sparse[0], dict):
        for row in raw_sparse:
            sparse_vectors.append({int(k): float(v) for k, v in row.items()})
        return dense_vectors, sparse_vectors

    # Caso 2: matriz esparsa 2D (coo_array, coo_matrix, csr_array, csr_matrix, etc.)
    # Convertemos para CSR e iteramos usando indptr/indices/data.
    if hasattr(raw_sparse, "tocsr"):
        sparse_csr = raw_sparse.tocsr()
    else:
        raise RuntimeError(f"Formato inesperado para sparse embeddings: {type(raw_sparse)}")

    indptr = sparse_csr.indptr
    indices = sparse_csr.indices
    data = sparse_csr.data
    n_rows = sparse_csr.shape[0]

    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        row_dict = {int(idx): float(val) for idx, val in zip(row_indices, row_data)}
        sparse_vectors.append(row_dict)

    return dense_vectors, sparse_vectors
