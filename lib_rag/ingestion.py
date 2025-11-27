import hashlib
import os
import time
from datetime import datetime
from typing import Optional, List

from .config import get_config, get_logger
from .minio_client import (
    stat_object,
    download_file,
    upload_file,
    move_file,
    list_new_files,
)
from .chunking import extract_text_from_file, chunk_document_text
from .embeddings import generate_dense, generate_hybrid
from .milvus_client import create_collections, insert_embeddings, delete_doc


def _compute_doc_id(bucket: str, key: str, etag: Optional[str]) -> str:
    raw = f"{bucket}:{key}:{etag or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def delete_local_files(paths: List[str], correlation_id: Optional[str] = None) -> None:
    cfg = get_config()
    logger = get_logger(__name__)
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
                logger.info(
                    f"Arquivo local removido: {p}",
                    extra={"event": "local_file_deleted", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
                )
        except Exception as exc:
            logger.warning(
                f"Falha ao remover arquivo local {p}: {exc}",
                extra={"event": "local_file_delete_failed", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
            )


def ingest_single_object(
    bucket: str,
    key: str,
    correlation_id: Optional[str] = None,
) -> str:
    """
    Pipeline idempotente de ingestão de um único objeto:
    raw -> tmp -> extração -> chunking -> embeddings -> indexação -> processed.
    Retorna doc_id.
    """
    cfg = get_config()
    logger = get_logger(__name__)

    # Info do objeto para etag
    obj_stat = stat_object(bucket, key, correlation_id)
    etag = getattr(obj_stat, "etag", None)
    doc_id = _compute_doc_id(bucket, key, etag)

    logger.info(
        f"Iniciando ingestão de {bucket}/{key} doc_id={doc_id}",
        extra={"event": "ingest_start", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
    )

    # Caminho tmp local
    filename = os.path.basename(key)
    tmp_dir = os.path.join(cfg.path_tmp, cfg.tenant_id, doc_id)
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, filename)

    try:
        # 1. raw -> tmp
        download_file(bucket, key, local_path, correlation_id, doc_id=doc_id)

        # 2. extração (Docling)
        text = extract_text_from_file(local_path, correlation_id, doc_id)

        # 3. chunking
        chunks = chunk_document_text(text, correlation_id, doc_id)
        if not chunks:
            logger.warning(
                "Nenhum chunk gerado; movendo para processed mesmo assim",
                extra={"event": "ingest_no_chunks", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
            )
        # 4. embeddings
        chunk_texts = [c["text"] for c in chunks]
        embedding_mode = cfg.embedding_mode

        if embedding_mode == "dense":
            dense_vectors = generate_dense(chunk_texts)
            sparse_vectors = None
        elif embedding_mode in {"hybrid", "all"}:
            dense_vectors, sparse_vectors = generate_hybrid(chunk_texts)
        else:
            raise ValueError("EMBEDDING_MODE inválido")

        # 5. indexação (idempotente: apagar doc anterior)
        create_collections(correlation_id)
        delete_doc(doc_id, correlation_id)

        created_at = datetime.utcnow().isoformat() + "Z"
        rows_dense = []
        rows_hybrid = []

        for idx, ch in enumerate(chunks):
            base = {
                "pk": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "chunk_id": int(ch["chunk_id"]),
                "text": ch["text"],
                "source_bucket": bucket,
                "source_path": key,
                "dense_vector": dense_vectors[idx],
                "embedding_model_name": cfg.embedding_model_name,
                "embedding_model_version": cfg.embedding_model_version,
                "rag_schema_version": cfg.rag_schema_version,
                "created_at": created_at,
            }
            if embedding_mode in {"dense", "all"}:
                rows_dense.append(base.copy())
            if embedding_mode in {"hybrid", "all"} and sparse_vectors is not None:
                hybrid_row = base.copy()
                hybrid_row["sparse_vector"] = sparse_vectors[idx]
                rows_hybrid.append(hybrid_row)

        if rows_dense:
            insert_embeddings("dense", rows_dense, correlation_id)
        if rows_hybrid:
            insert_embeddings("hybrid", rows_hybrid, correlation_id)

        # 6. move para processed
        processed_key = key.replace(f"raw/{cfg.tenant_id}/", f"processed/{cfg.tenant_id}/", 1)
        move_file(cfg.minio_bucket_raw, key, cfg.minio_bucket_processed, processed_key, correlation_id, doc_id=doc_id)

        logger.info(
            f"Ingestão concluída para doc_id={doc_id}",
            extra={"event": "ingest_done", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
        )
        return doc_id

    finally:
        delete_local_files([local_path], correlation_id)


def run_ingestion_batch(correlation_id: Optional[str] = None) -> None:
    """
    Executa ingestão em batch para todos arquivos novos em raw/{TENANT_ID}/.
    """
    cfg = get_config()
    logger = get_logger(__name__)
    objects = list_new_files(correlation_id)
    logger.info(
        f"run_ingestion_batch encontrou {len(objects)} objetos",
        extra={"event": "ingest_batch_start", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
    )

    for obj in objects:
        try:
            ingest_single_object(obj["bucket"], obj["key"], correlation_id)
        except Exception as exc:
            logger.error(
                f"Falha na ingestão de {obj['bucket']}/{obj['key']}: {exc}",
                extra={"event": "ingest_error", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
            )
