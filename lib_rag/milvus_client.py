import time
from typing import List, Dict, Optional, Tuple

from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)

from .config import get_config, get_logger


RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 10.0


def _retry(operation: str, func, logger, correlation_id: Optional[str], *args, **kwargs):
    log_doc_id = kwargs.pop("doc_id", None)

    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"{operation} sucesso",
                extra={
                    "event": operation,
                    "tenant_id": get_config().tenant_id,
                    "correlation_id": correlation_id,
                    "doc_id": log_doc_id,
                    "latency_ms": latency_ms,
                },
            )
            return result
        except Exception as exc:
            last_exc = exc
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            logger.warning(
                f"{operation} falhou (tentativa {attempt}/{RETRY_ATTEMPTS}), retry em {delay}s",
                extra={
                    "event": f"{operation}_retry",
                    "tenant_id": get_config().tenant_id,
                    "correlation_id": correlation_id,
                    "doc_id": log_doc_id,
                },
            )
            if attempt == RETRY_ATTEMPTS:
                break
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def connect_milvus() -> MilvusClient:
    cfg = get_config()
    logger = get_logger(__name__)
    logger.info(
        "Conectando ao Milvus",
        extra={"event": "milvus_connect", "tenant_id": cfg.tenant_id, "correlation_id": None, "doc_id": None},
    )
    client = MilvusClient(
        uri=cfg.milvus_uri,
        user=cfg.milvus_username,
        password=cfg.milvus_password,
        db_name=cfg.milvus_db,
    )
    return client


def create_db(correlation_id: Optional[str] = None) -> None:
    """
    Cria o database se suportado. Para clusters serverless, ignorar se sem permissão.
    """
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()

    def _list():
        return client.list_databases()

    try:
        dbs = _retry("milvus_list_databases", _list, logger, correlation_id)
        if cfg.milvus_db not in dbs:
            def _create():
                client.create_database(db_name=cfg.milvus_db)

            _retry("milvus_create_database", _create, logger, correlation_id)
            logger.info(
                f"Database criado: {cfg.milvus_db}",
                extra={"event": "milvus_db_created", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
            )
    except Exception as exc:
        # Em ambientes onde create_database não é permitido, apenas logar
        logger.warning(
            f"Não foi possível criar/listar databases: {exc}",
            extra={"event": "milvus_db_create_skipped", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
        )


def _build_dense_schema(cfg) -> CollectionSchema:
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=cfg.embedding_max_length),
        FieldSchema(name="source_bucket", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=cfg.embedding_model_dim),
        FieldSchema(name="embedding_model_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="embedding_model_version", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="rag_schema_version", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
    ]
    return CollectionSchema(fields, f"RAG dense collection for tenant {cfg.tenant_id}")


def _build_hybrid_schema(cfg) -> CollectionSchema:
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=cfg.embedding_max_length),
        FieldSchema(name="source_bucket", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=cfg.embedding_model_dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="embedding_model_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="embedding_model_version", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="rag_schema_version", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
    ]
    return CollectionSchema(fields, f"RAG hybrid collection for tenant {cfg.tenant_id}")


def create_collections(correlation_id: Optional[str] = None) -> None:
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()

    # Dense
    def _has_dense():
        return client.has_collection(cfg.milvus_collection_dense)

    dense_exists = _retry("milvus_has_collection_dense", _has_dense, logger, correlation_id)
    if not dense_exists:
        dense_schema = _build_dense_schema(cfg)
        index_params = MilvusClient.prepare_index_params()
        # Índice para o vetor denso
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024},
        )

        def _create_dense():
            client.create_collection(
                collection_name=cfg.milvus_collection_dense,
                schema=dense_schema,
                index_params=index_params,
                consistency_level="Strong",
            )

        _retry("milvus_create_collection_dense", _create_dense, logger, correlation_id)
        logger.info(
            f"Collection densa criada: {cfg.milvus_collection_dense}",
            extra={"event": "milvus_collection_dense_created", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
        )

    # Hybrid
    def _has_hybrid():
        return client.has_collection(cfg.milvus_collection_hybrid)

    hybrid_exists = _retry("milvus_has_collection_hybrid", _has_hybrid, logger, correlation_id)
    if not hybrid_exists:
        hybrid_schema = _build_hybrid_schema(cfg)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024},
        )

        def _create_hybrid():
            client.create_collection(
                collection_name=cfg.milvus_collection_hybrid,
                schema=hybrid_schema,
                index_params=index_params,
                consistency_level="Strong",
            )

        _retry("milvus_create_collection_hybrid", _create_hybrid, logger, correlation_id)
        logger.info(
            f"Collection híbrida criada: {cfg.milvus_collection_hybrid}",
            extra={"event": "milvus_collection_hybrid_created", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
        )

    # Garantir load (necessário antes de search)
    def _load_dense():
        client.load_collection(cfg.milvus_collection_dense)

    def _load_hybrid():
        client.load_collection(cfg.milvus_collection_hybrid)

    _retry("milvus_load_collection_dense", _load_dense, logger, correlation_id)
    _retry("milvus_load_collection_hybrid", _load_hybrid, logger, correlation_id)


def collection_exists(collection_name: str, correlation_id: Optional[str] = None) -> bool:
    """
    Verifica se uma collection existe no Milvus.
    """
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()

    def _has():
        return client.has_collection(collection_name)

    exists = _retry("milvus_has_collection_generic", _has, logger, correlation_id)
    logger.info(
        f"collection_exists: {collection_name} -> {exists}",
        extra={
            "event": "milvus_collection_exists",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
        },
    )
    return bool(exists)


def infer_search_mode(collection_name: str, correlation_id: Optional[str] = None) -> str:
    """
    Inspeciona o schema da collection e infere o modo de busca:

    - Se existir campo 'sparse_vector' -> 'hybrid'
    - Caso contrário -> 'dense'

    NÃO depende do nome da collection.
    Baseia-se exclusivamente no retorno de MilvusClient.describe_collection().
    """
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()

    def _describe():
        return client.describe_collection(collection_name=collection_name)

    desc = _retry("milvus_describe_collection", _describe, logger, correlation_id)

    # --- Extrair lista de campos de forma robusta ---
    fields = []

    if isinstance(desc, dict):
        # MilvusClient (2.5.x) -> campos vêm direto em 'fields'
        # Exemplo: {'collection_name': 'x', 'fields': [ {...}, {...} ], ...}
        fields = desc.get("fields") or []
    else:
        # Caso o client retorne um objeto com atributo 'fields'
        maybe_fields = getattr(desc, "fields", None)
        if maybe_fields is not None:
            fields = maybe_fields
        else:
            # Último fallback: tentar achar 'schema.fields'
            schema = getattr(desc, "schema", None)
            if schema is not None:
                fields = getattr(schema, "fields", []) or getattr(schema, "fields_schema", []) or []

    field_names = set()

    for f in fields:
        if isinstance(f, dict):
            name = f.get("name") or f.get("field_name")
        else:
            name = getattr(f, "name", None) or getattr(f, "field_name", None)
        if name:
            field_names.add(name)

    has_sparse = "sparse_vector" in field_names

    mode = "hybrid" if has_sparse else "dense"

    logger.info(
        f"infer_search_mode: {collection_name} -> {mode}",
        extra={
            "event": "milvus_infer_search_mode",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
            "field_names": list(field_names),
        },
    )
    return mode


def insert_embeddings(collection_kind: str, rows: List[Dict], correlation_id: Optional[str] = None) -> None:
    """
    collection_kind: 'dense' ou 'hybrid'
    rows: lista de dicionários, UMA dict por linha, compatível com o schema do Milvus.
      Exemplo de row para coleção híbrida:
      {
        "pk": "docid_chunkid",
        "doc_id": "...",
        "chunk_id": 0,
        "text": "...",
        "source_bucket": "...",
        "source_path": "...",
        "dense_vector": [...],
        "sparse_vector": {token_idx: weight, ...},
        "embedding_model_name": "...",
        "embedding_model_version": "...",
        "rag_schema_version": "...",
        "created_at": "...",
      }
    """
    if not rows:
        return

    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()

    if collection_kind == "dense":
        collection_name = cfg.milvus_collection_dense
    elif collection_kind == "hybrid":
        collection_name = cfg.milvus_collection_hybrid
    else:
        raise ValueError("collection_kind deve ser 'dense' ou 'hybrid'")

    def _insert():
        # row-based insert: lista de dicts
        return client.insert(
            collection_name=collection_name,
            data=rows,
            progress_bar=False,
        )

    _retry("milvus_insert_embeddings", _insert, logger, correlation_id)
    logger.info(
        f"Insert em {collection_name} - {len(rows)} linhas",
        extra={
            "event": "milvus_insert_done",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
        },
    )


def search_dense(
    query_vector: List[float],
    top_k: int = 5,
    correlation_id: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Tuple[List[Dict], float]:
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()
    output_fields = [
        "doc_id",
        "chunk_id",
        "text",
        "source_bucket",
        "source_path",
        "embedding_model_name",
        "embedding_model_version",
        "rag_schema_version",
        "created_at",
    ]

    if collection_name is None:
        collection_name = cfg.milvus_collection_dense

    def _search():
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        return client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            anns_field="dense_vector",      # 👈 ESSA LINHA É O PONTO-CHAVE
            search_params=search_params,
            output_fields=output_fields,
        )

    start = time.perf_counter()
    results = _retry("milvus_search_dense", _search, logger, correlation_id)
    latency_ms = (time.perf_counter() - start) * 1000

    hits_list: List[Dict] = []
    if results:
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                hits_list.append(
                    {
                        "score": float(hit.get("distance", 0.0)),
                        "doc_id": entity.get("doc_id"),
                        "chunk_id": int(entity.get("chunk_id")),
                        "text": entity.get("text"),
                        "metadata": {
                            "source_bucket": entity.get("source_bucket"),
                            "source_path": entity.get("source_path"),
                            "embedding_model_name": entity.get("embedding_model_name"),
                            "embedding_model_version": entity.get("embedding_model_version"),
                            "rag_schema_version": entity.get("rag_schema_version"),
                            "created_at": entity.get("created_at"),
                        },
                    }
                )

    logger.info(
        f"search_dense retornou {len(hits_list)} resultados",
        extra={
            "event": "milvus_search_dense_results",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
            "latency_ms": latency_ms,
        },
    )
    return hits_list, latency_ms


def search_hybrid(
    query_dense: List[float],
    query_sparse: Dict[int, float],
    top_k: int = 5,
    correlation_id: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Tuple[List[Dict], float]:
    """
    Hybrid search (dense + sparse) com AnnSearchRequest + RRFRanker.
    Se collection_name for None, usa cfg.milvus_collection_hybrid.
    """
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()
    output_fields = [
        "doc_id",
        "chunk_id",
        "text",
        "source_bucket",
        "source_path",
        "embedding_model_name",
        "embedding_model_version",
        "rag_schema_version",
        "created_at",
    ]

    if collection_name is None:
        collection_name = cfg.milvus_collection_hybrid

    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=top_k,
    )

    dense_req = AnnSearchRequest(
        data=[query_dense],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
    )

    ranker = RRFRanker()

    def _hybrid():
        return client.hybrid_search(
            collection_name=collection_name,
            reqs=[sparse_req, dense_req],
            ranker=ranker,  # <- importante para evitar o TypeError que você viu
            limit=top_k,
            output_fields=output_fields,
        )

    start = time.perf_counter()
    results = _retry("milvus_search_hybrid", _hybrid, logger, correlation_id)
    latency_ms = (time.perf_counter() - start) * 1000

    hits_list: List[Dict] = []
    if results:
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                hits_list.append(
                    {
                        "score": float(hit.get("distance", 0.0)),
                        "doc_id": entity.get("doc_id"),
                        "chunk_id": int(entity.get("chunk_id")),
                        "text": entity.get("text"),
                        "metadata": {
                            "source_bucket": entity.get("source_bucket"),
                            "source_path": entity.get("source_path"),
                            "embedding_model_name": entity.get("embedding_model_name"),
                            "embedding_model_version": entity.get("embedding_model_version"),
                            "rag_schema_version": entity.get("rag_schema_version"),
                            "created_at": entity.get("created_at"),
                        },
                    }
                )

    logger.info(
        f"search_hybrid retornou {len(hits_list)} resultados",
        extra={
            "event": "milvus_search_hybrid_results",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
            "latency_ms": latency_ms,
        },
    )
    return hits_list, latency_ms


def delete_doc(doc_id: str, correlation_id: Optional[str] = None) -> None:
    """
    Remove um documento (todas as chunks) das collections dense e hybrid.
    """
    cfg = get_config()
    logger = get_logger(__name__)
    client = connect_milvus()

    expr = f'doc_id == "{doc_id}"'

    def _del_dense():
        return client.delete(cfg.milvus_collection_dense, filter=expr)

    def _del_hybrid():
        return client.delete(cfg.milvus_collection_hybrid, filter=expr)

    try:
        _retry("milvus_delete_dense", _del_dense, logger, correlation_id, doc_id=doc_id)
    except Exception:
        # se collection ainda não existe, ignorar
        logger.warning(
            "delete_doc: falha ao deletar em dense (ignorando)",
            extra={"event": "milvus_delete_dense_failed", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
        )
    try:
        _retry("milvus_delete_hybrid", _del_hybrid, logger, correlation_id, doc_id=doc_id)
    except Exception:
        logger.warning(
            "delete_doc: falha ao deletar em hybrid (ignorando)",
            extra={"event": "milvus_delete_hybrid_failed", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
        )
