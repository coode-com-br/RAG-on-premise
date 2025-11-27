import os
import sys
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime
from typing import Optional


class JsonFormatter(logging.Formatter):
    """JSON formatter para logs estruturados."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "module": record.name,
            "event": getattr(record, "event", None) or record.getMessage().split(" ")[0],
            "tenant_id": getattr(record, "tenant_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
            "doc_id": getattr(record, "doc_id", None),
        }
        # mensagem completa fica em "message" (não incluir payload sensível)
        log["message"] = record.getMessage()
        return json.dumps(log, ensure_ascii=False)


def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)


@dataclass(frozen=True)
class AppConfig:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_bucket_raw: str
    minio_bucket_processed: str

    milvus_uri: str
    milvus_username: str
    milvus_password: str
    milvus_db: str
    milvus_collection_dense: str
    milvus_collection_hybrid: str

    tenant_id: str
    app_env: str

    vllm_server_url: str
    vllm_server_url_api: str
    vllm_model_name: str

    embedding_mode: str
    embedding_model_name: str
    embedding_model_version: str
    embedding_model_dim: int
    embedding_batch_size: int
    embedding_max_length: int
    embedding_device: str

    rag_schema_version: str

    path_base: str
    path_raw: str
    path_processed: str
    path_tmp: str

    max_chunks_per_doc: int
    chunk_size: int
    chunk_overlap: int


def _require_env(name: str, *, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"[config] Variável de ambiente obrigatória ausente: {name}")
    return value.strip()


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(name: str, value: str) -> int:
    try:
        v = int(value)
    except ValueError:
        raise RuntimeError(f"[config] Variável {name} precisa ser um inteiro. Valor atual: {value}")
    if v <= 0:
        raise RuntimeError(f"[config] Variável {name} precisa ser > 0. Valor atual: {v}")
    return v


def get_logger(name: str) -> logging.Logger:
    _setup_logging()
    return logging.getLogger(name)


@lru_cache()
def get_config() -> AppConfig:
    _setup_logging()
    logger = get_logger(__name__)

    tenant_id = _require_env("TENANT_ID", default="tenant-dev")
    app_env = _require_env("APP_ENV", default="dev")
    if app_env not in {"dev", "stage", "prod"}:
        raise RuntimeError(f"[config] APP_ENV deve ser dev|stage|prod. Valor atual: {app_env}")

    minio_endpoint = _require_env("MINIO_ENDPOINT", default="minio-api-ai-s3-minio-dev.apps.lambari.labredhat.seprol")
    minio_access_key = _require_env("MINIO_ACCESS_KEY", default="models")
    minio_secret_key = _require_env("MINIO_SECRET_KEY", default="seprol@2025")
    minio_secure = _parse_bool(_require_env("MINIO_SECURE", default="false"))
    minio_bucket_raw = _require_env("MINIO_BUCKET_RAW", default="rag-dev-raw")
    minio_bucket_processed = _require_env("MINIO_BUCKET_PROCESSED", default="rag-dev-processed")

    milvus_uri = _require_env("MILVUS_URI", default="http://milvus-dev.ai-vectordb-milvus-dev.svc.cluster.local:19530")
    milvus_username = _require_env("MILVUS_USERNAME", default="default")
    milvus_password = _require_env("MILVUS_PASSWORD", default="default")
    milvus_db = _require_env("MILVUS_DB", default="default")

    # Collections: respeitar padrão rag_{TENANT_ID}_*
    coll_dense_env = os.getenv("MILVUS_COLLECTION_DENSE", "fin_dense").strip()
    coll_hybrid_env = os.getenv("MILVUS_COLLECTION_HYBRID", "fin_hybrid").strip()

    if "{TENANT_ID}" in coll_dense_env:
        coll_dense = coll_dense_env.replace("{TENANT_ID}", tenant_id)
    elif coll_dense_env:
        coll_dense = coll_dense_env
    else:
        coll_dense = f"rag_{tenant_id}_dense"

    if "{TENANT_ID}" in coll_hybrid_env:
        coll_hybrid = coll_hybrid_env.replace("{TENANT_ID}", tenant_id)
    elif coll_hybrid_env:
        coll_hybrid = coll_hybrid_env
    else:
        coll_hybrid = f"rag_{tenant_id}_hybrid"

    vllm_server_url = _require_env("VLLM_SERVER_URL", default="http://inference-server-ocp-ai-inference-server-ocp-dev.apps.lambari.labredhat.seprol")
    vllm_server_url_api = os.getenv("VLLM_SERVER_URL_API", "/v1/chat/completions").strip() or "/v1/chat/completions"
    vllm_model_name = _require_env("VLLM_MODEL_NAME", default="RedHatAI/Qwen3-8B-quantized.w4a16")

    embedding_mode = _require_env("EMBEDDING_MODE", default="all")
    if embedding_mode not in {"dense", "hybrid", "all"}:
        raise RuntimeError("EMBEDDING_MODE deve ser one of: dense|hybrid|all")

    embedding_model_name = _require_env("EMBEDDING_MODEL_NAME", default="BAAI/bge-m3")
    embedding_model_version = _require_env("EMBEDDING_MODEL_VERSION", default="1")
    embedding_model_dim = _parse_int("EMBEDDING_MODEL_DIM", _require_env("EMBEDDING_MODEL_DIM", default="1024"))
    embedding_batch_size = _parse_int("EMBEDDING_BATCH_SIZE", _require_env("EMBEDDING_BATCH_SIZE", default="32"))
    embedding_max_length = _parse_int("EMBEDDING_MAX_LENGTH", _require_env("EMBEDDING_MAX_LENGTH", default="1024"))
    embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
    if embedding_device not in {"cpu", "cuda", "cuda:0"}:
        raise RuntimeError("EMBEDDING_DEVICE deve ser one of: cpu|cuda|cuda:0")

    rag_schema_version = _require_env("RAG_SCHEMA_VERSION", default="1")

    path_base = os.getenv("PATH_BASE", "/opt/app-root/src").strip() or "/opt/app-root/src/tmp"
    path_raw = os.getenv("PATH_RAW", os.path.join(path_base, "raw"))
    path_processed = os.getenv("PATH_PROCESSED", os.path.join(path_base, "processed"))
    path_tmp = os.getenv("PATH_TMP", os.path.join(path_base, "tmp"))

    max_chunks_per_doc = _parse_int("MAX_CHUNKS_PER_DOC", _require_env("MAX_CHUNKS_PER_DOC", default="256"))
    chunk_size = _parse_int("CHUNK_SIZE", _require_env("CHUNK_SIZE", default="512"))
    chunk_overlap = _parse_int("CHUNK_OVERLAP", _require_env("CHUNK_OVERLAP", default="64"))
    if chunk_overlap >= chunk_size:
        raise RuntimeError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")

    cfg = AppConfig(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_secure=minio_secure,
        minio_bucket_raw=minio_bucket_raw,
        minio_bucket_processed=minio_bucket_processed,
        milvus_uri=milvus_uri,
        milvus_username=milvus_username,
        milvus_password=milvus_password,
        milvus_db=milvus_db,
        milvus_collection_dense=coll_dense,
        milvus_collection_hybrid=coll_hybrid,
        tenant_id=tenant_id,
        app_env=app_env,
        vllm_server_url=vllm_server_url,
        vllm_server_url_api=vllm_server_url_api,
        vllm_model_name=vllm_model_name,
        embedding_mode=embedding_mode,
        embedding_model_name=embedding_model_name,
        embedding_model_version=embedding_model_version,
        embedding_model_dim=embedding_model_dim,
        embedding_batch_size=embedding_batch_size,
        embedding_max_length=embedding_max_length,
        embedding_device=embedding_device,
        rag_schema_version=rag_schema_version,
        path_base=path_base,
        path_raw=path_raw,
        path_processed=path_processed,
        path_tmp=path_tmp,
        max_chunks_per_doc=max_chunks_per_doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(
        "Config carregada",
        extra={
            "event": "config_loaded",
            "tenant_id": tenant_id,
            "correlation_id": None,
            "doc_id": None,
        },
    )
    return cfg
