import os
import time
from typing import List, Optional

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource

from .config import get_config, get_logger

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 10.0

def _retry(operation: str, func, logger, correlation_id: Optional[str], *args, **kwargs):
    # REMOVER doc_id dos kwargs antes de chamar a função
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
        except S3Error as exc:
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


def get_minio_client() -> Minio:
    cfg = get_config()
    logger = get_logger(__name__)
    logger.info(
        "Inicializando cliente MinIO",
        extra={"event": "minio_init", "tenant_id": cfg.tenant_id, "correlation_id": None, "doc_id": None},
    )
    return Minio(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=cfg.minio_secure,
    )


def create_buckets(correlation_id: Optional[str] = None) -> None:
    cfg = get_config()
    cli = get_minio_client()
    logger = get_logger(__name__)

    for bucket in {cfg.minio_bucket_raw, cfg.minio_bucket_processed}:
        def _exists():
            return cli.bucket_exists(bucket_name=bucket)

        exists = _retry("minio_bucket_exists", _exists, logger, correlation_id)
        if not exists:
            def _make():
                cli.make_bucket(bucket_name=bucket)

            _retry("minio_make_bucket", _make, logger, correlation_id)
            logger.info(
                f"Bucket criado: {bucket}",
                extra={"event": "minio_bucket_created", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id},
            )


def exists(bucket: str, key: str, correlation_id: Optional[str] = None) -> bool:
    cfg = get_config()
    logger = get_logger(__name__)

    def _stat():
        return cli.stat_object(bucket_name=bucket, object_name=key)

    cli = get_minio_client()
    try:
        _retry("minio_stat_object", _stat, logger, correlation_id)
        return True
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket"):
            return False
        raise


def stat_object(bucket: str, key: str, correlation_id: Optional[str] = None):
    logger = get_logger(__name__)
    cli = get_minio_client()

    def _stat():
        return cli.stat_object(bucket_name=bucket, object_name=key)

    return _retry("minio_stat_object", _stat, logger, correlation_id)


def list_new_files(correlation_id: Optional[str] = None) -> List[dict]:
    """
    Lista objetos no bucket RAW sob raw/{TENANT_ID}/ que ainda não possuem
    contraparte no bucket PROCESSED em processed/{TENANT_ID}/.
    """
    cfg = get_config()
    cli = get_minio_client()
    logger = get_logger(__name__)

    prefix_raw = f"raw/{cfg.tenant_id}/"
    objects = cli.list_objects(
        bucket_name=cfg.minio_bucket_raw,
        prefix=prefix_raw,
        recursive=True,
    )

    new_files = []
    for obj in objects:
        obj_key = obj.object_name
        processed_key = obj_key.replace(f"raw/{cfg.tenant_id}/", f"processed/{cfg.tenant_id}/", 1)
        if not exists(cfg.minio_bucket_processed, processed_key, correlation_id):
            new_files.append(
                {
                    "bucket": cfg.minio_bucket_raw,
                    "key": obj_key,
                    "size": obj.size,
                    "etag": obj.etag,
                    "last_modified": obj.last_modified,
                }
            )

    logger.info(
        f"list_new_files encontrou {len(new_files)} objetos novos",
        extra={
            "event": "minio_list_new_files",
            "tenant_id": cfg.tenant_id,
            "correlation_id": correlation_id,
            "doc_id": None,
        },
    )
    return new_files


def download_file(bucket: str, key: str, local_path: str, correlation_id: Optional[str] = None, doc_id: Optional[str] = None) -> None:
    cfg = get_config()
    cli = get_minio_client()
    logger = get_logger(__name__)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    def _download():
        cli.fget_object(bucket_name=bucket, object_name=key, file_path=local_path)

    _retry("minio_download_file", _download, logger, correlation_id, doc_id=doc_id)
    logger.info(
        f"Download concluído: s3://{bucket}/{key} -> {local_path}",
        extra={"event": "minio_download_done", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
    )


def upload_file(bucket: str, key: str, local_path: str, correlation_id: Optional[str] = None, doc_id: Optional[str] = None) -> None:
    cfg = get_config()
    cli = get_minio_client()
    logger = get_logger(__name__)

    def _upload():
        cli.fput_object(bucket_name=bucket, object_name=key, file_path=local_path)

    _retry("minio_upload_file", _upload, logger, correlation_id, doc_id=doc_id)
    logger.info(
        f"Upload concluído: {local_path} -> s3://{bucket}/{key}",
        extra={"event": "minio_upload_done", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
    )


def move_file(src_bucket: str, src_key: str, dest_bucket: str, dest_key: str, correlation_id: Optional[str] = None, doc_id: Optional[str] = None) -> None:
    """
    Move via copy + delete, idempotente.
    """
    cfg = get_config()
    cli = get_minio_client()
    logger = get_logger(__name__)

    if exists(dest_bucket, dest_key, correlation_id):
        # já movido
        logger.info(
            f"move_file: destino já existe, operação idempotente s3://{dest_bucket}/{dest_key}",
            extra={"event": "minio_move_skip", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
        )
        return

    def _copy():
        source = CopySource(src_bucket, src_key)
        cli.copy_object(
            bucket_name=dest_bucket,
            object_name=dest_key,
            source=source
        )

    def _remove():
        cli.remove_object(
            bucket_name=src_bucket,
            object_name=src_key
        )

    _retry("minio_copy_object", _copy, logger, correlation_id, doc_id=doc_id)
    _retry("minio_remove_object", _remove, logger, correlation_id, doc_id=doc_id)

    logger.info(
        f"Move concluído: s3://{src_bucket}/{src_key} -> s3://{dest_bucket}/{dest_key}",
        extra={"event": "minio_move_done", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
    )
