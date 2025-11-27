import os
import re
from collections import Counter
from typing import List, Optional

from docling.document_converter import DocumentConverter

from .config import get_config, get_logger


_converter: Optional[DocumentConverter] = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def extract_text_from_file(path: str, correlation_id: Optional[str] = None, doc_id: Optional[str] = None) -> str:
    """
    Extrai texto via Docling (PDF, DOCX, etc.).
    """
    cfg = get_config()
    logger = get_logger(__name__)
    conv = _get_converter()
    result = conv.convert(path)
    text = result.document.export_to_text()

    logger.info(
        f"Texto extraído de {os.path.basename(path)} (len={len(text)})",
        extra={"event": "docling_extract", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
    )
    return sanitize_text(text)


def sanitize_text(text: str) -> str:
    """
    Normaliza espaços e remove linhas de header/footer repetitivas.
    """
    # Normalizar quebras de linha
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    stripped = [ln.strip() for ln in lines]
    counts = Counter([ln for ln in stripped if ln])

    # Linhas muito frequentes e curtas são consideradas ruído (headers/footers)
    noise_candidates = {ln for ln, c in counts.items() if c >= 3 and len(ln) <= 80}

    filtered_lines = [ln for ln in stripped if ln and ln not in noise_candidates]
    dedup_lines: List[str] = []
    last = None
    for ln in filtered_lines:
        if ln != last:
            dedup_lines.append(ln)
        last = ln

    return "\n".join(dedup_lines).strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> List[str]:
    """
    Chunk por parágrafos, com controle de overlap e limite de chunks.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        to_add = para
        # se parágrafo maior que chunk_size, quebrar internamente
        while len(to_add) > chunk_size:
            if len(chunks) >= max_chunks:
                return chunks
            chunk = to_add[:chunk_size]
            chunks.append(chunk)
            to_add = to_add[chunk_size - chunk_overlap :]

        if not current:
            current = to_add
        elif len(current) + len(to_add) + 2 <= chunk_size:
            current = current + "\n\n" + to_add
        else:
            if len(chunks) >= max_chunks:
                break
            chunks.append(current)
            current = to_add

        if len(chunks) >= max_chunks:
            break

    if current and len(chunks) < max_chunks:
        chunks.append(current)

    return chunks


def chunk_document_text(
    text: str,
    correlation_id: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> List[dict]:
    """
    Retorna lista de dicts: {"chunk_id": int, "text": str}
    """
    cfg = get_config()
    logger = get_logger(__name__)
    pieces = chunk_text(
        text,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        max_chunks=cfg.max_chunks_per_doc,
    )
    logger.info(
        f"Chunking gerou {len(pieces)} chunks",
        extra={"event": "chunking_done", "tenant_id": cfg.tenant_id, "correlation_id": correlation_id, "doc_id": doc_id},
    )
    return [{"chunk_id": idx, "text": t} for idx, t in enumerate(pieces)]
