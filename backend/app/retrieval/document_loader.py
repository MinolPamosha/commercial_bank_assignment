"""Load markdown knowledge docs with YAML front-matter metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {"doc_id": path.stem, "title": path.stem}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    return meta, body


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= max_chars:
                buf = para
            else:
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i : i + max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks or [text[:max_chars]]


def load_documents(path: str | None = None) -> list[DocumentChunk]:
    settings = get_settings()
    root = Path(path or settings.documents_path)
    chunks: list[DocumentChunk] = []
    for file in sorted(root.glob("*.md")):
        meta, body = _parse_markdown(file)
        doc_id = str(meta.get("doc_id", file.stem))
        title = str(meta.get("title", file.stem))
        parts = _chunk_text(body)
        for idx, part in enumerate(parts):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}::c{idx}",
                    doc_id=doc_id,
                    title=title,
                    text=part,
                    metadata={
                        "department": meta.get("department", "general"),
                        "document_type": meta.get("document_type", "unknown"),
                        "access_level": meta.get("access_level", "internal"),
                        "created_date": str(meta.get("created_date", "")),
                        "tags": meta.get("tags", []),
                        "title": title,
                        "doc_id": doc_id,
                        "source_file": file.name,
                    },
                )
            )
    logger.info("documents_loaded", files=len(list(root.glob('*.md'))), chunks=len(chunks))
    return chunks
