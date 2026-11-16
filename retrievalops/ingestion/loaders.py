"""Load raw files (PDF, DOCX, TXT/MD) into Documents."""

from __future__ import annotations

import logging
from pathlib import Path

from retrievalops.types import Document

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # extract_text() can return None/"" for scanned or image-only pages
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path}")


def load_directory(root: str | Path, recursive: bool = True) -> list[Document]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory not found: {root}")
    pattern = "**/*" if recursive else "*"
    docs = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        text = read_file(path)
        if not text.strip():
            log.warning("No extractable text in %s (scanned PDF?) - skipped", path)
            continue
        docs.append(Document(doc_id=path.relative_to(root).as_posix(), text=text))
    return docs
