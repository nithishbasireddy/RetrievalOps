"""Offset-preserving chunkers.

Every chunk is an exact slice ``doc.text[start:end]`` so retrieved chunks can be
scored against character-level gold spans.
"""

from __future__ import annotations

import re

from retrievalops.config import ChunkingConfig
from retrievalops.types import Chunk, Document

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _split_pieces(text: str, start: int, end: int, size: int, seps: tuple[str, ...]) -> list[tuple[int, int]]:
    """Cut text[start:end] into contiguous pieces of length <= size.

    Pieces cover the range exactly; each separator stays attached to the piece before it.
    """
    if end - start <= size:
        return [(start, end)]
    for i, sep in enumerate(seps):
        if sep == "":
            return [(s, min(s + size, end)) for s in range(start, end, size)]
        if text.find(sep, start, end) == -1:
            continue
        pieces: list[tuple[int, int]] = []
        cur = start
        while cur < end:
            nxt = text.find(sep, cur, end)
            piece_end = end if nxt == -1 else nxt + len(sep)
            if piece_end - cur <= size:
                pieces.append((cur, piece_end))
            else:
                pieces.extend(_split_pieces(text, cur, piece_end, size, seps[i + 1 :]))
            cur = piece_end
        return pieces
    return [(start, end)]  # unreachable: "" is always the last separator


def _merge_pieces(pieces: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i, n = 0, len(pieces)
    while i < n:
        j = i
        while j < n and pieces[j][1] - pieces[i][0] <= size:
            j += 1
        j = max(j, i + 1)
        spans.append((pieces[i][0], pieces[j - 1][1]))
        if j >= n:
            break
        # back up so the next chunk re-uses up to `overlap` chars of trailing pieces
        k = j
        while k - 1 > i and pieces[j - 1][1] - pieces[k - 1][0] <= overlap:
            k -= 1
        i = k
    return spans


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_spans(text: str, cfg: ChunkingConfig) -> list[tuple[int, int]]:
    if not text:
        return []
    size, overlap = cfg.chunk_size, cfg.chunk_overlap
    if cfg.strategy == "fixed":
        step = size - overlap
        raw = []
        for s in range(0, len(text), step):
            raw.append((s, min(s + size, len(text))))
            if s + size >= len(text):
                break
    else:
        raw = _merge_pieces(_split_pieces(text, 0, len(text), size, DEFAULT_SEPARATORS), size, overlap)
    out = []
    for s, e in raw:
        s, e = _strip(text, s, e)
        if e > s:
            out.append((s, e))
    return out


def _safe_id(doc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id)


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    base = _safe_id(doc.doc_id)
    return [
        Chunk(chunk_id=f"{base}#{i:05d}", doc_id=doc.doc_id, start=s, end=e, text=doc.text[s:e])
        for i, (s, e) in enumerate(chunk_spans(doc.text, cfg))
    ]


def chunk_corpus(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, cfg))
    return chunks
