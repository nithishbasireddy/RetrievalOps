import itertools
import random

import pytest

from retrievalops.config import ChunkingConfig
from retrievalops.ingestion.chunking import chunk_document, chunk_spans
from retrievalops.types import Document


def _random_text(seed: int, n: int = 20000) -> str:
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "Section", "1.2", "contract", "the", "party", "x" * 40]
    seps = [" ", " ", " ", ". ", "\n", "\n\n", "  "]
    return "".join(rng.choice(words) + rng.choice(seps) for _ in range(n // 6))[:n]


@pytest.mark.parametrize("strategy", ["recursive", "fixed"])
@pytest.mark.parametrize("size,overlap", [(1000, 200), (500, 0), (256, 64), (100, 99)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chunks_are_exact_slices_within_size(strategy, size, overlap, seed):
    text = _random_text(seed)
    cfg = ChunkingConfig(strategy=strategy, chunk_size=size, chunk_overlap=overlap)
    doc = Document("d", text)
    chunks = chunk_document(doc, cfg)
    assert chunks
    for c in chunks:
        assert c.text == text[c.start : c.end]
        assert 0 < c.end - c.start <= size
        assert c.text == c.text.strip()
    # chunks are ordered and every non-whitespace char is covered
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    covered = bytearray(len(text))
    for c in chunks:
        covered[c.start : c.end] = b"\x01" * (c.end - c.start)
    assert all(covered[i] or text[i].isspace() for i in range(len(text)))


def test_overlap_is_respected_and_progress_is_made():
    text = " ".join(f"w{i}" for i in range(3000))
    spans = chunk_spans(text, ChunkingConfig(chunk_size=200, chunk_overlap=50))
    for (s1, e1), (s2, e2) in itertools.pairwise(spans):
        assert s2 > s1
        assert max(0, e1 - s2) <= 50


def test_recursive_prefers_paragraph_boundaries():
    para = "A" * 300
    text = "\n\n".join([para] * 4)
    spans = chunk_spans(text, ChunkingConfig(chunk_size=700, chunk_overlap=0))
    for s, e in spans:
        assert text[s:e].count("A" * 300) >= 1
        assert not text[s:e].startswith("\n")


def test_empty_and_whitespace_documents():
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    assert chunk_spans("", cfg) == []
    assert chunk_spans("   \n\n  ", cfg) == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
