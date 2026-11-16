"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice ``doc.text[start:end]`` of a source document."""

    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Span:
    """A gold evidence span ``doc.text[start:end]``."""

    doc_id: str
    start: int
    end: int


@dataclass
class Query:
    qid: str
    text: str
    gold_spans: list[Span] = field(default_factory=list)
    reference_answer: str | None = None
    subset: str | None = None


@dataclass
class Dataset:
    name: str
    documents: list[Document]
    queries: list[Query]

    @property
    def has_gold_spans(self) -> bool:
        return any(q.gold_spans for q in self.queries)
