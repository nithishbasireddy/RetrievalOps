"""A retrieval pipeline = chunked corpus + first-stage retriever (+ optional reranker).

Indexes are built once per pipeline and reused across all queries.
"""

from __future__ import annotations

from retrievalops.config import RetrievalConfig
from retrievalops.embeddings import Encoder, default_device
from retrievalops.retrieval.indexes import BM25Index, DenseIndex, Ranked, Reranker, rrf_fuse
from retrievalops.types import Chunk


class RetrievalPipeline:
    def __init__(
        self,
        cfg: RetrievalConfig,
        chunks: list[Chunk],
        device: str | None = None,
        cache_dir: str | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self.cfg = cfg
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self.device = device or default_device()
        self.encoder: Encoder | None = None
        self.dense: DenseIndex | None = None
        self.bm25: BM25Index | None = None

        if cfg.retriever in ("dense", "hybrid"):
            assert cfg.embedding_model  # guaranteed by config validation
            self.encoder = Encoder.get(cfg.embedding_model, self.device)
            emb = self.encoder.encode_passages(self.texts, cache_dir=cache_dir)
            self.dense = DenseIndex(emb, self.device)
        if cfg.retriever in ("bm25", "hybrid"):
            self.bm25 = bm25_index if bm25_index is not None else BM25Index(self.texts)
        self.reranker = Reranker.get(cfg.reranker, self.device) if cfg.reranker else None

    def search(self, queries: list[str]) -> list[Ranked]:
        cfg = self.cfg
        first_k = cfg.candidate_k if (self.reranker or cfg.retriever == "hybrid") else cfg.top_k

        dense_res = bm25_res = None
        if self.dense is not None and self.encoder is not None:
            dense_res = self.dense.search(self.encoder.encode_queries(queries), first_k)
        if self.bm25 is not None:
            bm25_res = self.bm25.search(queries, first_k)

        results: list[Ranked] = []
        for qi, query in enumerate(queries):
            if cfg.retriever == "dense":
                ranked = dense_res[qi]
            elif cfg.retriever == "bm25":
                ranked = bm25_res[qi]
            else:
                ranked = rrf_fuse([dense_res[qi], bm25_res[qi]], k=cfg.rrf_k, limit=cfg.candidate_k)
            if self.reranker is not None:
                ranked = self.reranker.rerank(query, ranked, self.texts, cfg.top_k)
            results.append(ranked[: cfg.top_k])
        return results
