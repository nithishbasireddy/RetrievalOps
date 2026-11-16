"""First-stage indexes. Each ``search`` returns, per query, a ranked list of (chunk_index, score)."""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

log = logging.getLogger(__name__)

Ranked = list[tuple[int, float]]


class DenseIndex:
    """Exact inner-product search over L2-normalised embeddings (= cosine similarity).

    Exact search keeps results deterministic and free of ANN approximation error, which matters
    when comparing configurations. 1M x 1024 fp16 vectors is ~2 GB, well within one GPU.
    """

    def __init__(self, embeddings: np.ndarray, device: str):
        import torch

        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.matrix = torch.from_numpy(embeddings).to(device=device, dtype=dtype)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def search(self, query_embeddings: np.ndarray, k: int, batch_size: int = 256) -> list[Ranked]:
        import torch

        k = min(k, len(self))
        if k == 0:
            return [[] for _ in range(len(query_embeddings))]
        out: list[Ranked] = []
        with torch.no_grad():
            for i in range(0, len(query_embeddings), batch_size):
                q = torch.from_numpy(query_embeddings[i : i + batch_size]).to(self.device, self.matrix.dtype)
                scores = (q @ self.matrix.T).float()
                top = torch.topk(scores, k, dim=1)
                for idx_row, sc_row in zip(top.indices.cpu().tolist(), top.values.cpu().tolist()):
                    out.append(list(zip(idx_row, sc_row)))
        return out


class BM25Index:
    """Okapi BM25 (bm25s) with English stop-words and Snowball stemming."""

    def __init__(self, texts: list[str]):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.n = len(texts)
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=False)

    def __len__(self) -> int:
        return self.n

    def search(self, queries: list[str], k: int) -> list[Ranked]:
        k = min(k, self.n)
        if k == 0 or not queries:
            return [[] for _ in queries]
        tokens = self._bm25s.tokenize(
            queries, stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False
        )
        out: list[Ranked] = []
        for toks in tokens:
            if not toks:  # query made only of stop-words / punctuation
                out.append([])
                continue
            docs, scores = self.retriever.retrieve([toks], k=k, show_progress=False)
            # drop zero-score tail: those chunks share no term with the query
            out.append([(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0])
        return out


def rrf_fuse(rankings: list[Ranked], k: int = 60, limit: int | None = None) -> Ranked:
    """Reciprocal Rank Fusion (Cormack et al., 2009)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused[:limit] if limit is not None else fused


class Reranker:
    _instances: ClassVar[dict[tuple[str, str], Reranker]] = {}

    def __init__(self, model_name: str, device: str, max_length: int = 512, batch_size: int = 64):
        import torch
        from sentence_transformers import CrossEncoder

        kwargs = {"torch_dtype": torch.bfloat16} if device.startswith("cuda") else {}
        log.info("Loading reranker %s on %s", model_name, device)
        from retrievalops.models import resolve_model_path

        self.model = CrossEncoder(
            resolve_model_path(model_name), device=device, max_length=max_length, model_kwargs=kwargs
        )
        self.batch_size = batch_size

    @classmethod
    def get(cls, model_name: str, device: str) -> Reranker:
        key = (model_name, device)
        if key not in cls._instances:
            cls._instances[key] = cls(model_name, device)
        return cls._instances[key]

    def rerank(self, query: str, candidates: Ranked, texts: list[str], k: int) -> Ranked:
        if not candidates:
            return []
        pairs = [(query, texts[i]) for i, _ in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            ((i, float(s)) for (i, _), s in zip(candidates, np.asarray(scores).reshape(-1))),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:k]
