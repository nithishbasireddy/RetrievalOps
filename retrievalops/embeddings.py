"""Sentence-embedding encoders with model-correct query/passage prompts and a disk cache.

Many retrieval embedders are trained with asymmetric prompts; omitting them silently degrades
quality (e.g. E5 without "query: "/"passage: "), so prompts are resolved per model here.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

log = logging.getLogger(__name__)

_BGE_EN_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Prompts:
    query: str = ""
    passage: str = ""
    use_model_prompts: bool = False  # use the prompts shipped in the model's ST config


def resolve_prompts(model_name: str) -> Prompts:
    m = model_name.lower()
    if "/e5-" in m or "multilingual-e5" in m:
        return Prompts("query: ", "passage: ")
    if "bge-" in m and "-en" in m and "bge-m3" not in m:
        return Prompts(_BGE_EN_QUERY, "")
    if "qwen3-embedding" in m or "gte-qwen" in m or "e5-mistral" in m:
        return Prompts(use_model_prompts=True)
    return Prompts()  # symmetric models: MiniLM, mpnet, bge-m3, ...


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Encoder:
    _instances: ClassVar[dict[tuple[str, str], Encoder]] = {}

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or default_device()
        self.batch_size = batch_size
        kwargs = {"torch_dtype": torch.bfloat16} if self.device.startswith("cuda") else {}
        log.info("Loading embedding model %s on %s", model_name, self.device)
        from retrievalops.models import resolve_model_path

        self.model = SentenceTransformer(resolve_model_path(model_name), device=self.device, model_kwargs=kwargs)
        self.prompts = resolve_prompts(model_name)
        if self.prompts.use_model_prompts and "query" not in (self.model.prompts or {}):
            raise ValueError(f"{model_name} has no 'query' prompt in its config; add it to resolve_prompts")

    @classmethod
    def get(cls, model_name: str, device: str | None = None) -> Encoder:
        key = (model_name, device or default_device())
        if key not in cls._instances:
            cls._instances[key] = cls(model_name, device)
        return cls._instances[key]

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        kwargs: dict = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": len(texts) > 5000,
        }
        if self.prompts.use_model_prompts:
            if kind == "query":
                kwargs["prompt_name"] = "query"
            elif "document" in (self.model.prompts or {}):
                kwargs["prompt_name"] = "document"
        else:
            prefix = self.prompts.query if kind == "query" else self.prompts.passage
            if prefix:
                texts = [prefix + t for t in texts]
        return self.model.encode(texts, **kwargs).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_passages(self, texts: list[str], cache_dir: str | Path | None = None) -> np.ndarray:
        if cache_dir is None:
            return self._encode(texts, "passage")
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(repr(self.prompts).encode())
        for t in texts:
            h.update(t.encode("utf-8", errors="replace"))
            h.update(b"\x00")
        path = Path(cache_dir) / "embeddings" / self.model_name.replace("/", "__") / f"{h.hexdigest()[:24]}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return arr
        arr = self._encode(texts, "passage")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(path)
        return arr
