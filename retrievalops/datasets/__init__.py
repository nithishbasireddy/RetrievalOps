"""Dataset registry."""

from __future__ import annotations

from retrievalops.config import DatasetConfig
from retrievalops.types import Dataset


def load_dataset(cfg: DatasetConfig) -> Dataset:
    if cfg.name == "local_legal":
        from retrievalops.datasets.local import load_local_legal

        return load_local_legal(cfg)
    if cfg.name == "legalbench_rag":
        from retrievalops.datasets.legalbench_rag import load_legalbench_rag

        return load_legalbench_rag(cfg)
    raise ValueError(f"Unknown dataset: {cfg.name!r}")
