"""The original hand-built legal set: raw contracts + 10 questions with reference answers.

It has no gold evidence spans, so only answer-level evaluation applies to it.
"""

from __future__ import annotations

import json
from pathlib import Path

from retrievalops.config import DatasetConfig
from retrievalops.ingestion.loaders import load_directory
from retrievalops.types import Dataset, Query


def load_local_legal(cfg: DatasetConfig) -> Dataset:
    root = Path(cfg.path or "data")
    docs = load_directory(root / "raw_documents")
    with open(root / "benchmark_dataset" / "legal_benchmark_gold.json") as f:
        items = json.load(f)
    queries = [
        Query(qid=str(it["id"]), text=it["question"], reference_answer=it.get("reference_answer"))
        for it in items
    ]
    return Dataset(name="local_legal", documents=docs, queries=queries)
