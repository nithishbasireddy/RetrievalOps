"""LegalBench-RAG (Pipitone & Alami, 2024; arXiv:2408.10343).

Expected layout (the official download):
    <path>/corpus/<subset>/*.txt
    <path>/benchmarks/<subset>.json   {"tests": [{"query": ..., "snippets": [{"file_path", "span"}]}]}

Protocol follows the reference implementation (zeroentropy-cc/legalbenchrag, benchmark.py):
  * one shared corpus for all selected subsets, made of every file referenced by a gold snippet;
  * ``sampling: lbr_mini`` reproduces LegalBench-RAG-mini: per subset, tests are ordered by a
    random key seeded with their first snippet's file path, the first 194 are kept, and the corpus
    is restricted to files referenced by the kept tests (SORT_BY_DOCUMENT=True);
  * ``sampling: random`` draws ``max_queries_per_subset`` tests uniformly (seeded) over the full corpus.
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

from retrievalops.config import DatasetConfig
from retrievalops.types import Dataset, Document, Query, Span

ALL_SUBSETS = ["contractnli", "cuad", "maud", "privacy_qa"]
LBR_MINI_SIZE = 194


def _nfc(path: str) -> str:
    # benchmarks/*.json store some file paths in NFD form while the corpus files are NFC
    return unicodedata.normalize("NFC", path)


def _lbr_mini_order(tests: list[dict]) -> list[int]:
    def key(i: int) -> float:
        random.seed(tests[i]["snippets"][0]["file_path"])
        return random.random()

    order = sorted(range(len(tests)), key=key)
    random.seed()  # do not leave the global RNG in a fixed state
    return order


def load_legalbench_rag(cfg: DatasetConfig) -> Dataset:
    if not cfg.path:
        raise ValueError("legalbench_rag needs dataset.path (folder with corpus/ and benchmarks/)")
    root = Path(cfg.path).expanduser()
    subsets = cfg.subsets or ALL_SUBSETS
    unknown = set(subsets) - set(ALL_SUBSETS)
    if unknown:
        raise ValueError(f"Unknown LegalBench-RAG subsets: {sorted(unknown)}")

    queries: list[Query] = []
    referenced_all: set[str] = set()
    referenced_used: set[str] = set()
    for subset in subsets:
        with open(root / "benchmarks" / f"{subset}.json") as f:
            tests = json.load(f)["tests"]
        referenced_all |= {_nfc(s["file_path"]) for t in tests for s in t["snippets"]}

        idx = list(range(len(tests)))
        if cfg.sampling == "lbr_mini":
            n = cfg.max_queries_per_subset or LBR_MINI_SIZE
            if len(tests) > n:
                idx = _lbr_mini_order(tests)[:n]
        elif cfg.sampling == "random" and cfg.max_queries_per_subset is not None and (
            cfg.max_queries_per_subset < len(idx)
        ):
            idx = sorted(random.Random(cfg.seed).sample(idx, cfg.max_queries_per_subset))

        for i in idx:
            t = tests[i]
            spans = [Span(_nfc(s["file_path"]), int(s["span"][0]), int(s["span"][1])) for s in t["snippets"]]
            referenced_used.update(s.doc_id for s in spans)
            queries.append(Query(qid=f"{subset}/{i}", text=t["query"], gold_spans=spans, subset=subset))

    corpus_files = referenced_used if cfg.sampling == "lbr_mini" else referenced_all
    if cfg.query_file:
        # replace the queries (e.g. synthetic ones) but keep the official corpus unchanged
        if cfg.sampling == "lbr_mini":
            raise ValueError("query_file cannot be combined with sampling: lbr_mini")
        with open(Path(cfg.query_file).expanduser()) as f:
            tests = json.load(f)["tests"]
        queries = []
        for t in tests:
            if t["subset"] not in subsets:
                continue
            spans = [Span(_nfc(s["file_path"]), int(s["span"][0]), int(s["span"][1])) for s in t["snippets"]]
            missing = {s.doc_id for s in spans} - corpus_files
            if missing:
                raise ValueError(f"query {t['qid']} references files outside the corpus: {sorted(missing)[:2]}")
            queries.append(Query(qid=t["qid"], text=t["query"], gold_spans=spans, subset=t["subset"]))
    documents = []
    for doc_id in sorted(corpus_files):
        path = root / "corpus" / doc_id
        if not path.is_file():
            raise FileNotFoundError(f"Gold file missing from corpus: {path}")
        documents.append(Document(doc_id=doc_id, text=path.read_text(encoding="utf-8")))
    return Dataset(name="legalbench_rag", documents=documents, queries=queries)
