"""Run every retrieval configuration of an experiment file over one dataset.

Outputs, under ``<output_dir>/<run_name>/``:
    manifest.json             experiment config, dataset stats, git commit, library versions
    per_query/<config>.jsonl  one line per query: ranked chunks (+ answer) + metrics
    summary.csv               mean metrics per config x subset (+ micro/macro "ALL" rows, 95% CIs)

Completed configs are skipped on re-run, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from retrievalops.config import ExperimentFile, RetrievalConfig
from retrievalops.datasets import load_dataset
from retrievalops.embeddings import default_device
from retrievalops.ingestion.chunking import chunk_corpus
from retrievalops.metrics import all_k_metrics, count_relevant
from retrievalops.retrieval.indexes import BM25Index
from retrievalops.retrieval.pipeline import RetrievalPipeline
from retrievalops.types import Chunk, Dataset, Span

log = logging.getLogger(__name__)

QUERY_BATCH = 256
N_BOOTSTRAP = 1000


def _git_commit() -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("torch", "sentence-transformers", "transformers", "bm25s"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _count_lines(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for _ in fh)


class Runner:
    def __init__(self, exp: ExperimentFile, run_name: str | None = None, max_queries: int | None = None):
        self.exp = exp
        self.device = exp.device or default_device()
        self.dataset: Dataset = load_dataset(exp.dataset)
        if max_queries is not None:
            self.dataset.queries = self.dataset.queries[:max_queries]
        self.run_dir = Path(exp.output_dir) / (run_name or f"{self.dataset.name}_{datetime.now():%Y%m%d_%H%M%S}")
        (self.run_dir / "per_query").mkdir(parents=True, exist_ok=True)
        self._chunks: dict[str, list[Chunk]] = {}
        self._bm25: dict[str, BM25Index] = {}
        self.generator = None
        if exp.generation is not None:
            from retrievalops.generation import Generator

            self.generator = Generator(exp.generation)

    # ---- caches shared across configs -------------------------------------------------
    def chunks_for(self, cfg: RetrievalConfig) -> list[Chunk]:
        key = cfg.chunking.key()
        if key not in self._chunks:
            self._chunks[key] = chunk_corpus(self.dataset.documents, cfg.chunking)
            log.info("Chunking %s -> %d chunks", key, len(self._chunks[key]))
        return self._chunks[key]

    def bm25_for(self, cfg: RetrievalConfig, chunks: list[Chunk]) -> BM25Index | None:
        if cfg.retriever not in ("bm25", "hybrid"):
            return None
        key = cfg.chunking.key()
        if key not in self._bm25:
            self._bm25[key] = BM25Index([c.text for c in chunks])
        return self._bm25[key]

    # ---- main loop --------------------------------------------------------------------
    def write_manifest(self) -> None:
        ds = self.dataset
        manifest = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "versions": _versions(),
            "device": self.device,
            "dataset": {
                "name": ds.name,
                "n_documents": len(ds.documents),
                "n_chars": sum(len(d.text) for d in ds.documents),
                "n_queries": len(ds.queries),
                "has_gold_spans": ds.has_gold_spans,
            },
            "experiment": self.exp.model_dump(),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def run(self, only: list[str] | None = None) -> pd.DataFrame:
        self.write_manifest()
        configs = [c for c in self.exp.configs if only is None or c.name in only]
        if only:
            missing = set(only) - {c.name for c in configs}
            if missing:
                raise ValueError(f"Unknown config names: {sorted(missing)}")
        for cfg in configs:
            out_path = self.run_dir / "per_query" / f"{cfg.name}.jsonl"
            if out_path.exists() and _count_lines(out_path) == len(self.dataset.queries):
                log.info("[%s] already complete - skipping", cfg.name)
                continue
            self.run_config(cfg, out_path)
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
        return self.summarize()

    def run_config(self, cfg: RetrievalConfig, out_path: Path) -> None:
        ks = self.exp.eval_ks
        chunks = self.chunks_for(cfg)
        t0 = time.perf_counter()
        pipe = RetrievalPipeline(
            cfg, chunks, device=self.device, cache_dir=self.exp.cache_dir, bm25_index=self.bm25_for(cfg, chunks)
        )
        build_s = time.perf_counter() - t0

        chunks_by_doc: dict[str, list[Span]] = defaultdict(list)
        for c in chunks:
            chunks_by_doc[c.doc_id].append(Span(c.doc_id, c.start, c.end))

        queries = self.dataset.queries
        tmp_path = out_path.with_suffix(".jsonl.tmp")
        search_s = 0.0
        with open(tmp_path, "w") as f:
            for b in tqdm(range(0, len(queries), QUERY_BATCH), desc=cfg.name, leave=False):
                batch = queries[b : b + QUERY_BATCH]
                t1 = time.perf_counter()
                ranked = pipe.search([q.text for q in batch])
                search_s += time.perf_counter() - t1
                for q, hits in zip(batch, ranked):
                    retrieved = [chunks[i] for i, _ in hits]
                    rec: dict = {
                        "qid": q.qid,
                        "subset": q.subset,
                        "query": q.text,
                        "retrieved": [
                            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "start": c.start, "end": c.end, "score": s}
                            for c, (_, s) in zip(retrieved, hits)
                        ],
                    }
                    if q.gold_spans:
                        gold_docs = {s.doc_id for s in q.gold_spans}
                        n_rel = count_relevant(
                            (s for d in gold_docs for s in chunks_by_doc.get(d, [])), q.gold_spans
                        )
                        rec["n_relevant_chunks"] = n_rel
                        rec["metrics"] = all_k_metrics(
                            [Span(c.doc_id, c.start, c.end) for c in retrieved], q.gold_spans, ks, n_rel
                        )
                    if self.generator is not None:
                        rec["answer"] = self.generator.answer(q.text, [c.text for c in retrieved])
                        rec["reference_answer"] = q.reference_answer
                    f.write(json.dumps(rec) + "\n")
        tmp_path.replace(out_path)
        meta = {
            "config": cfg.model_dump(),
            "n_chunks": len(chunks),
            "mean_chunk_chars": float(np.mean([c.end - c.start for c in chunks])) if chunks else 0.0,
            "index_build_s": build_s,
            "search_ms_per_query": 1000 * search_s / max(1, len(queries)),
        }
        out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("[%s] done: %d chunks, build %.1fs, %.1f ms/query", cfg.name, len(chunks), build_s,
                 meta["search_ms_per_query"])

    def summarize(self) -> pd.DataFrame:
        rows = []
        for path in sorted((self.run_dir / "per_query").glob("*.jsonl")):
            name = path.stem
            meta_path = path.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            with open(path) as fh:
                recs = [json.loads(line) for line in fh]
            recs = [r for r in recs if "metrics" in r]
            if not recs:
                continue
            df = pd.DataFrame([{**r["metrics"], "subset": r["subset"] or "all"} for r in recs])
            metric_cols = [c for c in df.columns if c != "subset"]
            groups = {s: g for s, g in df.groupby("subset")}
            groups["ALL_micro"] = df
            for subset, g in groups.items():
                row = {"config": name, "subset": subset, "n": len(g), **g[metric_cols].mean().to_dict()}
                for col in metric_cols:
                    if col.startswith(("recall@", "ndcg@")):
                        row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = _bootstrap_ci(g[col].to_numpy())
                row.update({k: meta.get(k) for k in ("n_chunks", "index_build_s", "search_ms_per_query")})
                rows.append(row)
            if len(groups) > 2:  # macro average over subsets, as in LegalBench-RAG
                per_subset = df.groupby("subset")[metric_cols].mean()
                rows.append({"config": name, "subset": "ALL_macro", "n": len(df), **per_subset.mean().to_dict(),
                             **{k: meta.get(k) for k in ("n_chunks", "index_build_s", "search_ms_per_query")}})
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary.to_csv(self.run_dir / "summary.csv", index=False)
        return summary
