"""Experiment configuration: validated pydantic models loaded from YAML.

A YAML file describes one dataset plus a set of retrieval configurations. Configs
come from an explicit ``configs:`` mapping (each entry overrides ``base:``) and/or
a ``grid:`` mapping of dotted keys to value lists (cartesian product over ``base:``).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_Strict):
    strategy: Literal["recursive", "fixed"] = "recursive"
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(200, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def key(self) -> str:
        return f"{self.strategy}-{self.chunk_size}-{self.chunk_overlap}"


class RetrievalConfig(_Strict):
    name: str
    chunking: ChunkingConfig = ChunkingConfig()
    retriever: Literal["dense", "bm25", "hybrid"] = "dense"
    embedding_model: str | None = None
    reranker: str | None = None
    top_k: int = Field(10, gt=0)
    candidate_k: int = Field(50, gt=0)  # first-stage pool for fusion / reranking
    rrf_k: int = Field(60, gt=0)

    @model_validator(mode="after")
    def _check(self) -> RetrievalConfig:
        if self.retriever in ("dense", "hybrid") and not self.embedding_model:
            raise ValueError(f"[{self.name}] retriever={self.retriever} needs embedding_model")
        if self.candidate_k < self.top_k:
            raise ValueError(f"[{self.name}] candidate_k must be >= top_k")
        return self


class DatasetConfig(_Strict):
    name: Literal["local_legal", "legalbench_rag"]
    path: str | None = None
    subsets: list[str] | None = None
    sampling: Literal["none", "random", "lbr_mini"] = "none"
    max_queries_per_subset: int | None = None
    seed: int = 0


class GenerationConfig(_Strict):
    """OpenAI-compatible endpoint (local vLLM by default)."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    context_k: int = 3


class ExperimentFile(_Strict):
    dataset: DatasetConfig
    eval_ks: list[int] = [1, 3, 5, 10]
    output_dir: str = "results"
    cache_dir: str = "data/cache"
    device: str | None = None
    generation: GenerationConfig | None = None
    configs: list[RetrievalConfig]

    @model_validator(mode="after")
    def _check(self) -> ExperimentFile:
        names = [c.name for c in self.configs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate config names: {sorted(dupes)}")
        if not self.configs:
            raise ValueError("no configs defined")
        for c in self.configs:
            if max(self.eval_ks) > c.top_k:
                raise ValueError(f"[{c.name}] top_k={c.top_k} < max(eval_ks)={max(self.eval_ks)}")
        return self


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _short(value: Any) -> str:
    s = str(value).rsplit("/", 1)[-1] if value is not None else "none"
    return s.replace(" ", "")


def expand_configs(raw: dict) -> list[dict]:
    base = raw.get("base", {}) or {}
    out: list[dict] = []
    for name, override in (raw.get("configs") or {}).items():
        out.append(_deep_merge(base, {**(override or {}), "name": name}))
    grid = raw.get("grid") or {}
    if grid:
        keys = list(grid)
        seen: set[str] = set()
        for values in itertools.product(*(grid[k] for k in keys)):
            cfg = copy.deepcopy(base)
            for k, v in zip(keys, values):
                _set_dotted(cfg, k, v)
            if cfg.get("retriever") == "bm25":
                cfg["embedding_model"] = None  # unused by BM25: avoid duplicate grid points
            sig = repr(sorted((k, repr(v)) for k, v in cfg.items()))
            if sig in seen:
                continue
            seen.add(sig)
            parts = []
            for k in keys:
                v = cfg
                for part in k.split("."):
                    v = v.get(part) if isinstance(v, dict) else None
                parts.append(f"{k.rsplit('.', 1)[-1]}={_short(v)}")
            cfg["name"] = "__".join(parts)
            out.append(cfg)
    return out


def load_experiment(path: str | Path) -> ExperimentFile:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = dict(raw)
    configs = expand_configs(raw)
    raw.pop("base", None)
    raw.pop("grid", None)
    raw["configs"] = configs
    return ExperimentFile.model_validate(raw)
