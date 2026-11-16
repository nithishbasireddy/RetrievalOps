# RetrievalOps

Reproducible evaluation and configuration selection for retrieval-augmented generation (RAG)
pipelines, starting with legal documents.

RetrievalOps runs a set of retrieval configurations (chunking × retriever × embedder × reranker)
over a dataset. It scores every ranked list against character-level gold evidence spans and
records per-query results, so configurations can be compared and selected with statistical care.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Data

| Dataset | Gold labels | Location |
|---|---|---|
| `local_legal`: the original 5 contracts + 10 questions | reference answers only | `data/` |
| `legalbench_rag`: LegalBench-RAG ([Pipitone & Alami, 2024](https://arxiv.org/abs/2408.10343)); 6,889 queries over CUAD, MAUD, ContractNLI, PrivacyQA | character spans | download from the [LegalBench-RAG repo](https://github.com/zeroentropy-cc/legalbenchrag), unzip so that `corpus/` and `benchmarks/` sit in one folder (default `../datasets/legalbenchrag`) |

`sampling: lbr_mini` reproduces the official LegalBench-RAG-mini protocol: 194 queries per subset,
with the corpus restricted to the documents those queries reference.

## Run

```bash
retrievalops list configs/lbr_grid_v1.yaml           # show the configs a file expands to
retrievalops run  configs/lbr_mini_baselines.yaml    # run them all
retrievalops run  configs/lbr_mini_baselines.yaml --only bm25 dense_e5 --max-queries 50
retrievalops run  configs/lbr_grid_v1.yaml --run-name grid_v1   # re-run the same name to resume
```

Each run writes to `results/<run_name>/`:

- `manifest.json`: the experiment config, dataset statistics, git commit, library versions
- `per_query/<config>.jsonl`: ranked chunks (with character offsets) and metrics for each query
- `per_query/<config>.meta.json`: chunk count, index build time, latency
- `summary.csv`: mean metrics per config × subset, micro and macro averages, and bootstrap 95% CIs

## Metrics

- **`precision@k` / `recall@k`**: character-set overlap between the top-k chunks and the gold spans.
  These are the primary metrics. They stay in [0, 1] and chunk overlap cannot inflate them.
- **`precision_lbr@k` / `recall_lbr@k`**: the exact LegalBench-RAG reference formulas, for comparison
  with published numbers.
- **`hit@k`, `mrr@k`, `ndcg@k`**: chunk-level ranking metrics. A chunk counts as relevant if it
  overlaps a gold span.
- **`doc_recall@k`**: the fraction of gold documents that appear in the top-k.

## Configuration files

An experiment YAML has one `dataset`, one `base` config, and named `configs` and/or a `grid` of
dotted keys (expanded as a cartesian product; BM25 duplicates are dropped). The schema is validated
in `retrievalops/config.py`, and unknown keys are an error.

Retriever options: `dense` (exact cosine search on GPU), `bm25` (bm25s, stemming + stop-words), and
`hybrid` (Reciprocal Rank Fusion). Any of them can take an optional cross-encoder `reranker`. Each
embedder gets its correct query/passage prompts (E5, BGE, Qwen3-Embedding, ...) automatically.

## Tests

```bash
pytest -m "not gpu"   # fast unit tests
pytest                # + end-to-end GPU tests with real models
```

## Layout

```
retrievalops/
  config.py            experiment schema + grid expansion
  datasets/            local_legal, legalbench_rag loaders
  ingestion/           file loaders, offset-preserving chunkers
  embeddings.py        encoders with model-specific prompts, on-disk embedding cache
  retrieval/           dense / BM25 / RRF / cross-encoder reranker, RetrievalPipeline
  metrics.py           span-level + ranking metrics
  generation.py        answer generation through an OpenAI-compatible endpoint (local vLLM)
  runner.py            experiment runner (resumable), summaries with bootstrap CIs
  cli.py               `retrievalops` command
configs/               experiment files
tests/                 unit + end-to-end tests
```
