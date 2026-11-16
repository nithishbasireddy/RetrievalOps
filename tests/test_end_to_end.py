"""GPU smoke test: real models, real local corpus, both dataset types."""

import json
import os
from pathlib import Path

import pytest

from retrievalops.config import load_experiment
from retrievalops.runner import Runner

REPO = Path(__file__).resolve().parent.parent
LBR = Path(os.environ.get("LBR_PATH", REPO.parent / "datasets" / "legalbenchrag"))

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")]


def _exp(tmp_path, dataset_yaml):
    p = tmp_path / "e.yaml"
    p.write_text(f"""
dataset: {dataset_yaml}
eval_ks: [1, 5]
output_dir: {tmp_path / 'results'}
cache_dir: {tmp_path / 'cache'}
base:
  chunking: {{chunk_size: 500, chunk_overlap: 0}}
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  top_k: 5
  candidate_k: 20
configs:
  dense: {{retriever: dense}}
  bm25: {{retriever: bm25}}
  hybrid: {{retriever: hybrid}}
  hybrid_rerank: {{retriever: hybrid, reranker: BAAI/bge-reranker-v2-m3}}
""")
    return load_experiment(p)


def test_local_legal_end_to_end(tmp_path):
    exp = _exp(tmp_path, f"{{name: local_legal, path: {REPO / 'data'}}}")
    runner = Runner(exp, run_name="t")
    runner.run()
    for name in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        with open(runner.run_dir / "per_query" / f"{name}.jsonl") as fh:
            recs = [json.loads(x) for x in fh]
        assert len(recs) == 10
        assert all(1 <= len(r["retrieved"]) <= 5 for r in recs)


@pytest.mark.skipif(not LBR.exists(), reason="LegalBench-RAG not downloaded")
def test_lbr_subset_end_to_end_and_resume(tmp_path):
    exp = _exp(tmp_path, f"{{name: legalbench_rag, path: {LBR}, subsets: [privacy_qa]}}")
    runner = Runner(exp, run_name="t", max_queries=30)
    summary = runner.run()
    allrows = summary[summary.subset == "ALL_micro"].set_index("config")
    assert set(allrows.index) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    for col in ["recall@5", "precision@5", "ndcg@5"]:
        assert ((allrows[col] >= 0) & (allrows[col] <= 1)).all()
    assert (allrows["recall@5"] > 0).all()  # every method finds something on privacy_qa
    # rerun is a no-op (resume) and gives identical summary
    summary2 = Runner(exp, run_name="t", max_queries=30).run()
    assert summary2.equals(summary)
