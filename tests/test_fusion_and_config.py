import textwrap

import pytest

from retrievalops.config import load_experiment
from retrievalops.retrieval.indexes import BM25Index, rrf_fuse


def test_rrf_rewards_agreement():
    a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    b = [(3, 9.0), (1, 8.0), (4, 7.0)]
    fused = rrf_fuse([a, b], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]
    assert len(fused) == 4
    assert rrf_fuse([a, b], k=60, limit=2) == fused[:2]


def test_bm25_ranks_matching_chunk_first_and_handles_stopword_query():
    idx = BM25Index(["the cat sat on the mat", "indemnification obligations of the supplier", "dogs bark"])
    res = idx.search(["Who bears indemnification obligations?", "the of and", "zebra"], k=5)
    assert res[0][0][0] == 1
    assert res[1] == []
    assert res[2] == []


def _write(tmp_path, body):
    p = tmp_path / "exp.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_grid_and_named_configs(tmp_path):
    p = _write(tmp_path, """
        dataset: {name: local_legal, path: data}
        eval_ks: [1, 5]
        base:
          retriever: dense
          embedding_model: m
          top_k: 10
        configs:
          bm25_only: {retriever: bm25}
        grid:
          chunking.chunk_size: [256, 512]
          retriever: [dense, hybrid]
    """)
    exp = load_experiment(p)
    names = [c.name for c in exp.configs]
    assert names[0] == "bm25_only"
    assert len(names) == 5 and len(set(names)) == 5
    assert {c.chunking.chunk_size for c in exp.configs[1:]} == {256, 512}


def test_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="embedding_model"):
        load_experiment(_write(tmp_path, """
            dataset: {name: local_legal}
            configs: {x: {retriever: dense}}
        """))
    with pytest.raises(ValueError, match="top_k"):
        load_experiment(_write(tmp_path, """
            dataset: {name: local_legal}
            eval_ks: [1, 20]
            configs: {x: {retriever: bm25, top_k: 10, candidate_k: 50}}
        """))
    with pytest.raises(ValueError):
        load_experiment(_write(tmp_path, """
            dataset: {name: local_legal}
            configs: {x: {retriever: bm25, typo_field: 1}}
        """))


def test_grid_drops_duplicate_bm25_points(tmp_path):
    exp = load_experiment(_write(tmp_path, """
        dataset: {name: local_legal}
        eval_ks: [1]
        base: {top_k: 5, candidate_k: 10}
        grid:
          retriever: [dense, bm25]
          embedding_model: [m1, m2]
    """))
    assert [c.retriever for c in exp.configs].count("bm25") == 1
    assert len(exp.configs) == 3
