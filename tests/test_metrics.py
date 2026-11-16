import math

import pytest

from retrievalops.metrics import all_k_metrics, count_relevant, merge_intervals, span_metrics
from retrievalops.types import Span


def test_merge_intervals():
    assert merge_intervals([(5, 10), (0, 3), (2, 4), (10, 12), (20, 20)]) == [(0, 4), (5, 12)]


def test_perfect_retrieval():
    gold = [Span("a", 10, 20)]
    m = span_metrics([Span("a", 10, 20)], gold, k=1)
    assert m["precision"] == m["recall"] == m["hit"] == m["mrr"] == m["ndcg"] == 1.0
    assert m["precision_lbr"] == m["recall_lbr"] == 1.0


def test_partial_and_wrong_doc():
    gold = [Span("a", 0, 100)]
    ret = [Span("b", 0, 100), Span("a", 50, 150)]
    m1 = span_metrics(ret, gold, k=1)
    assert m1["hit"] == 0 and m1["precision"] == 0 and m1["recall"] == 0
    m2 = span_metrics(ret, gold, k=2)
    assert m2["recall"] == pytest.approx(0.5)
    assert m2["precision"] == pytest.approx(50 / 200)
    assert m2["mrr"] == pytest.approx(0.5)
    assert m2["doc_recall"] == 1.0


def test_union_metrics_not_inflated_by_overlapping_chunks():
    gold = [Span("a", 0, 100)]
    ret = [Span("a", 0, 100), Span("a", 0, 100)]
    m = span_metrics(ret, gold, k=2)
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["recall_lbr"] == 2.0  # reference formula double counts - documented behaviour


def test_ndcg_with_true_relevant_count_is_bounded():
    gold = [Span("a", 0, 1000)]
    chunks = [Span("a", i, i + 100) for i in range(0, 1000, 100)] + [Span("a", 2000, 2100)]
    n_rel = count_relevant(chunks, gold)
    assert n_rel == 10
    m = span_metrics(chunks[:5], gold, k=5, n_relevant=n_rel)
    assert m["ndcg"] == pytest.approx(1.0)
    m_bad = span_metrics([chunks[-1]] + chunks[:4], gold, k=5, n_relevant=n_rel)
    assert 0 < m_bad["ndcg"] < 1


def test_empty_retrieval():
    m = span_metrics([], [Span("a", 0, 10)], k=3)
    assert m["precision"] == m["recall"] == m["hit"] == m["mrr"] == m["ndcg"] == 0.0


def test_all_k_keys():
    m = all_k_metrics([Span("a", 0, 10)], [Span("a", 0, 10)], ks=[1, 5])
    assert {"recall@1", "recall@5", "ndcg@5"} <= set(m)
    assert not any(math.isnan(v) for v in m.values())
