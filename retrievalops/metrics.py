"""Retrieval metrics against character-level gold spans.

Two families are reported:

* ``precision_lbr`` / ``recall_lbr`` reproduce the LegalBench-RAG reference formulas exactly
  (sums of pairwise overlaps; overlapping retrieved chunks are counted more than once, so
  ``recall_lbr`` can exceed 1 when chunks overlap). Use these only to compare with published numbers.
* ``precision`` / ``recall`` use character *sets* (union of retrieved chars vs union of gold chars),
  so they stay in [0, 1] and are not inflated by chunk overlap. These are the primary metrics.

Plus rank-aware chunk-level metrics, where a chunk counts as relevant if it overlaps any gold span:
``hit`` (any relevant chunk in top-k), ``mrr`` and ``ndcg`` (binary gain).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from retrievalops.types import Span

Interval = tuple[int, int]


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    out: list[list[int]] = []
    for s, e in sorted(i for i in intervals if i[1] > i[0]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _by_doc(spans: Iterable[Span]) -> dict[str, list[Interval]]:
    d: dict[str, list[Interval]] = defaultdict(list)
    for s in spans:
        d[s.doc_id].append((s.start, s.end))
    return {k: merge_intervals(v) for k, v in d.items()}


def _intersection_len(a: list[Interval], b: list[Interval]) -> int:
    """Length of the intersection of two sorted, merged interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _overlaps(span: Span, gold: dict[str, list[Interval]]) -> bool:
    return any(min(span.end, e) > max(span.start, s) for s, e in gold.get(span.doc_id, []))


def lbr_precision_recall(retrieved: Sequence[Span], gold: Sequence[Span]) -> tuple[float, float]:
    def overlap(a: Span, b: Span) -> int:
        if a.doc_id != b.doc_id:
            return 0
        return max(0, min(a.end, b.end) - max(a.start, b.start))

    total_ret = sum(r.end - r.start for r in retrieved)
    total_gold = sum(g.end - g.start for g in gold)
    rel = sum(overlap(r, g) for r in retrieved for g in gold)
    precision = rel / total_ret if total_ret else 0.0
    recall = rel / total_gold if total_gold else 0.0
    return precision, recall


def span_metrics(
    retrieved: Sequence[Span], gold: Sequence[Span], k: int, n_relevant: int | None = None
) -> dict[str, float]:
    """Metrics for the top-``k`` of a ranked list of retrieved spans.

    ``n_relevant`` is the number of corpus chunks overlapping any gold span (for the nDCG ideal);
    if unknown, the number of gold spans is used as an approximation.
    """
    top = list(retrieved[:k])
    gold_iv = _by_doc(gold)
    ret_iv = _by_doc(top)
    ret_len = sum(e - s for ivs in ret_iv.values() for s, e in ivs)
    gold_len = sum(e - s for ivs in gold_iv.values() for s, e in ivs)
    inter = sum(_intersection_len(ret_iv[d], gold_iv[d]) for d in ret_iv if d in gold_iv)

    rel = [_overlaps(s, gold_iv) for s in top]
    first = next((i for i, r in enumerate(rel) if r), None)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    n_ideal = min(k, n_relevant if n_relevant is not None else len(gold))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    p_lbr, r_lbr = lbr_precision_recall(top, gold)
    gold_docs = set(gold_iv)
    return {
        "precision": inter / ret_len if ret_len else 0.0,
        "recall": inter / gold_len if gold_len else 0.0,
        "precision_lbr": p_lbr,
        "recall_lbr": r_lbr,
        "hit": float(first is not None),
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "doc_recall": len(gold_docs & {s.doc_id for s in top}) / len(gold_docs) if gold_docs else 0.0,
    }


def count_relevant(chunk_spans: Iterable[Span], gold: Sequence[Span]) -> int:
    gold_iv = _by_doc(gold)
    return sum(1 for c in chunk_spans if c.doc_id in gold_iv and _overlaps(c, gold_iv))


def all_k_metrics(
    retrieved: Sequence[Span], gold: Sequence[Span], ks: Sequence[int], n_relevant: int | None = None
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        for name, v in span_metrics(retrieved, gold, k, n_relevant).items():
            out[f"{name}@{k}"] = v
    return out
