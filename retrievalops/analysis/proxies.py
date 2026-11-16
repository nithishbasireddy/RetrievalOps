"""Label-free proxies for configuration quality.

All proxies see only *unlabelled* queries (the text a deployment would log) and the configs'
retrieved context, truncated to the same character budget used by the oracle metric.

* consensus         agreement with the other configs: mean fraction of configs that also retrieved
                    each character of this config's context (a no-LLM baseline, in the spirit of
                    "ranking retrieval systems without relevance judgments", Soboroff et al. 2001).
* judge_precision   length-weighted LLM-judged relevance of the context.
* judge_recall      pooled recall: share of the judged-relevant text (union over all configs'
                    contexts, as in TREC pooling) that this config retrieved.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from retrievalops.metrics import _intersection_len, merge_intervals, truncate_to_budget
from retrievalops.types import Document, Query, Span

Contexts = dict[str, dict[str, list[Span]]]  # config -> qid -> budget-truncated spans


def load_contexts(run_dir: str | Path, qids: set[str], budget: int) -> tuple[Contexts, Contexts]:
    """Return (truncated contexts, full chunks) for the given queries, for every config."""
    trunc: Contexts = {}
    full: Contexts = {}
    for path in sorted((Path(run_dir) / "per_query").glob("*.jsonl")):
        t, f = {}, {}
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r["qid"] not in qids:
                    continue
                spans = [Span(x["doc_id"], x["start"], x["end"]) for x in r["retrieved"]]
                t[r["qid"]] = truncate_to_budget(spans, budget)
                f[r["qid"]] = spans[: len(t[r["qid"]])]
        trunc[path.stem], full[path.stem] = t, f
    return trunc, full


def _by_doc(spans: Sequence[Span]) -> dict[str, list[tuple[int, int]]]:
    d: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s in spans:
        d[s.doc_id].append((s.start, s.end))
    return {k: merge_intervals(v) for k, v in d.items()}


def consensus_scores(ctx: Contexts, qids: Sequence[str]) -> dict[str, np.ndarray]:
    """Per config, per query: mean (over its chars) fraction of *other* configs covering that char."""
    configs = list(ctx)
    n = len(configs)
    out = {c: np.zeros(len(qids)) for c in configs}
    for j, qid in enumerate(qids):
        ivs = {c: _by_doc(ctx[c].get(qid, [])) for c in configs}
        docs = {d for c in configs for d in ivs[c]}
        # elementary segments per doc with coverage counts
        cover: dict[str, list[tuple[int, int, int]]] = {}
        for d in docs:
            events = []
            for c in configs:
                for s, e in ivs[c].get(d, []):
                    events += [(s, 1), (e, -1)]
            events.sort()
            segs, cnt, prev = [], 0, None
            for pos, delta in events:
                if prev is not None and pos > prev and cnt > 0:
                    segs.append((prev, pos, cnt))
                cnt += delta
                prev = pos
            cover[d] = segs
        for c in configs:
            total = agree = 0.0
            for d, cis in ivs[c].items():
                for s, e in cis:
                    for a, b, cnt in cover[d]:
                        lo, hi = max(s, a), min(e, b)
                        if hi > lo:
                            agree += (hi - lo) * (cnt - 1) / max(1, n - 1)
                    total += e - s
            out[c][j] = agree / total if total else 0.0
    return out


def judge_pool(full: Contexts, qids: Sequence[str]) -> dict[str, set[Span]]:
    """Unique full chunks to judge per query (union over configs)."""
    pool: dict[str, set[Span]] = defaultdict(set)
    for c in full:
        for q in qids:
            pool[q].update(full[c].get(q, []))
    return pool


def run_judge(judge, pool: dict[str, set[Span]], queries: dict[str, Query], docs: dict[str, Document],
              desc: str = "judging") -> dict[tuple[str, Span], float]:
    items = [(q, s) for q in pool for s in sorted(pool[q], key=lambda x: (x.doc_id, x.start))]
    from retrievalops.judging import JudgeRequest

    reqs = [JudgeRequest(queries[q].text, s.doc_id, docs[s.doc_id].text[s.start : s.end]) for q, s in items]
    judgments = judge.judge(reqs, desc=desc)
    return {item: j.relevance for item, j in zip(items, judgments)}


def judged_scores(trunc: Contexts, full: Contexts, qids: Sequence[str], rel: dict[tuple[str, Span], float],
                  threshold: float = 0.5) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    configs = list(trunc)
    precision = {c: np.zeros(len(qids)) for c in configs}
    recall = {c: np.zeros(len(qids)) for c in configs}
    for j, q in enumerate(qids):
        relevant = [s for c in configs for s in full[c].get(q, []) if rel[(q, s)] >= threshold]
        rel_iv = _by_doc(relevant)
        rel_len = sum(e - s for ivs in rel_iv.values() for s, e in ivs)
        for c in configs:
            t, f = trunc[c].get(q, []), full[c].get(q, [])
            lengths = np.array([s.end - s.start for s in t], dtype=float)
            scores = np.array([rel[(q, fs)] for fs in f], dtype=float)
            precision[c][j] = float((lengths * scores).sum() / lengths.sum()) if lengths.sum() else 0.0
            if rel_len:
                civ = _by_doc(t)
                inter = sum(_intersection_len(civ[d], rel_iv[d]) for d in civ if d in rel_iv)
                recall[c][j] = inter / rel_len
    return precision, recall


def judge_vs_gold(rel: dict[tuple[str, Span], float], queries: dict[str, Query]) -> dict[str, float]:
    """Chunk-level validity of the judge against gold spans (a chunk is relevant if it overlaps gold)."""
    from sklearn.metrics import cohen_kappa_score, roc_auc_score

    y_true, y_score = [], []
    for (q, s), r in rel.items():
        gold = _by_doc(queries[q].gold_spans)
        overlap = _intersection_len([(s.start, s.end)], gold.get(s.doc_id, []))
        y_true.append(int(overlap > 0))
        y_score.append(r)
    y_true_a, y_score_a = np.array(y_true), np.array(y_score)
    out = {"n_pairs": float(len(y_true)), "gold_positive_rate": float(y_true_a.mean())}
    if 0 < y_true_a.sum() < len(y_true_a):
        out["auc"] = float(roc_auc_score(y_true_a, y_score_a))
        out["kappa@0.5"] = float(cohen_kappa_score(y_true_a, (y_score_a >= 0.5).astype(int)))
        pred = y_score_a >= 0.5
        out["precision@0.5"] = float(y_true_a[pred].mean()) if pred.any() else 0.0
        out["recall@0.5"] = float(pred[y_true_a == 1].mean())
    return out
