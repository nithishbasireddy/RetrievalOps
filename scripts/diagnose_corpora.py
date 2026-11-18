"""Why do label-free selectors fail on some corpora? Compact diagnostics per corpus.

    python scripts/diagnose_corpora.py configs/lbr_full_grid_v1.yaml --run-name lbr_full_grid_v1
Uses only cached judgments (seed 0, first 100 queries per corpus, 4000c).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrievalops.analysis import proxies as P
from retrievalops.analysis.selection import matrix, oracle
from retrievalops.config import load_experiment
from retrievalops.datasets import load_dataset
from retrievalops.judging import LLMJudge
from retrievalops.metrics import _intersection_len


def short(c: str) -> str:
    d = dict(kv.split("=") for kv in c.split("__"))
    emb = {"e5-large-v2": "e5", "Qwen3-Embedding-0.6B": "qwen3", "none": "-"}.get(d["embedding_model"], d["embedding_model"])
    return f"{d['chunk_size']}/{d['retriever']}/{emb}/{'rr' if d['reranker'] != 'none' else 'no-rr'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judge", default="Qwen/Qwen3.5-27B")
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    run_dir = Path(exp.output_dir) / args.run_name
    ds = load_dataset(exp.dataset)
    queries = {q.qid: q for q in ds.queries}
    docs = {d.doc_id: d for d in ds.documents}
    df = pd.read_csv(run_dir / "analysis" / "per_query_metrics.csv")
    metric = f"recall@{args.budget}c"
    judge = LLMJudge(args.judge, base_url="http://127.0.0.1:9/v1", cache_path=Path(exp.cache_dir) / "judgments.sqlite")
    pd.set_option("display.width", 200)

    print("=== (d) query / gold-span shape per corpus ===")
    rows = []
    for corpus in sorted({q.subset for q in ds.queries}):
        qs = [q for q in ds.queries if q.subset == corpus]
        lens = [s.end - s.start for q in qs for s in q.gold_spans]
        rows.append({"corpus": corpus, "queries": len(qs), "spans/query": np.mean([len(q.gold_spans) for q in qs]),
                     "median_span_chars": np.median(lens), "gold_chars/query": np.mean([sum(s.end - s.start for s in q.gold_spans) for q in qs]),
                     "docs": len({s.doc_id for q in qs for s in q.gold_spans}),
                     "queries/doc": len(qs) / len({s.doc_id for q in qs for s in q.gold_spans}),
                     "distinct_query_templates": len(Counter(q.text.split(";", 1)[-1].strip() for q in qs))})
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    for corpus in sorted({q.subset for q in ds.queries}):
        M, configs, cols = matrix(df, metric, corpus)
        orc = oracle(M, configs)
        q = orc.quality
        print(f"\n=== {corpus}: (a) oracle spread: best {q.max():.3f}, 2nd {np.sort(q)[-2]:.3f}, "
              f"median {np.median(q):.3f}, worst {q.min():.3f}; tied {int(orc.tie.sum())}/{len(q)}; "
              f"top-5 range {np.sort(q)[-1] - np.sort(q)[-5]:.3f}")
        pool_q = sorted(df[df.subset == corpus].qid.unique())
        order = random.Random(f"0/{corpus}").sample(pool_q, min(args.n, len(pool_q)))
        trunc, full = P.load_contexts(run_dir, set(order), args.budget)
        rels = P.run_judge(judge, P.judge_pool(full, order), queries, docs, desc=corpus)
        _, rec = P.judged_scores(trunc, full, order, rels["strict"])
        sel = [cols.index(x) for x in order]
        t = pd.DataFrame({"config": [short(c) for c in configs], "oracle": q,
                          "gold@sample": M[:, sel].mean(axis=1),
                          "judge_recall": [rec[c].mean() for c in configs]})
        t["oracle_rank"] = t.oracle.rank(ascending=False).astype(int)
        t["judge_rank"] = t.judge_recall.rank(ascending=False).astype(int)
        print("(b) top-6 by oracle and top-6 by judge:")
        print(t.sort_values("oracle", ascending=False).head(6).round(3).to_string(index=False))
        print(t.sort_values("judge_recall", ascending=False).head(6).round(3).to_string(index=False))
        print("    mean oracle/judge rank by factor:")
        for fac, idx in (("chunk", 0), ("retriever", 1), ("rerank", 3)):
            t[fac] = t.config.str.split("/").str[idx]
            print("     ", t.groupby(fac)[["oracle_rank", "judge_rank"]].mean().round(1).to_dict("index"))

        # (c) judge positives that miss gold: same document (gold incomplete?) or wrong document?
        pos_gold = pos_samedoc = pos_otherdoc = 0
        for (qid, s), r in rels["strict"].items():
            if r < 0.5:
                continue
            gold = queries[qid].gold_spans
            g_iv = [(g.start, g.end) for g in gold if g.doc_id == s.doc_id]
            if _intersection_len([(s.start, s.end)], sorted(g_iv)) > 0:
                pos_gold += 1
            elif g_iv:
                pos_samedoc += 1
            else:
                pos_otherdoc += 1
        tot = max(1, pos_gold + pos_samedoc + pos_otherdoc)
        print(f"(c) strict judge positives: overlap gold {pos_gold/tot:.0%}, right doc but outside gold "
              f"{pos_samedoc/tot:.0%}, wrong doc {pos_otherdoc/tot:.0%}  (n={tot})")


if __name__ == "__main__":
    main()
