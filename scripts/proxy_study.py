"""Label-free configuration selection study on a finished grid run.

    python scripts/proxy_study.py configs/lbr_full_grid_v1.yaml --run-name lbr_full_grid_v1 \
        --budget 4000 --n-unlabelled 100 \
        [--judge-model Qwen/Qwen2.5-3B-Instruct --judge-url http://127.0.0.1:8010/v1]

Writes to results/<run>/analysis/:
    per_query_metrics.csv     oracle metrics for every (config, query)
    oracle_<corpus>.csv       true ranking, regret and tie set per corpus
    label_budget.csv          regret of choosing with n labelled queries
    proxies.csv               each proxy x corpus: Kendall tau, regret, pick in tie set
    judge_validity.json       judge vs gold at chunk level
    fixed_choice.csv          regret of using one global config for every corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrievalops.analysis import proxies as P
from retrievalops.analysis.selection import (
    evaluate_scores,
    fixed_choice_baselines,
    label_budget_curve,
    load_per_query,
    matrix,
    oracle,
)
from retrievalops.config import load_experiment
from retrievalops.datasets import load_dataset

log = logging.getLogger("proxy_study")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--n-unlabelled", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--judge-tag", help="name for this judge in outputs (default: model name)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    exp = load_experiment(args.experiment)
    run_dir = Path(exp.output_dir) / args.run_name
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(exp.dataset)
    metric = f"recall@{args.budget}c"

    cache = out / "per_query_metrics.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = load_per_query(run_dir, ds)
        df.to_csv(cache, index=False)
    corpora = sorted(df.subset.unique())
    log.info("Loaded %d (config, query) rows; corpora: %s", len(df), corpora)

    # ---- oracle + label-budget baseline ------------------------------------------------------
    orcs, curves = {}, []
    for corpus in corpora:
        M, configs, _ = matrix(df, metric, corpus)
        orcs[corpus] = oracle(M, configs, seed=args.seed)
        orcs[corpus].to_frame().to_csv(out / f"oracle_{corpus}__{metric}.csv", index=False)
        curve = label_budget_curve(M, orcs[corpus], seed=args.seed)
        curve.insert(0, "corpus", corpus)
        curves.append(curve)
    pd.concat(curves).to_csv(out / f"label_budget__{metric}.csv", index=False)
    fixed_choice_baselines(orcs).to_csv(out / f"fixed_choice__{metric}.csv", index=False)

    # ---- label-free proxies on a sample of unlabelled queries ---------------------------------
    queries = {q.qid: q for q in ds.queries}
    docs = {d.doc_id: d for d in ds.documents}
    rng = random.Random(args.seed)
    rows = []
    judge = None
    if args.judge_model:
        from retrievalops.judging import LLMJudge

        judge = LLMJudge(args.judge_model, base_url=args.judge_url, cache_path=Path(exp.cache_dir) / "judgments.sqlite")
    judge_tag = args.judge_tag or (args.judge_model or "").rsplit("/", 1)[-1]
    validity = {}
    for corpus in corpora:
        orc = orcs[corpus]
        pool_q = sorted(df[df.subset == corpus].qid.unique())
        qids = sorted(rng.sample(pool_q, min(args.n_unlabelled, len(pool_q))))
        trunc, full = P.load_contexts(run_dir, set(qids), args.budget)
        assert list(trunc) == orc.configs, "config order mismatch between run files and oracle"

        n_q = len(qids)

        def record(name: str, per_query: dict[str, np.ndarray], orc=orc, corpus=corpus, n=n_q) -> None:
            scores = np.array([per_query[c].mean() for c in orc.configs])
            rows.append({"corpus": corpus, "proxy": name, "n_unlabelled": n, **evaluate_scores(scores, orc)})

        record("consensus", P.consensus_scores(trunc, qids))
        if judge is not None:
            pool = P.judge_pool(full, qids)
            rel = P.run_judge(judge, pool, queries, docs, desc=f"judge {corpus}")
            prec, rec = P.judged_scores(trunc, full, qids, rel)
            record(f"judge_precision[{judge_tag}]", prec)
            record(f"judge_recall[{judge_tag}]", rec)
            validity[corpus] = P.judge_vs_gold(rel, queries)
        # reference point: the same queries *with* their gold labels
        M, configs, cols = matrix(df, metric, corpus)
        sel = [cols.index(q) for q in qids if q in cols]
        record("gold_labels_same_queries", {c: M[i, sel] for i, c in enumerate(configs)})

    prox = pd.DataFrame(rows)
    proxy_path = out / f"proxies__{metric}__n{args.n_unlabelled}_s{args.seed}.csv"
    if proxy_path.exists():  # keep results from other judges run with the same settings
        old = pd.read_csv(proxy_path)
        prox = pd.concat([old[~old.set_index(["corpus", "proxy"]).index.isin(
            prox.set_index(["corpus", "proxy"]).index)], prox])
    prox.to_csv(proxy_path, index=False)
    if validity:
        (out / f"judge_validity__{judge_tag}__n{args.n_unlabelled}_s{args.seed}.json").write_text(
            json.dumps(validity, indent=2))

    pd.set_option("display.width", 200)
    print("\nORACLE best / #tied per corpus:")
    print(fixed_choice_baselines(orcs).to_string(index=False))
    print("\nLABEL-BUDGET CURVE:")
    print(pd.concat(curves).round(4).to_string(index=False))
    print("\nPROXIES:")
    print(prox.drop(columns=["picked"]).round(4).to_string(index=False))
    if validity:
        print("\nJUDGE VALIDITY:", json.dumps(validity, indent=1))


if __name__ == "__main__":
    main()
