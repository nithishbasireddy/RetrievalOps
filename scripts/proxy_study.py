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
    equivalent_labels,
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
    ap.add_argument("--n-unlabelled", type=int, nargs="+", default=[100],
                    help="query-sample sizes; samples are nested (smaller ones are prefixes)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
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
        orcs[corpus] = oracle(M, configs, seed=0)
        orcs[corpus].to_frame().to_csv(out / f"oracle_{corpus}__{metric}.csv", index=False)
        curve = label_budget_curve(M, orcs[corpus], seed=0)
        curve.insert(0, "corpus", corpus)
        curves.append(curve)
    pd.concat(curves).to_csv(out / f"label_budget__{metric}.csv", index=False)
    fixed_choice_baselines(orcs).to_csv(out / f"fixed_choice__{metric}.csv", index=False)

    # ---- label-free proxies on samples of unlabelled queries ----------------------------------
    queries = {q.qid: q for q in ds.queries}
    docs = {d.doc_id: d for d in ds.documents}
    judge = None
    if args.judge_model:
        from retrievalops.judging import LLMJudge

        judge = LLMJudge(args.judge_model, base_url=args.judge_url, cache_path=Path(exp.cache_dir) / "judgments.sqlite")
    judge_tag = args.judge_tag or (args.judge_model or "").rsplit("/", 1)[-1]
    curves_by_corpus = {c: g for c, g in pd.concat(curves).groupby("corpus")}
    rows, validity = [], {}
    n_max = max(args.n_unlabelled)
    for seed in args.seeds:
        for corpus in corpora:
            orc = orcs[corpus]
            M, configs, cols = matrix(df, metric, corpus)
            pool_q = sorted(df[df.subset == corpus].qid.unique())
            order = random.Random(f"{seed}/{corpus}").sample(pool_q, min(n_max, len(pool_q)))
            trunc_all, full_all = P.load_contexts(run_dir, set(order), args.budget)
            assert list(trunc_all) == orc.configs, "config order mismatch between run files and oracle"
            rels_all = {}
            if judge is not None:  # judge the largest sample once; smaller samples are its prefixes
                rels_all = P.run_judge(judge, P.judge_pool(full_all, order), queries, docs,
                                       desc=f"judge {corpus} s{seed}")
                validity[f"{corpus}/s{seed}"] = {m: P.judge_vs_gold(r, queries) for m, r in rels_all.items()}
            for n in sorted(set(args.n_unlabelled)):
                qids = order[:n]
                n_q = len(qids)

                def record(name, per_query, orc=orc, corpus=corpus, n=n_q, seed=seed):
                    scores = np.array([per_query[c].mean() for c in orc.configs])
                    rows.append({"corpus": corpus, "proxy": name, "n_unlabelled": n, "seed": seed,
                                 **evaluate_scores(scores, orc)})

                record("consensus", P.consensus_scores(trunc_all, qids))
                for mode, rel in rels_all.items():
                    prec, rec = P.judged_scores(trunc_all, full_all, qids, rel)
                    record(f"judge_precision[{judge_tag},{mode}]", prec)
                    record(f"judge_recall[{judge_tag},{mode}]", rec)
                sel = [cols.index(q) for q in qids if q in cols]
                record("gold_labels_same_queries", {c: M[i, sel] for i, c in enumerate(configs)})

    prox = pd.DataFrame(rows)
    proxy_path = out / f"proxies__{metric}.csv"
    key = ["corpus", "proxy", "n_unlabelled", "seed"]
    if proxy_path.exists():  # merge with results of other judges / settings
        old = pd.read_csv(proxy_path)
        keep = ~old.set_index(key).index.isin(prox.set_index(key).index)
        prox = pd.concat([old[keep], prox], ignore_index=True)
    prox.to_csv(proxy_path, index=False)
    if validity:
        vpath = out / f"judge_validity__{judge_tag}.json"
        old_v = json.loads(vpath.read_text()) if vpath.exists() else {}
        vpath.write_text(json.dumps({**old_v, **validity}, indent=2))

    # ---- summary: mean over seeds, and equivalent number of labels ----------------------------
    summ = (prox.groupby(["proxy", "n_unlabelled", "corpus"])
            .agg(kendall_tau=("kendall_tau", "mean"), regret=("regret", "mean"),
                 pick_in_tie_set=("pick_in_tie_set", "mean"), n_seeds=("seed", "nunique"))
            .reset_index())
    summ["equiv_labels"] = [equivalent_labels(curves_by_corpus[r.corpus], r.regret) for r in summ.itertuples()]
    summ.to_csv(out / f"proxy_summary__{metric}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nORACLE best / #tied per corpus:")
    print(fixed_choice_baselines(orcs).to_string(index=False))
    print("\nLABEL-BUDGET CURVE (selected n):")
    lc = pd.concat(curves)
    print(lc[lc.n_labels.isin([1, 5, 10, 20, 50, 100])].round(4).to_string(index=False))
    for col in ("kendall_tau", "pick_in_tie_set", "equiv_labels"):
        print(f"\nPROXIES - {col} (mean over seeds):")
        print(summ.pivot_table(index=["proxy", "n_unlabelled"], columns="corpus", values=col).round(2).to_string())


if __name__ == "__main__":
    main()
