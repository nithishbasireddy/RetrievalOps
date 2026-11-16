"""Warm the judgment cache for the query samples proxy_study.py will use (same seeding),
using whichever configs of a run have finished so far. Safe to run while a grid is in progress."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrievalops.analysis import proxies as P
from retrievalops.config import load_experiment
from retrievalops.datasets import load_dataset
from retrievalops.judging import LLMJudge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-url", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    exp = load_experiment(args.experiment)
    run_dir = Path(exp.output_dir) / args.run_name
    ds = load_dataset(exp.dataset)
    queries = {q.qid: q for q in ds.queries}
    docs = {d.doc_id: d for d in ds.documents}
    judge = LLMJudge(args.judge_model, base_url=args.judge_url, cache_path=Path(exp.cache_dir) / "judgments.sqlite")
    for seed in args.seeds:
        for corpus in sorted({q.subset for q in ds.queries}):
            pool_q = sorted(q.qid for q in ds.queries if q.subset == corpus)
            order = random.Random(f"{seed}/{corpus}").sample(pool_q, min(args.n, len(pool_q)))
            _, full = P.load_contexts(run_dir, set(order), args.budget)
            P.run_judge(judge, P.judge_pool(full, order), queries, docs, desc=f"prejudge {corpus} s{seed}")


if __name__ == "__main__":
    main()
