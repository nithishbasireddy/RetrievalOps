"""Command line entry point.

    retrievalops run configs/lbr_mini_baselines.yaml [--only NAME ...] [--max-queries N] [--run-name NAME]
    retrievalops run configs/lbr_full_grid_v1.yaml --match "chunk_size=256__" --run-name grid  # one shard
    retrievalops summarize configs/lbr_full_grid_v1.yaml --run-name grid   # merge shards
    retrievalops list configs/lbr_mini_baselines.yaml
"""

from __future__ import annotations

import argparse
import logging
import re

from retrievalops.config import load_experiment


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="retrievalops")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment file")
    r.add_argument("experiment")
    r.add_argument("--only", nargs="+", help="run only these config names")
    r.add_argument("--match", help="run only configs whose name matches this regex (for sharding)")
    r.add_argument("--max-queries", type=int, help="truncate the query set (smoke tests)")
    r.add_argument("--run-name", help="output sub-folder name; re-use it to resume a sweep")
    r.add_argument("--device")
    ls = sub.add_parser("list", help="list the configs an experiment file expands to")
    ls.add_argument("experiment")
    ls.add_argument("--match")
    sm = sub.add_parser("summarize", help="(re)build summary.csv of an existing run")
    sm.add_argument("experiment")
    sm.add_argument("--run-name", required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    exp = load_experiment(args.experiment)

    selected = [c.name for c in exp.configs]
    if getattr(args, "match", None):
        selected = [n for n in selected if re.search(args.match, n)]
        if not selected:
            raise SystemExit(f"--match {args.match!r} selects no configs")

    if args.cmd == "list":
        for name in selected:
            print(name)
        print(f"{len(selected)} configs")
        return

    if args.device:
        exp.device = args.device
    from retrievalops.runner import Runner

    if args.cmd == "summarize":
        runner = Runner(exp, run_name=args.run_name)
        summary = runner.summarize()
    else:
        only = args.only
        if args.match:
            only = [n for n in selected if only is None or n in only]
        runner = Runner(exp, run_name=args.run_name, max_queries=args.max_queries)
        summary = runner.run(only=only)
    print(f"\nResults: {runner.run_dir}")
    if not summary.empty:
        cols = [c for c in ("config", "subset", "n", "precision@1", "recall@1", "recall@5", "recall@10",
                            "ndcg@10", "search_ms_per_query") if c in summary.columns]
        print(summary[summary.subset.isin(["ALL_macro", "ALL_micro"])][cols].to_string(index=False))


if __name__ == "__main__":
    main()
