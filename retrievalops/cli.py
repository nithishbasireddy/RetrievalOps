"""Command line entry point.

    retrievalops run configs/lbr_mini_baselines.yaml [--only NAME ...] [--max-queries N] [--run-name NAME]
    retrievalops list configs/lbr_mini_baselines.yaml
"""

from __future__ import annotations

import argparse
import logging

from retrievalops.config import load_experiment


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="retrievalops")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment file")
    r.add_argument("experiment")
    r.add_argument("--only", nargs="+", help="run only these config names")
    r.add_argument("--max-queries", type=int, help="truncate the query set (smoke tests)")
    r.add_argument("--run-name", help="output sub-folder name; re-use it to resume a sweep")
    r.add_argument("--device")
    ls = sub.add_parser("list", help="list the configs an experiment file expands to")
    ls.add_argument("experiment")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    exp = load_experiment(args.experiment)

    if args.cmd == "list":
        for c in exp.configs:
            print(c.name)
        print(f"{len(exp.configs)} configs")
        return

    if args.device:
        exp.device = args.device
    from retrievalops.runner import Runner

    runner = Runner(exp, run_name=args.run_name, max_queries=args.max_queries)
    summary = runner.run(only=args.only)
    print(f"\nResults: {runner.run_dir}")
    if not summary.empty:
        cols = [c for c in ("config", "subset", "n", "precision@1", "recall@1", "recall@5", "recall@10",
                            "ndcg@10", "search_ms_per_query") if c in summary.columns]
        print(summary[summary.subset.isin(["ALL_macro", "ALL_micro"])][cols].to_string(index=False))


if __name__ == "__main__":
    main()
