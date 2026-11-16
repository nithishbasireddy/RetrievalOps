"""Configuration-selection analysis.

Terminology
-----------
For one corpus (e.g. a LegalBench-RAG subset), ``M`` is a [n_configs x n_queries] matrix of a
per-query quality metric computed with gold labels (default: recall at a fixed context budget).

* true quality      q_c = mean_j M[c, j]; the oracle choice is argmax_c q_c.
* tie set           configs whose paired-bootstrap 95% CI of (q_best - q_c) contains 0, i.e. not
                    distinguishable from the oracle on the full label set.
* regret(c)         q_best - q_c   (absolute, in metric units).
* selector          any rule mapping cheap evidence to a config; scored by regret, Kendall tau
                    between its scores and q, and whether its pick lies in the tie set.

The "n labels" selector picks argmax of the mean over n randomly sampled labelled queries; its
expected regret as a function of n is the label-budget curve every label-free proxy must beat.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from retrievalops.metrics import budget_metrics
from retrievalops.types import Dataset, Span

DEFAULT_BUDGETS = (2000, 4000, 8000)


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_per_query(run_dir: str | Path, dataset: Dataset, budgets: Sequence[int] = DEFAULT_BUDGETS) -> pd.DataFrame:
    """Long table: one row per (config, query) with stored @k metrics plus budget metrics."""
    gold = {q.qid: q.gold_spans for q in dataset.queries}
    rows = []
    for path in sorted((Path(run_dir) / "per_query").glob("*.jsonl")):
        config = path.stem
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r["qid"] not in gold or not gold[r["qid"]]:
                    continue
                spans = [Span(x["doc_id"], x["start"], x["end"]) for x in r["retrieved"]]
                rows.append({
                    "config": config,
                    "qid": r["qid"],
                    "subset": r.get("subset") or "all",
                    **r.get("metrics", {}),
                    **budget_metrics(spans, gold[r["qid"]], budgets),
                })
    if not rows:
        raise ValueError(f"No per-query results with gold labels under {run_dir}")
    return pd.DataFrame(rows)


def matrix(df: pd.DataFrame, metric: str, subset: str | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """[n_configs x n_queries] matrix; only queries answered by every config are kept."""
    d = df if subset is None else df[df.subset == subset]
    wide = d.pivot_table(index="config", columns="qid", values=metric, aggfunc="first").dropna(axis=1)
    return wide.to_numpy(dtype=float), list(wide.index), list(wide.columns)


# ---------------------------------------------------------------------------------------------
# Oracle ranking and tie set
# ---------------------------------------------------------------------------------------------
def paired_bootstrap_tie_set(M: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> np.ndarray:
    """Boolean mask of configs statistically indistinguishable from the best (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    q = M.mean(axis=1)
    best = int(np.argmax(q))
    n = M.shape[1]
    D = M[best][None, :] - M  # [n_configs, n_queries] paired differences
    diffs = np.empty((M.shape[0], n_boot))
    for b0 in range(0, n_boot, 100):  # batched to bound memory
        idx = rng.integers(0, n, size=(min(100, n_boot - b0), n))
        diffs[:, b0 : b0 + idx.shape[0]] = D[:, idx].mean(axis=2)
    lo = np.quantile(diffs, alpha / 2, axis=1)
    tie = lo <= 0
    tie[best] = True
    return tie


@dataclass
class OracleRanking:
    configs: list[str]
    quality: np.ndarray
    tie: np.ndarray

    @property
    def best(self) -> int:
        return int(np.argmax(self.quality))

    def regret(self, chosen: int) -> float:
        return float(self.quality[self.best] - self.quality[chosen])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"config": self.configs, "quality": self.quality, "tied_with_best": self.tie})
        df["regret"] = self.quality.max() - df["quality"]
        return df.sort_values("quality", ascending=False).reset_index(drop=True)


def oracle(M: np.ndarray, configs: list[str], seed: int = 0) -> OracleRanking:
    return OracleRanking(configs, M.mean(axis=1), paired_bootstrap_tie_set(M, seed=seed))


# ---------------------------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------------------------
def evaluate_scores(scores: np.ndarray, orc: OracleRanking) -> dict[str, float]:
    """How well per-config proxy scores (higher = better) recover the oracle."""
    chosen = int(np.argmax(scores))
    tau = kendalltau(scores, orc.quality).statistic
    rho = spearmanr(scores, orc.quality).statistic
    return {
        "kendall_tau": float(tau),
        "spearman": float(rho),
        "regret": orc.regret(chosen),
        "rel_regret": orc.regret(chosen) / orc.quality.max() if orc.quality.max() > 0 else 0.0,
        "pick_in_tie_set": float(orc.tie[chosen]),
        "picked": orc.configs[chosen],
    }


def label_budget_curve(
    M: np.ndarray,
    orc: OracleRanking,
    budgets: Sequence[int] = (5, 10, 20, 50, 100, 200, 500),
    reps: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Expected regret of choosing by the mean over n labelled queries, for each n."""
    rng = np.random.default_rng(seed)
    n_q = M.shape[1]
    rows = []
    for n in budgets:
        if n > n_q:
            continue
        regrets, ties, taus = [], [], []
        for _ in range(reps):
            idx = rng.choice(n_q, size=n, replace=False)
            s = M[:, idx].mean(axis=1)
            # break exact ties at random rather than by config order
            chosen = int(rng.choice(np.flatnonzero(s == s.max())))
            regrets.append(orc.regret(chosen))
            ties.append(orc.tie[chosen])
            taus.append(kendalltau(s, orc.quality).statistic)
        regrets = np.asarray(regrets)
        rows.append({
            "n_labels": n,
            "mean_regret": regrets.mean(),
            "p90_regret": np.quantile(regrets, 0.9),
            "p_pick_in_tie_set": float(np.mean(ties)),
            "mean_kendall_tau": float(np.nanmean(taus)),
        })
    return pd.DataFrame(rows)


def fixed_choice_baselines(orcs: dict[str, OracleRanking]) -> pd.DataFrame:
    """Regret of 'one config for every corpus' choices: the global-best config and each config.

    The global choice is the config with the best macro-average quality across corpora; it shows
    the cost of not selecting per corpus at all.
    """
    configs = next(iter(orcs.values())).configs
    for o in orcs.values():
        if o.configs != configs:
            raise ValueError("corpora must share the same config list")
    macro = np.mean([o.quality for o in orcs.values()], axis=0)
    g = int(np.argmax(macro))
    rows = []
    for name, o in orcs.items():
        rows.append({"corpus": name, "global_best": configs[g], "regret_global_best": o.regret(g),
                     "oracle": configs[o.best], "n_tied": int(o.tie.sum())})
    return pd.DataFrame(rows)
