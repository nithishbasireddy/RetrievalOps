import numpy as np
import pytest

from retrievalops.analysis.selection import (
    evaluate_scores,
    fixed_choice_baselines,
    label_budget_curve,
    oracle,
)
from retrievalops.metrics import budget_metrics, truncate_to_budget
from retrievalops.types import Span


def test_truncate_to_budget_cuts_last_chunk():
    spans = [Span("a", 0, 300), Span("a", 1000, 1300), Span("b", 0, 300)]
    out = truncate_to_budget(spans, 450)
    assert out == [Span("a", 0, 300), Span("a", 1000, 1150)]
    assert truncate_to_budget(spans, 10_000) == spans


def test_budget_metrics_equal_context_comparison():
    gold = [Span("a", 0, 1000)]
    small_chunks = [Span("a", i, i + 250) for i in range(0, 1000, 250)]
    big_chunks = [Span("a", 0, 1000)]
    m_small = budget_metrics(small_chunks, gold, [500])
    m_big = budget_metrics(big_chunks, gold, [500])
    assert m_small["recall@500c"] == pytest.approx(0.5)
    assert m_big["recall@500c"] == pytest.approx(0.5)
    assert m_small["used_chars@500c"] == 500


def _synthetic(seed=0, n_q=400):
    rng = np.random.default_rng(seed)
    true = np.array([0.50, 0.49, 0.40, 0.20])
    M = np.clip(true[:, None] + rng.normal(0, 0.25, size=(4, n_q)), 0, 1)
    return M, ["a", "b", "c", "d"]


def test_oracle_and_tie_set():
    M, names = _synthetic()
    o = oracle(M, names)
    assert names[o.best] in ("a", "b")
    assert o.tie[o.best]
    assert o.tie[names.index("a")] and o.tie[names.index("b")]  # 0.01 apart: not separable
    assert not o.tie[names.index("d")]
    assert o.to_frame().iloc[0]["regret"] == 0


def test_evaluate_scores_perfect_and_inverted():
    M, names = _synthetic()
    o = oracle(M, names)
    perfect = evaluate_scores(o.quality, o)
    assert perfect["kendall_tau"] == pytest.approx(1.0)
    assert perfect["regret"] == 0 and perfect["pick_in_tie_set"] == 1
    inverted = evaluate_scores(-o.quality, o)
    assert inverted["kendall_tau"] == pytest.approx(-1.0)
    assert inverted["picked"] == "d"


def test_label_budget_curve_regret_decreases():
    M, names = _synthetic()
    o = oracle(M, names)
    curve = label_budget_curve(M, o, budgets=(2, 20, 200), reps=300)
    r = curve.mean_regret.to_numpy()
    assert r[0] > r[2]
    assert (curve.p_pick_in_tie_set.diff().dropna() >= -0.05).all()


def test_fixed_choice_baselines():
    M1, names = _synthetic(0)
    M2 = M1[::-1].copy()  # opposite ordering on a second corpus
    df = fixed_choice_baselines({"x": oracle(M1, names), "y": oracle(M2, names)})
    assert set(df.corpus) == {"x", "y"}
    assert (df.regret_global_best > 0).any()


def test_equivalent_labels_interpolates():
    import pandas as pd

    from retrievalops.analysis.selection import equivalent_labels

    curve = pd.DataFrame({"n_labels": [1, 10, 100], "mean_regret": [0.10, 0.05, 0.01]})
    assert equivalent_labels(curve, 0.2) == 1
    assert equivalent_labels(curve, 0.05) == 10
    assert equivalent_labels(curve, 0.0) == float("inf")
    mid = equivalent_labels(curve, 0.03)
    assert 10 < mid < 100
    assert mid == pytest.approx(10 ** 1.5)  # halfway in regret -> geometric midpoint in n


def test_ppi_reduces_to_label_mean_without_judge_signal_and_uses_good_judge():
    from retrievalops.analysis.selection import assisted_selection_curve, ppi_scores

    rng = np.random.default_rng(0)
    G = rng.random((3, 10))
    J_const = np.ones((3, 40))
    assert np.allclose(ppi_scores(G, J_const[:, :10], J_const), G.mean(axis=1))

    # a judge that is a noisy-but-unbiased copy of gold should cut regret at small n
    true = np.array([0.50, 0.45, 0.30, 0.20])
    G = np.clip(true[:, None] + rng.normal(0, 0.3, size=(4, 400)), 0, 1)
    J = G + rng.normal(0, 0.05, size=G.shape)
    o = oracle(G, list("abcd"))
    curve = assisted_selection_curve(G, J, o, budgets=(5,), reps=300)
    r = curve.set_index("selector").mean_regret
    assert r["ppi"] < r["labels_only"]
