from types import SimpleNamespace

import numpy as np
import pytest

from retrievalops.analysis import proxies as P
from retrievalops.judging import Judgment, JudgmentCache, parse_digit_logprobs
from retrievalops.types import Query, Span


def _alt(tok, p):
    return SimpleNamespace(token=tok, logprob=float(np.log(p)))


def test_parse_digit_logprobs_and_relevance():
    j = parse_digit_logprobs([_alt("2", 0.6), _alt(" 1", 0.3), _alt("0", 0.05), _alt("The", 0.05)])
    assert (j.p0, j.p1, j.p2) == pytest.approx((0.05, 0.3, 0.6))
    assert j.label == 2
    assert j.relevance == pytest.approx((0.15 + 0.6) / 0.95)
    assert Judgment(0, 0, 0).relevance == 0.0


def test_judgment_cache_roundtrip(tmp_path):
    c = JudgmentCache(tmp_path / "j.sqlite")
    c.put("k1", Judgment(0.1, 0.2, 0.7))
    c.commit()
    c2 = JudgmentCache(tmp_path / "j.sqlite")
    assert c2.get_many(["k1", "missing"]) == {"k1": Judgment(0.1, 0.2, 0.7)}


def test_consensus_rewards_agreement():
    ctx = {
        "a": {"q": [Span("d", 0, 100)]},
        "b": {"q": [Span("d", 0, 100)]},
        "c": {"q": [Span("d", 500, 600)]},
    }
    s = P.consensus_scores(ctx, ["q"])
    assert s["a"][0] == pytest.approx(0.5)  # 1 of 2 others agree on every char
    assert s["c"][0] == 0.0


def test_judged_scores_precision_and_pooled_recall():
    rel_chunk, irr_chunk, rel2 = Span("d", 0, 100), Span("d", 100, 200), Span("d", 300, 400)
    full = {"good": {"q": [rel_chunk, rel2]}, "bad": {"q": [irr_chunk, rel_chunk]}}
    trunc = {"good": {"q": [rel_chunk, Span("d", 300, 350)]}, "bad": {"q": [irr_chunk, Span("d", 0, 50)]}}
    rel = {("q", rel_chunk): 1.0, ("q", rel2): 1.0, ("q", irr_chunk): 0.0}
    prec, rec = P.judged_scores(trunc, full, ["q"], rel)
    assert prec["good"][0] == pytest.approx(1.0)
    assert prec["bad"][0] == pytest.approx(50 / 150)
    assert rec["good"][0] == pytest.approx(150 / 200)  # pooled relevant text = 200 chars
    assert rec["bad"][0] == pytest.approx(50 / 200)


def test_judge_vs_gold_perfect_judge():
    q = Query("q", "x", gold_spans=[Span("d", 0, 100)])
    rel = {("q", Span("d", 0, 50)): 0.9, ("q", Span("d", 200, 300)): 0.1, ("q", Span("e", 0, 10)): 0.2}
    v = P.judge_vs_gold(rel, {"q": q})
    assert v["auc"] == 1.0 and v["kappa@0.5"] == 1.0


def test_strict_vs_graded_relevance():
    same_doc_not_answer = Judgment(0.0, 1.0, 0.0)
    assert same_doc_not_answer.score("graded") == pytest.approx(0.5)
    assert same_doc_not_answer.score("strict") == 0.0
    with pytest.raises(ValueError):
        same_doc_not_answer.score("other")
