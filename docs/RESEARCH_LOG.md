# Research log: label-free RAG configuration selection

## Question

A team deploying RAG on a new corpus must choose chunking, embedder, retriever and reranker,
usually without any relevance labels. **How reliably can cheap, label-free evidence pick a
configuration that is as good as the one gold labels would pick, and how many labels is that
evidence worth?**

## Protocol (v1)

- **Corpora**: the 4 LegalBench-RAG subsets (ContractNLI, CUAD, MAUD, PrivacyQA), each treated as
  a separate deployment. All are retrieved from one shared corpus, as in the reference protocol.
- **Configuration space**: `configs/lbr_full_grid_v1.yaml`, 30 configs
  (chunk size {256, 500, 1000} × {dense E5, dense Qwen3-Emb, BM25, hybrid E5, hybrid Qwen3} × reranker on/off).
- **Oracle metric**: recall at a fixed **context budget** of 4,000 characters (`recall@4000c`), so
  configs with different chunk sizes are compared at equal LLM context length. Metrics at a fixed
  number of chunks k favour large chunks.
- **Oracle**: mean over all labelled queries of the corpus. The **tie set** is the configs whose
  paired-bootstrap 95% CI of (best − config) contains 0.
- **Selector evaluation**: regret = q(best) − q(picked); whether the pick lies in the tie set; Kendall τ
  between the selector's scores and q.
- **Label-budget baseline**: choose by the mean over n random labelled queries (1,000 repetitions).
- **Label-free proxies** use n unlabelled queries per corpus:
  - `consensus`: agreement of a config's context with the other configs' contexts (no LLM)
  - `judge_precision[J,mode]`: length-weighted LLM-judged relevance of the context
  - `judge_recall[J,mode]`: pooled recall against the union of judged-relevant text
  - modes: `graded` = (0.5·P(1) + P(2)) / Z; `strict` = P(2) / Z, where P(g) is the probability
    the judge assigns to relevance grade g (read from the logprobs of its first output token) and Z = P(0) + P(1) + P(2)
- **Judges**: Qwen2.5-3B-Instruct and Qwen3.5-27B (local vLLM, thinking disabled), prompt `v1`
  in `retrievalops/judging.py`.

## Findings so far

### 2026-11-17: baselines on LegalBench-RAG-mini (10 configs, 776 queries)
- recall@64 (macro): dense E5 55.1, hybrid E5 + bge-reranker 55.4, fixed-size E5 55.7. The top 3
  are statistically tied. BM25 is 32.3 and it drags hybrids down (hybrid E5 50.4 < dense E5).
- The best config differs by corpus (ContractNLI: BGE-M3/Qwen3; CUAD/MAUD: E5 + reranker;
  PrivacyQA: fixed chunks). The motivation holds: no single choice is right everywhere.

### 2026-11-17: pilot selection study (mini run, 10 configs, n = 50, seed 0)
- **Label budget**: with 50 labelled queries, the pick is in the tie set 85–99% of the time
  (by corpus). With 5 labels, 32–88%.
- **Consensus is a poor selector**: τ = −0.38…0.38, and its pick was outside the tie set on 3/4 corpora.
- **The judge's chunk-level accuracy does not predict its selection quality.** The 27B judge has
  higher AUC vs gold (0.89–0.99) than the 3B (0.65–0.88). Yet with `graded` scoring it selected
  *worse* (τ = −0.64 on MAUD, outside the tie set on CUAD/MAUD).
  - Cause: the 27B reliably grades same-contract passages as "1", and graded scoring turns those
    into 0.5 relevance. The proxy then rewards *document* hits rather than *answer* hits.
  - With `strict` scoring the 27B judge recovers (τ = 0.78/0.82/0.82 on CUAD/MAUD/PrivacyQA, pick
    in the tie set on all 4 corpora), close to using gold labels for the same 50 queries.
- ContractNLI: 7/10 configs are tied, so its oracle ranking is mostly noise. τ there says little.

Caveat: pilot scale (10 configs, one seed). Must be re-tested on the full 30-config grid, with
several seeds and n ∈ {25, 50, 100}.

### 2026-11-17: full study v1 (30 configs, full LegalBench-RAG, recall@4000c, n ∈ {25,50,100}, 3 seeds)
- **Strict-scored LLM judge works on 3/4 corpora**: Kendall τ 0.79–0.90 on CUAD/PrivacyQA and
  0.46–0.76 on MAUD. Its pick is worth ~100–250 labelled queries.
- **It fails on ContractNLI (τ ≈ 0).** Diagnosis:
  - 97% of strict-judge positives come from the wrong NDA.
  - In 34% of ContractNLI queries (39% CUAD, 47% PrivacyQA) the party names in the query appear
    nowhere in the document text, so the judge cannot tell same-template NDAs apart.
  - The oracle says reranking hurts on ContractNLI; the judge says it helps.
- 38–60% of judge positives lie in the right document but outside gold. This points to incomplete
  gold spans; a manual audit is planned.
- Consensus is useless. Synthetic questions: style-matched beats naive, both weaker than the judges.
- PPI++ gives no gain over labels alone; a judge shortlist (top-3/5) helps only for very small n.
- With 100 labels, P(pick in tie set) is only ~0.65 on ContractNLI and CUAD.
- **Weak point**: 1000-char chunks win everywhere, and the global-best config has ~0 regret except
  on ContractNLI (0.083), so the per-corpus selection motivation was weak on this grid.

### 2026-11-17: judge prompt v2 (document opening shown) and budget robustness
- v2 raises chunk-level AUC (27B ContractNLI 0.86 → 0.93; 3B MAUD 0.76 → 0.92), but selection
  barely moves (ContractNLI τ −0.05 → 0.12), and 3B on MAUD *drops* (0.72 → −0.08).
  More evidence that chunk accuracy ≠ selection quality.
- Conclusions are stable at 2000 / 4000 / 8000-char budgets (judges not run at 8000).

### 2026-11-18: wider grid v2 (70 configs: + 128/2000-char chunks, + MiniLM embedder)
Oracle at recall@4000c (all 6,889 queries):

| corpus | oracle best | #tied | spread best–worst | regret of one global config |
|---|---|---|---|---|
| ContractNLI | 2000 / hybrid / Qwen3 / no rerank (0.251) | 1 | 0.23 | **0.114 (−45% relative)** |
| CUAD | 1000 / dense / Qwen3 / rerank (0.381) | 2 | 0.33 | 0.027 (−7%) |
| MAUD | 2000 / dense / Qwen3 / rerank (0.040) | 15 | 0.04 | 0.002 (tied) |
| PrivacyQA | 2000 / dense / E5 / no rerank (0.507) | 14 | 0.45 | 0.019 (tied) |

- **Per-corpus selection now matters**: the best macro config (2000/dense/E5/rerank) loses 45% of
  ContractNLI recall and 7% on CUAD. The four corpora pick four different configs, which differ
  in chunk size, embedder *and* whether to rerank.
- The global choice also flips with the budget (2000c: 1000-char chunks; 4000c: 2000-char chunks).
- Labels: ~10 labels already beat the global config on ContractNLI (regret 0.064 < 0.114);
  CUAD needs ~30. With 100 labels P(pick in tie set) is 0.66 / 0.44 / 0.80 / 1.00.
- Consensus remains useless (pick never in the tie set; τ ≈ 0.1 on ContractNLI/MAUD).
- Next: judges + synthetic questions on this grid (`scripts/run_judge_grid_v2.sh`).

## Next
1. Full grid finishes → oracle on full LegalBench-RAG; repeat the proxy study (2 judges × 2 modes,
   seeds 0–2, n ∈ {25, 50, 100}).
2. Express each proxy's result as an **equivalent number of labels** (the n at which the
   label-budget curve reaches the same regret).
3. Hybrid selector: few labels + many judged unlabelled queries (prediction-powered estimate).
4. Synthetic-question proxy (RAGAS/ARES style) and its lexical bias toward BM25.
5. Add non-legal corpora to test whether the findings generalise.
