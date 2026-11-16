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

## Next
1. Full grid finishes → oracle on full LegalBench-RAG; repeat the proxy study (2 judges × 2 modes,
   seeds 0–2, n ∈ {25, 50, 100}).
2. Express each proxy's result as an **equivalent number of labels** (the n at which the
   label-budget curve reaches the same regret).
3. Hybrid selector: few labels + many judged unlabelled queries (prediction-powered estimate).
4. Synthetic-question proxy (RAGAS/ARES style) and its lexical bias toward BM25.
5. Add non-legal corpora to test whether the findings generalise.
