#!/usr/bin/env bash
# Main study on the 70-config grid v2: synthetic-question runs, then both LLM judges (prompt v1,
# graded + strict scoring), consensus and gold-label baselines, at a 4000-char budget.
# GPU split: the 3B judge on GPU 0; the 27B judge replicated over GPUs 1-3 (vLLM data parallel).
# Runs unattended and stops the judge servers at the end. Progress: bash scripts/status.sh
set -u
cd "$(dirname "$0")/.."
S=results/study_v1/STATUS_judge_grid_v2; L=results/study_v2; mkdir -p $L
echo "judge study v2 STARTED $(date +%H:%M)" > $S
declare -A E=([e5]=e5-large-v2 [qwen3]=Qwen3-Embedding-0.6B [minilm]=all-MiniLM-L6-v2)
SHARDS="128:qwen3 128:e5 2000:qwen3 2000:e5 128:minilm 2000:minilm 128:bm25 2000:bm25 256:minilm 500:minilm 1000:minilm"

# ---- 1. grid v2 on the two synthetic question sets (the v1 configs are reused) --------------------
for st in naive style_matched; do
  RUN=lbr_synth_${st}_v2; CFG=configs/$RUN.yaml; D=results/$RUN
  mkdir -p $D/per_query $D/logs
  cp -ln results/lbr_synth_${st}_v1/per_query/* $D/per_query/ 2>/dev/null
  echo "synthetic $st: running ($(date +%H:%M))" >> $S
  i=0; pids=()
  for sh in $SHARDS; do
    c=${sh%%:*}; e=${sh##*:}
    if [ "$e" = bm25 ]; then rx="^chunk_size=${c}__retriever=bm25__"
    else rx="^chunk_size=${c}__retriever=(dense|hybrid)__embedding_model=${E[$e]}__"; fi
    CUDA_VISIBLE_DEVICES=$((i % 4)) .venv/bin/retrievalops run $CFG --match "$rx" --run-name $RUN \
      > $D/logs/shard_${c}_${e}.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  fails=0; for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
  n=$(ls $D/per_query/*.jsonl | wc -l)
  [ "$fails" -gt 0 ] || [ "$n" -ne 70 ] && { echo "FAILED synthetic $st ($n/70 configs) - see $D/logs/" >> $S; exit 1; }
  .venv/bin/retrievalops summarize $CFG --run-name $RUN > $D/logs/summarize.log 2>&1
  echo "synthetic $st: done 70/70 ($(date +%H:%M))" >> $S
done

# no-judge pass first: builds the synthetic metric caches (so the two judge runs below don't race to
# write them) and the consensus / synthetic / gold-label proxies
COMMON="configs/lbr_full_grid_v2.yaml --run-name lbr_full_grid_v2 --budget 4000 --n-unlabelled 25 50 100 --seeds 0 1 2"
SYN="--synthetic naive=configs/lbr_synth_naive_v2.yaml:lbr_synth_naive_v2 style_matched=configs/lbr_synth_style_matched_v2.yaml:lbr_synth_style_matched_v2"
echo "synthetic + consensus proxies ($(date +%H:%M))" >> $S
.venv/bin/python scripts/proxy_study.py $COMMON $SYN > $L/study_nojudge.log 2>&1 \
  || { echo "FAILED no-judge pass - see $L/study_nojudge.log" >> $S; exit 1; }

# ---- 2. judge servers ---------------------------------------------------------------------------
mkdir -p /scratch/$USER/retrievalops-serve/logs
start() {  # session model_dir name gpus port mem extra
  tmux new-session -d -s "$1" "bash scripts/serve_judge.sh $2 $3 $4 $5 $6 $7 2>&1 | tee /scratch/$USER/retrievalops-serve/logs/$1.log"
  for _ in $(seq 120); do curl -s 127.0.0.1:$5/v1/models >/dev/null && return 0
    tmux has-session -t "$1" 2>/dev/null || return 1; sleep 10; done; return 1; }
MM="--limit-mm-per-prompt '{\"image\":0,\"video\":0}'"
start judge-qwen3b /scratch/$USER/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct 0 8010 0.25 "" \
  || { echo "FAILED to start 3B server" >> $S; exit 1; }
CONC27=192
if start judge-qwen27b /scratch/$USER/Qwen3.5-27B Qwen/Qwen3.5-27B 1,2,3 8011 0.5 "--data-parallel-size 3 --max-num-seqs 64 $MM"; then
  echo "27B server: data parallel on GPUs 1-3 ($(date +%H:%M))" >> $S
else  # fall back to one GPU
  tmux kill-session -t judge-qwen27b 2>/dev/null; sleep 20; CONC27=64
  start judge-qwen27b /scratch/$USER/Qwen3.5-27B Qwen/Qwen3.5-27B 1 8011 0.5 "--max-num-seqs 64 $MM" \
    || { echo "FAILED to start 27B server" >> $S; tmux kill-session -t judge-qwen3b; exit 1; }
  echo "27B server: data parallel failed, using GPU 1 only ($(date +%H:%M))" >> $S
fi

# ---- 3. both judges at once (judgment cache is WAL sqlite; output merges are file-locked) --------
echo "judging with 3B and 27B in parallel ($(date +%H:%M))" >> $S
.venv/bin/python scripts/proxy_study.py $COMMON $SYN --judge-model Qwen/Qwen2.5-3B-Instruct \
  --judge-url http://127.0.0.1:8010/v1 --judge-tag qwen2.5-3b > $L/study_3b.log 2>&1 &
p3=$!
.venv/bin/python scripts/proxy_study.py $COMMON $SYN --judge-model Qwen/Qwen3.5-27B --judge-concurrency $CONC27 \
  --judge-url http://127.0.0.1:8011/v1 --judge-tag qwen3.5-27b > $L/study_27b.log 2>&1 &
p27=$!
wait $p3 && echo "3B study done ($(date +%H:%M))" >> $S || echo "FAILED 3B study - see $L/study_3b.log" >> $S
tmux kill-session -t judge-qwen3b 2>/dev/null
wait $p27 && echo "27B study done ($(date +%H:%M))" >> $S || echo "FAILED 27B study - see $L/study_27b.log" >> $S
tmux kill-session -t judge-qwen27b 2>/dev/null
echo "judge study v2 DONE $(date +%H:%M) (servers stopped)" >> $S
