#!/usr/bin/env bash
# Wider grid v2: 70 configs = the 30 v1 results (hard-linked, not recomputed) + 40 new ones,
# run as 11 shards (chunk size x embedder) spread over 4 GPUs, then the label-free selection analysis
# (oracle, label curve, consensus; no judge). Progress: bash scripts/status.sh
set -u
cd "$(dirname "$0")/.."
RUN=lbr_full_grid_v2; CFG=configs/$RUN.yaml; D=results/$RUN; S=results/study_v1/STATUS_grid_v2
mkdir -p $D/per_query $D/logs
echo "grid v2 STARTED $(date +%H:%M)" > $S
cp -ln results/lbr_full_grid_v1/per_query/* $D/per_query/ 2>/dev/null
declare -A E=([e5]=e5-large-v2 [qwen3]=Qwen3-Embedding-0.6B [minilm]=all-MiniLM-L6-v2)
# heaviest first, round-robin over GPUs so each GPU gets one heavy shard
SHARDS="128:qwen3 128:e5 2000:qwen3 2000:e5 128:minilm 2000:minilm 128:bm25 2000:bm25 256:minilm 500:minilm 1000:minilm"
GPUS=(${GPUS:-0 1 2 3}); i=0; pids=()
for sh in $SHARDS; do
  c=${sh%%:*}; e=${sh##*:}
  if [ "$e" = bm25 ]; then rx="^chunk_size=${c}__retriever=bm25__"
  else rx="^chunk_size=${c}__retriever=(dense|hybrid)__embedding_model=${E[$e]}__"; fi
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g .venv/bin/retrievalops run $CFG --match "$rx" --run-name $RUN > $D/logs/shard_${c}_${e}.log 2>&1 &
  pids+=($!); echo "  shard ${c}/${e} -> GPU $g" >> $S
  i=$((i + 1))
done
fails=0
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
n=$(ls $D/per_query/*.jsonl | wc -l)
echo "grid runs finished $(date +%H:%M): $n/70 configs, $fails shard(s) failed" >> $S
[ "$fails" -gt 0 ] && { echo "FAILED: see $D/logs/" >> $S; exit 1; }
.venv/bin/retrievalops summarize $CFG --run-name $RUN > $D/logs/summarize.log 2>&1
echo "running selection analysis ($(date +%H:%M))" >> $S
for b in 4000 2000; do
  .venv/bin/python scripts/proxy_study.py $CFG --run-name $RUN --budget $b --n-unlabelled 25 50 100 --seeds 0 1 2 \
    > $D/logs/analysis_${b}c.log 2>&1 || { echo "FAILED analysis ${b}c" >> $S; exit 1; }
done
echo "grid v2 DONE $(date +%H:%M)" >> $S
