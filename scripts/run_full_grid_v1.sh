#!/usr/bin/env bash
# Run the 30-config full LegalBench-RAG grid as 3 shards (one chunk size per GPU), then merge.
set -u
cd "$(dirname "$0")/.."
RUN=lbr_full_grid_v1
CFG=configs/lbr_full_grid_v1.yaml
mkdir -p results/$RUN
i=0
for size in 256 500 1000; do
  gpu=${GPUS[$i]:-$((i + 1))}
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/retrievalops run $CFG --match "chunk_size=${size}__" --run-name $RUN \
    > results/$RUN/shard_$size.log 2>&1 &
  echo "shard chunk_size=$size on GPU $gpu (pid $!)"
  i=$((i + 1))
done
wait
.venv/bin/retrievalops summarize $CFG --run-name $RUN > results/$RUN/summarize.log 2>&1
echo "GRID DONE $(date)"
