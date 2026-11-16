#!/usr/bin/env bash
# After the main grid: run the same 30 configs on both synthetic question sets (3 GPUs, by chunk size).
set -u
cd "$(dirname "$0")/.."
while tmux has-session -t retrievalops-grid 2>/dev/null; do sleep 60; done
for st in naive style_matched; do
  RUN=lbr_synth_${st}_v1; CFG=configs/$RUN.yaml
  mkdir -p results/$RUN
  i=0
  for size in 256 500 1000; do
    CUDA_VISIBLE_DEVICES=$((i + 1)) .venv/bin/retrievalops run $CFG --match "chunk_size=${size}__" --run-name $RUN \
      > results/$RUN/shard_$size.log 2>&1 &
    i=$((i + 1))
  done
  wait
  .venv/bin/retrievalops summarize $CFG --run-name $RUN > results/$RUN/summarize.log 2>&1
  echo "$RUN DONE $(date)"
done
