#!/usr/bin/env bash
# Step 1 follow-up: judge prompt v2 (shows the document opening) on all corpora, both judges.
# Starts the judge servers itself and shuts them down at the end. Progress: results/study_v1/STATUS_v2
set -u
cd "$(dirname "$0")/.."
LOG=results/study_v1; S=$LOG/STATUS_v2
echo "judge v2 STARTED $(date +%H:%M)" > $S
while tmux has-session -t budget 2>/dev/null; do sleep 60; done   # let the budget job finish first
start() { tmux new-session -d -s "$1" "bash scripts/serve_judge.sh $2 $3 0 $4 $5 $6 2>&1 | tee /scratch/$USER/retrievalops-serve/logs/$1.log"
  for _ in $(seq 120); do curl -s 127.0.0.1:$4/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
start judge-qwen3b /scratch/$USER/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct 8010 0.25 "" || { echo "FAILED to start 3B server" >> $S; exit 1; }
start judge-qwen27b /scratch/$USER/Qwen3.5-27B Qwen/Qwen3.5-27B 8011 0.5 "--max-num-seqs 64 --limit-mm-per-prompt '{\"image\":0,\"video\":0}'" \
  || { echo "FAILED to start 27B server" >> $S; exit 1; }
COMMON="configs/lbr_full_grid_v1.yaml --run-name lbr_full_grid_v1 --budget 4000 --n-unlabelled 25 50 100 --seeds 0 1 2 --judge-prompt v2"
echo "running 3B v2 ($(date +%H:%M))" >> $S
.venv/bin/python scripts/proxy_study.py $COMMON --judge-model Qwen/Qwen2.5-3B-Instruct --judge-url http://127.0.0.1:8010/v1 \
  --judge-tag qwen2.5-3b > $LOG/v2_3b.log 2>&1 || echo "FAILED 3B v2 - see $LOG/v2_3b.log" >> $S
echo "running 27B v2 ($(date +%H:%M))" >> $S
.venv/bin/python scripts/proxy_study.py $COMMON --judge-model Qwen/Qwen3.5-27B --judge-url http://127.0.0.1:8011/v1 \
  --judge-tag qwen3.5-27b > $LOG/v2_27b.log 2>&1 || echo "FAILED 27B v2 - see $LOG/v2_27b.log" >> $S
tmux kill-session -t judge-qwen3b 2>/dev/null; tmux kill-session -t judge-qwen27b 2>/dev/null
echo "judge v2 DONE $(date +%H:%M) (servers stopped)" >> $S
