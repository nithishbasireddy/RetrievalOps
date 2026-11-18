#!/usr/bin/env bash
# Show progress of the unattended study (safe to run any time).
cd "$(dirname "$0")/.."
for f in results/study_v1/STATUS results/study_v1/STATUS_budget results/study_v1/STATUS_v2 results/study_v1/STATUS_grid_v2 results/study_v1/STATUS_judge_grid_v2; do [ -f "$f" ] && cat "$f"; done
for f in results/study_v1/study_*.log results/study_v2/study_*.log; do
  [ -f "$f" ] && echo "$(basename $f): $(tail -c 300 "$f" | tr '\r' '\n' | grep -av '^$' | tail -1 | cut -c1-110)"
done
for s in study budget judgev2 gridv2 judgegridv2; do tmux has-session -t $s 2>/dev/null && echo "(tmux session $s is running)"; done; true
# per-shard progress of the v2 grid
for f in results/lbr_full_grid_v2/logs/shard_*.log; do
  [ -f "$f" ] && echo "$(basename $f .log): $(grep -ac 'done:' "$f") done | $(tail -c 250 "$f" | tr '\r' '\n' | grep -av '^$' | tail -1 | cut -c1-90)"
done
