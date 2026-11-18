#!/usr/bin/env bash
# Show progress of the unattended study (safe to run any time).
cd "$(dirname "$0")/.."
for f in results/study_v1/STATUS results/study_v1/STATUS_budget results/study_v1/STATUS_v2; do [ -f "$f" ] && cat "$f"; done
for f in results/study_v1/study_*.log; do
  [ -f "$f" ] && echo "$(basename $f): $(tail -c 300 "$f" | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-110)"
done
for s in study budget judgev2; do tmux has-session -t $s 2>/dev/null && echo "(tmux session $s is running)"; done; true
