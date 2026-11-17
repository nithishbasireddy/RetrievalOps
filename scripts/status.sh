#!/usr/bin/env bash
# Show progress of the unattended study (safe to run any time).
cd "$(dirname "$0")/.."
cat results/study_v1/STATUS 2>/dev/null || echo "study not started"
for f in results/study_v1/study_*.log; do
  [ -f "$f" ] && echo "$(basename $f): $(tail -c 300 "$f" | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-110)"
done
tmux has-session -t study 2>/dev/null && echo "(tmux session 'study' is running)" || echo "(tmux session 'study' has exited)"
