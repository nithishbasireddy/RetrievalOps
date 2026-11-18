#!/usr/bin/env bash
# Step 2: does the study hold at 2000 and 8000 characters of context? (no GPU needed)
# 2000c: judge proxies come entirely from the judgment cache (the 4000c pool contains them).
# 8000c: judge proxies would need new judgments, so only oracle / labels / consensus / synthetic.
set -u
cd "$(dirname "$0")/.."
LOG=results/study_v1; mkdir -p $LOG
COMMON="configs/lbr_full_grid_v1.yaml --run-name lbr_full_grid_v1 --n-unlabelled 25 50 100 --seeds 0 1 2"
SYN="--synthetic naive=configs/lbr_synth_naive_v1.yaml:lbr_synth_naive_v1 style_matched=configs/lbr_synth_style_matched_v1.yaml:lbr_synth_style_matched_v1"
echo "budget robustness STARTED $(date +%H:%M)" > $LOG/STATUS_budget
for J in "Qwen/Qwen2.5-3B-Instruct qwen2.5-3b" "Qwen/Qwen3.5-27B qwen3.5-27b"; do set -- $J
  .venv/bin/python scripts/proxy_study.py $COMMON --budget 2000 $SYN --judge-model $1 --judge-url http://127.0.0.1:9/v1 --judge-tag $2 \
    > $LOG/budget2000_$2.log 2>&1 || { echo "FAILED 2000c $2" >> $LOG/STATUS_budget; exit 1; }
done
.venv/bin/python scripts/proxy_study.py $COMMON --budget 8000 $SYN > $LOG/budget8000.log 2>&1 \
  || { echo "FAILED 8000c" >> $LOG/STATUS_budget; exit 1; }
echo "budget robustness DONE $(date +%H:%M)" >> $LOG/STATUS_budget
