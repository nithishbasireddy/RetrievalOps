#!/usr/bin/env bash
# Full label-free selection study on the 30-config grid. Runs unattended; steps run one after
# another (one process uses the judgment cache at a time). Progress: bash scripts/status.sh
set -u
cd "$(dirname "$0")/.."
LOG=results/study_v1
mkdir -p $LOG
echo "STARTED $(date)" > $LOG/STATUS
step() { echo "RUNNING: $1 ($(date +%H:%M))" >> $LOG/STATUS; }
fail() { echo "FAILED: $1 ($(date +%H:%M)) - see $LOG/$2" >> $LOG/STATUS; exit 1; }

COMMON="configs/lbr_full_grid_v1.yaml --run-name lbr_full_grid_v1 --budget 4000 --n-unlabelled 25 50 100 --seeds 0 1 2"
SYN="--synthetic naive=configs/lbr_synth_naive_v1.yaml:lbr_synth_naive_v1 style_matched=configs/lbr_synth_style_matched_v1.yaml:lbr_synth_style_matched_v1"

step "1/3 study with Qwen2.5-3B judge (+ synthetic + consensus)"
.venv/bin/python scripts/proxy_study.py $COMMON $SYN \
  --judge-model Qwen/Qwen2.5-3B-Instruct --judge-url http://127.0.0.1:8010/v1 --judge-tag qwen2.5-3b \
  > $LOG/study_3b.log 2>&1 || fail "3B study" study_3b.log

step "2/3 study with Qwen3.5-27B judge (slowest step, ~1-2 h)"
.venv/bin/python scripts/proxy_study.py $COMMON $SYN \
  --judge-model Qwen/Qwen3.5-27B --judge-url http://127.0.0.1:8011/v1 --judge-tag qwen3.5-27b \
  > $LOG/study_27b.log 2>&1 || fail "27B study" study_27b.log

step "3/3 writing report"
{
  echo "=== STUDY v1 REPORT ($(date)) ==="
  sed -n '/ORACLE best/,$p' $LOG/study_27b.log | grep -v -E "it/s|INFO"
  echo; echo "=== 3B judge-assisted selection ==="
  sed -n '/JUDGE-ASSISTED/,/JUDGE VALIDITY/p' $LOG/study_3b.log | grep -v -E "it/s|INFO"
} > $LOG/REPORT.txt
echo "DONE $(date) - report: $LOG/REPORT.txt" >> $LOG/STATUS
