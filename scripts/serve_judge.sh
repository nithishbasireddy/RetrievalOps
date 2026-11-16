#!/usr/bin/env bash
# Serve a local judge LLM with vLLM (OpenAI-compatible API on 127.0.0.1).
#   scripts/serve_judge.sh <model_dir> <served_name> <gpu> <port> [gpu_mem_fraction] [extra vllm args...]
# e.g. scripts/serve_judge.sh /scratch/mtech_bnreddy/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct 0 8010 0.25
# GPUs are shared: check `nvidia-smi` and pick a memory fraction that fits next to other jobs.
set -euo pipefail
MODEL_DIR=$1; NAME=$2; GPU=$3; PORT=$4; MEM=${5:-0.4}; shift $(( $# < 5 ? $# : 5 ))
VLLM_BIN=${VLLM_BIN:-/home/mtech_bnreddy/Code-G/DPWA/.venv/bin/vllm}
BASE=${SERVE_DIR:-/scratch/$USER/retrievalops-serve}
export XDG_CACHE_HOME=$BASE/cache TMPDIR=$BASE/tmp
export VLLM_CACHE_ROOT=$BASE/cache/vllm TRITON_CACHE_DIR=$BASE/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$BASE/cache/torchinductor CUDA_CACHE_PATH=$BASE/cache/nvidia
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
exec "$VLLM_BIN" serve "$MODEL_DIR" --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization "$MEM" \
  --max-model-len 8192 --max-num-seqs 128 --max-logprobs 20 "$@"
