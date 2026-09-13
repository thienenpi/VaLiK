#!/bin/bash
# Shared environment for every VaLiK reproduction job. Sourced, never submitted.
#
# Everything the pipeline touches has to live under /media/lhbac29, so this also
# redirects the caches that otherwise land in $HOME without asking:
#   HF_HOME          ~53 GB of weights (Qwen2-VL-7B, Qwen2.5-32B-AWQ, Qwen2.5-7B,
#                    CLIP-ViT-L/14, nomic-embed)
#   VLLM_CACHE_ROOT  vLLM's torch.compile artefacts, a few GB per server start
#   TRITON_CACHE_DIR Triton kernel cache, written on every vLLM launch
#   TMPDIR           LightRAG and vLLM both spill large temporaries here

export DATA_ROOT="${DATA_ROOT:-/media/lhbac29}"
export VALIK_ROOT="${SLURM_SUBMIT_DIR:-$DATA_ROOT/VaLiK}"

export HF_HOME="$DATA_ROOT/hf"
export VLLM_CACHE_ROOT="$DATA_ROOT/vllm"
export TRITON_CACHE_DIR="$DATA_ROOT/triton"
export TMPDIR="$DATA_ROOT/tmp"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" "$VALIK_ROOT/logs"

# LightRAG is vendored, not pip-installed: src/LightRAG/lightrag is imported directly
# so the pinned reference version the authors shipped stays in control.
export PYTHONPATH="$VALIK_ROOT/src/LightRAG:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# Models. Qwen2.5-32B-Instruct-AWQ is ~19 GB, so one 80 GB card holds it with room
# for the KV cache and the 137M embedding model alongside. If that repo 404s, the
# GPTQ build (Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4) is the drop-in alternative.
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
export KG_MODEL="${KG_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
export QA_MODEL="${QA_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export EMBED_MODEL="${EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
export CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

# Paper Sec 4.1: tau = 0.20 for ScienceQA.
export TAU="${TAU:-0.20}"

# Start a vLLM OpenAI server on a port nobody else on this node is using, wait for
# it, and make sure it dies with the job. Sets $BASE_URL.
start_vllm() {
    local model="$1"; shift
    local port=$((20000 + (SLURM_JOB_ID % 20000) + ${SLURM_ARRAY_TASK_ID:-0} * 13))
    export BASE_URL="http://127.0.0.1:${port}/v1"

    echo "=== starting vLLM: $model on port $port"
    vllm serve "$model" --port "$port" --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len 32768 \
        --disable-log-requests "$@" \
        > "$VALIK_ROOT/logs/vllm-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null' EXIT

    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null; then
            echo "=== vLLM ready after $SECONDS s"
            return 0
        fi
        kill -0 $VLLM_PID 2>/dev/null || {
            echo "ERROR: vLLM died during startup; see logs/vllm-${SLURM_JOB_ID}-*.log" >&2
            return 1
        }
        sleep 10
    done
    echo "ERROR: vLLM did not become healthy within 30 min" >&2
    return 1
}
