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
# Every caller cds to the repo root before sourcing this, so $PWD is the right
# fallback outside SLURM - and it keeps working if the repo is ever moved.
export VALIK_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="$DATA_ROOT/hf"
export VLLM_CACHE_ROOT="$DATA_ROOT/vllm"
export TRITON_CACHE_DIR="$DATA_ROOT/triton"
export TMPDIR="$DATA_ROOT/tmp"
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT=$DATA_ROOT does not exist." >&2
    echo "       On the cluster this is /media/lhbac29. Elsewhere, override it:" >&2
    echo "         DATA_ROOT=/some/scratch bash setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" "$VALIK_ROOT/logs"

# LightRAG is vendored, not pip-installed: src/LightRAG/lightrag is imported directly
# so the pinned reference version the authors shipped stays in control.
export PYTHONPATH="$VALIK_ROOT/src/LightRAG${PYTHONPATH:+:$PYTHONPATH}"
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

# Nodes the GPU stages must not land on, as a comma-separated list for sbatch.
#
# gpu02 only. It has the newest driver in the pool (555.42.02); gpu01 is 525.147.05
# and gpu04 is 535.216.03, and gpu01 is where every caption and prune shard died with
# "The NVIDIA driver on your system is too old (found version 12000)". Keeping the
# installed stack and running one node is the cheap fix - the alternative, getting
# gpu01 and gpu04 back, means reinstalling on an older CUDA 12.1 stack (setup.sh).
#
# The cost is concurrency: every stage here is sized for two concurrent tasks (two
# caption shards, two prune shards, two graphs, evals throttled %2). On one node they
# queue behind each other unless gpu02 has a second card, so expect roughly double
# the wall-clock in submit_all.sh's estimates.
export VALIK_EXCLUDE="${VALIK_EXCLUDE-gpu01,gpu03,gpu04}"

# Fail a GPU stage before it downloads 15 GB of weights, not after.
#
# The pinned stack should make this never fire; it is here for when it does. A driver
# torch cannot use either raises on the first CUDA call - caption paid for a 20 min
# weight download before finding out - or, worse, quietly reports no device and the
# stage runs on CPU, which is how prune "finished" in 7 minutes having written
# nothing. torch.cuda.is_available() catches both in about a second.
require_cuda() {
    python - "$@" <<'PY'
import os
import sys

import torch

stage = sys.argv[1] if len(sys.argv) > 1 else "this stage"
node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
built = torch.version.cuda or "cpu-only build"

try:
    ok = torch.cuda.is_available()
    err = ""
except Exception as e:  # a too-old driver raises rather than returning False
    ok, err = False, str(e)

if ok:
    print(f"=== cuda ok on {node}: {torch.cuda.get_device_name(0)}, "
          f"torch {torch.__version__} (built for CUDA {built})", flush=True)
    sys.exit(0)

print(
    f"FATAL: {stage} needs a GPU and torch cannot use one on {node}.\n"
    f"       torch {torch.__version__}, built for CUDA {built}.\n"
    f"       {err or 'torch.cuda.is_available() returned False.'}\n"
    "       CUDA 12.x wheels need driver >= 525.60.13; compare `nvidia-smi` above.\n"
    "       If this node's driver is simply older than the rest, skip it:\n"
    f"         VALIK_EXCLUDE={','.join(dict.fromkeys(filter(None, os.environ.get('VALIK_EXCLUDE', '').split(',') + [node])))} bash submit_all.sh ...\n"
    "       If they are all like this, lower the vLLM pin in setup.sh - it is what\n"
    "       decides which CUDA build of torch gets installed.\n"
    "       Failing here so afterok stops the chain instead of producing nothing.",
    file=sys.stderr, flush=True,
)
sys.exit(1)
PY
}

# Per-request logging: ask the installed vLLM which spelling it takes.
#
# Old builds log every request and take --disable-log-requests to stop; vLLM dropped
# that flag once request logging became opt-in behind --enable-log-requests, and
# argparse rejects an unknown flag outright - which is what killed every kg and eval
# job with "vllm: error: unrecognized arguments: --disable-log-requests" before the
# server ever loaded a model. Probing costs one `vllm serve --help` per job and keeps
# the pipeline working on both sides of that change, with no version pin to chase.
vllm_quiet_flag() {
    if [ -z "${VLLM_QUIET_FLAG+x}" ]; then
        local help
        help="$(vllm serve --help 2>/dev/null)"
        if printf '%s' "$help" | grep -q -- '--disable-log-requests'; then
            VLLM_QUIET_FLAG="--disable-log-requests"
        else
            # Newer vLLM: already quiet by default, nothing to pass.
            VLLM_QUIET_FLAG=""
        fi
        export VLLM_QUIET_FLAG
    fi
    printf '%s' "$VLLM_QUIET_FLAG"
}

# Start a vLLM OpenAI server on a port nobody else on this node is using, wait for
# it, and make sure it dies with the job. Sets $BASE_URL.
start_vllm() {
    local model="$1"; shift
    local jobid="${SLURM_JOB_ID:-$$}"
    local port=$((20000 + (jobid % 20000) + ${SLURM_ARRAY_TASK_ID:-0} * 13))
    export BASE_URL="http://127.0.0.1:${port}/v1"

    local quiet=()
    local flag; flag="$(vllm_quiet_flag)"
    [ -n "$flag" ] && quiet=("$flag")

    echo "=== starting vLLM: $model on port $port ${quiet[*]}"
    vllm serve "$model" --port "$port" --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len 32768 \
        "${quiet[@]}" "$@" \
        > "$VALIK_ROOT/logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null' EXIT

    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null; then
            echo "=== vLLM ready after $SECONDS s"
            return 0
        fi
        kill -0 $VLLM_PID 2>/dev/null || {
            # Show the reason here rather than only naming the file: a failed launch
            # is usually one line (a rejected flag, an OOM, a 404 model repo) and
            # having it in the job log is the difference between a 5-second and a
            # 10-minute diagnosis.
            echo "ERROR: vLLM died during startup; last 20 lines of" \
                 "logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log:" >&2
            tail -n 20 "$VALIK_ROOT/logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log" >&2
            return 1
        }
        sleep 10
    done
    echo "ERROR: vLLM did not become healthy within 30 min" >&2
    return 1
}
