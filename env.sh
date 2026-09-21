#!/bin/bash
# Shared environment for every VaLiK reproduction job. Sourced, never submitted.
#
# Caches are redirected under $DATA_ROOT (~53 GB of weights, plus vLLM/Triton
# artefacts and large temporaries) instead of filling up $HOME.

export DATA_ROOT="${DATA_ROOT:-/media/lhbac29}"
# Callers cd to the repo root first, so $PWD is the right fallback outside SLURM.
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

# LightRAG is vendored, not pip-installed: the authors' pinned copy under
# src/LightRAG is imported directly.
export PYTHONPATH="$VALIK_ROOT/src/LightRAG${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Qwen2.5-32B-AWQ is ~19 GB: one 80 GB card holds it plus the KV cache and the
# embedding model. If that repo 404s, Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 drops in.
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
export KG_MODEL="${KG_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
export QA_MODEL="${QA_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export EMBED_MODEL="${EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
export CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

# Paper Sec 4.1: tau = 0.20 for ScienceQA.
export TAU="${TAU:-0.20}"

# Nodes the GPU stages must not land on (comma-separated, for sbatch --exclude).
#
# gpu02 only: it is the only driver new enough for the CUDA 12.4 stack setup.sh
# installs. The cost is concurrency - the stages are sized for two concurrent tasks,
# so on one node expect roughly double submit_all.sh's wall-clock estimates. To use
# the whole pool, reinstall on the CUDA 12.1 stack (setup.sh) and clear this.
export VALIK_EXCLUDE="${VALIK_EXCLUDE-gpu01}"

# Fail a GPU stage before it downloads 15 GB of weights, not after. A driver torch
# cannot use either raises on the first CUDA call or, worse, reports no device and
# lets the stage run on CPU and "finish" having written nothing.
require_cuda() {
    python - "$@" <<'PY'
import os
import subprocess
import sys

import torch

stage = sys.argv[1] if len(sys.argv) > 1 else "this stage"
node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
built = torch.version.cuda or "cpu-only build"


def driver_cuda():
    """The CUDA version this node's driver supports, per nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.split()[0]
    except Exception:
        return None
    # driver -> CUDA is a table, not arithmetic; only this pool's majors are listed.
    major = int(out.split(".")[0])
    for floor, cuda in ((580, "13.0"), (555, "12.5"), (535, "12.2"), (525, "12.0")):
        if major >= floor:
            return f"{cuda} (driver {out})"
    return f"< 12.0 (driver {out})"

try:
    ok = torch.cuda.is_available()
    err = ""
except Exception as e:  # a too-old driver raises rather than returning False
    ok, err = False, str(e)

if ok:
    print(f"=== cuda ok on {node}: {torch.cuda.get_device_name(0)}, "
          f"torch {torch.__version__} (built for CUDA {built})", flush=True)
    sys.exit(0)

have = driver_cuda()
excl = ','.join(dict.fromkeys(filter(
    None, os.environ.get('VALIK_EXCLUDE', '').split(',') + [node])))

print(
    f"FATAL: {stage} needs a GPU and torch cannot use one on {node}.\n"
    f"       torch {torch.__version__}, built for CUDA {built}.\n"
    f"       this node's driver supports CUDA {have or 'unknown - no nvidia-smi'}.\n"
    f"       {err or 'torch.cuda.is_available() returned False.'}\n"
    "       A node runs this torch only if its driver is at least as new as the CUDA\n"
    "       it was built for. If this node is simply older than the rest, skip it:\n"
    f"         VALIK_EXCLUDE={excl} bash submit_all.sh ...\n"
    "       If no node is new enough - gpu02, the newest, is CUDA 12.5 - then torch\n"
    "       itself is too new. Lower the vLLM pin, which is what drags torch in:\n"
    '         VALIK_VLLM="vllm==0.7.3" bash setup.sh   (torch 2.5.1, CUDA 12.4)\n'
    "       Failing here so afterok stops the chain instead of producing nothing.",
    file=sys.stderr, flush=True,
)
sys.exit(1)
PY
}

# Per-request logging: old vLLM needs --disable-log-requests, newer builds dropped
# the flag and argparse rejects it outright, so ask --help which spelling it takes.
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
            # The reason is usually one line (rejected flag, OOM, 404 model repo),
            # so put it in the job log rather than only naming the file.
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
