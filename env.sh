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

# Qwen2.5-32B-AWQ is ~19 GB: gpu02's 40 GB A100 holds it plus the KV cache and the
# embedding model. If that repo 404s, Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 drops in.
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
export KG_MODEL="${KG_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
export QA_MODEL="${QA_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export EMBED_MODEL="${EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
export CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

# Paper Sec 4.1: tau = 0.20 for ScienceQA.
export TAU="${TAU:-0.20}"

# 32768 leaves only 29568 tokens of KV cache on a 40 GB card, and vLLM refuses to
# start. Build chunks are 1200 tokens and eval's context budget is 12000, so 16384 is
# ample; an 80 GB card can go back to 32768.
export VALIK_MAX_MODEL_LEN="${VALIK_MAX_MODEL_LEN:-16384}"

# Nodes the GPU stages must not land on (comma-separated, for sbatch --exclude).
#
# gpu02 only, and the reason is the GPU, not the driver: gpu02 is an A100 (compute
# capability 8.0), gpu04 a Tesla V100-DGXS (7.0). vLLM's AWQ kernels need 7.5 and its
# bfloat16 path needs 8.0, so neither vLLM stage - KG on Qwen2.5-32B-AWQ, eval on
# bf16 Qwen2.5-7B - can start on a V100 at all:
#   ValueError: The quantization method awq is not supported for the current GPU.
#               Minimum capability: 75. Current capability: 70.
# gpu01 is out on driver grounds; gpu03 has never been measured, so it stays out with
# it. The cost is concurrency - the stages are sized for two concurrent tasks, so on
# one node expect roughly double submit_all.sh's wall-clock estimates.
export VALIK_EXCLUDE="${VALIK_EXCLUDE-gpu03,gpu04}"

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

# vLLM does check that its kernels can run on this GPU - but only after the weights
# are fetched and an engine subprocess has spawned, so a scheduling mistake surfaces
# as 70 lines of traceback many minutes in. Check the same floors here, in seconds,
# and say which knob moves the job.
require_gpu_for_model() {
    python - "$1" <<'PY'
import os
import sys

import torch

model = sys.argv[1]
node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
major, minor = torch.cuda.get_device_capability()

# vLLM's get_min_capability() per quantization method, plus the bfloat16 floor that
# _check_if_gpu_supports_dtype() enforces for an unquantized bf16 checkpoint.
FLOORS = {"awq": 75, "awq_marlin": 75, "gptq": 60, "gptq_marlin": 80,
          "fp8": 89, "compressed-tensors": 75, "bitsandbytes": 75}


def required():
    """(floor, why) for this checkpoint, or (None, None) if we cannot tell."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model)
    except Exception as e:
        print(f"    note: cannot read config.json for {model} ({e});"
              " skipping the capability preflight", flush=True)
        return None, None
    quant = getattr(cfg, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else {}
    method = str(quant.get("quant_method", "")).lower()
    if method:
        # An unlisted method is one we have no floor for: let vLLM rule on it.
        return FLOORS.get(method), f"the {method} kernels"
    if str(getattr(cfg, "torch_dtype", "")).endswith("bfloat16"):
        return 80, "its bfloat16 weights (--dtype half avoids this, at some fidelity)"
    return 0, "this GPU"


floor, why = required()
if floor is None or major * 10 + minor >= floor:
    sys.exit(0)

excl = ','.join(dict.fromkeys(filter(
    None, os.environ.get('VALIK_EXCLUDE', '').split(',') + [node])))

print(
    f"FATAL: {torch.cuda.get_device_name(0)} on {node} is compute capability "
    f"{major}.{minor},\n"
    f"       but {model} needs {floor // 10}.{floor % 10} for {why}.\n"
    "       This node is the wrong shape for the stage, not merely a slow one, so\n"
    "       keep the job off it rather than waiting on it:\n"
    f"         VALIK_EXCLUDE={excl} bash submit_all.sh ...\n"
    "       Failing here so afterok stops the chain before the weights download.",
    file=sys.stderr, flush=True,
)
sys.exit(1)
PY
}

enable_hf_transfer() {
    if [ -z "${VALIK_HF_TRANSFER_CHECKED+x}" ]; then
        if python -c "import hf_transfer" 2>/dev/null; then
            export HF_HUB_ENABLE_HF_TRANSFER=1
        else
            export HF_HUB_ENABLE_HF_TRANSFER=0
        fi
        export VALIK_HF_TRANSFER_CHECKED=1
    fi
}

prefetch_model() {
    local model="$1"
    [ -d "$model" ] && return 0
    enable_hf_transfer
    python - "$model" <<'PY'
import sys
import time

from huggingface_hub import HfApi, snapshot_download

model = sys.argv[1]

ignore = None
try:
    files = HfApi().list_repo_files(model)
    if any(f.endswith(".safetensors") for f in files):
        ignore = ["*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5"]
except Exception as e:
    print(f"    could not list {model} ({e}); fetching every file", flush=True)

t0 = time.time()
path = snapshot_download(model, ignore_patterns=ignore)
print(f"=== weights ready: {model} ({time.time() - t0:.0f} s)\n    {path}", flush=True)
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

    require_gpu_for_model "$model" || return 1

    echo "=== fetching weights if the cache is cold: $model"
    prefetch_model "$model" || {
        echo "ERROR: could not fetch $model into HF_HOME=$HF_HOME" >&2
        return 1
    }

    echo "=== starting vLLM: $model on port $port ${quiet[*]}"
    vllm serve "$model" --port "$port" --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len "$VALIK_MAX_MODEL_LEN" \
        "${quiet[@]}" "$@" \
        > "$VALIK_ROOT/logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null' EXIT

    local log="$VALIK_ROOT/logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log"
    local wait_min="${VALIK_VLLM_WAIT_MIN:-45}"
    local i
    for i in $(seq 1 $((wait_min * 6))); do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null; then
            echo "=== vLLM ready after $SECONDS s"
            return 0
        fi
        kill -0 $VLLM_PID 2>/dev/null || {
            # The reason is usually one line (rejected flag, OOM, 404 model repo),
            # so put it in the job log rather than only naming the file.
            echo "ERROR: vLLM died during startup; last 20 lines of" \
                 "logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log:" >&2
            tail -n 20 "$log" >&2
            return 1
        }
        if [ $((i % 30)) -eq 0 ]; then
            echo "    still loading after $((i / 6)) of $wait_min min: $(tail -n 1 "$log")"
        fi
        sleep 10
    done
    echo "ERROR: vLLM did not become healthy within $wait_min min;" \
         "raise \$VALIK_VLLM_WAIT_MIN if the load is just slow. Last 20 lines of" \
         "logs/vllm-${jobid}-${SLURM_ARRAY_TASK_ID:-0}.log:" >&2
    tail -n 20 "$log" >&2
    return 1
}
