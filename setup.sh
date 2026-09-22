#!/bin/bash
# Run once on the login node: installs into the python310 conda env, fetches
# ScienceQA, and prefetches the model weights (~53 GB) so no GPU job has to.
#
# Usage: bash setup.sh

set -euo pipefail
cd "$(dirname "$0")" || exit 1
source env.sh
source ~/.bashrc
conda activate python310

echo "=== python: $(which python)"
echo "=== HF_HOME: $HF_HOME"

# vLLM first and alone: it pins torch, so installing it later swaps torch out from
# under everything else. The pin must have an upper bound - each vLLM drags in one
# torch, and a node runs it only if its driver is at least as new as the CUDA that
# torch was built for. Pool: gpu01 CUDA 12.0, gpu04 12.2, gpu02 12.5.
#
#   vllm <= 0.6.3.post1   torch 2.4.0   CUDA 12.1   every node
#   vllm 0.6.4 - 0.7.3    torch 2.5.1   CUDA 12.4   gpu02 only   <- the pin below
#   vllm 0.8.x            torch 2.6.0   CUDA 12.4   gpu02 only
#   vllm >= 0.9           torch >= 2.7  CUDA >= 12.6  no node
#   unpinned (0.29 today) torch 2.13.0  CUDA 13.0     no node
#
# Unpinned is what broke the last run: torch came down as a CUDA 13 build and every
# GPU stage died with "driver ... too old (found version 12050)" - 12050 being gpu02,
# the newest driver in the pool.
#
# For gpu01 + gpu04 instead, take the CUDA 12.1 stack and clear the exclusion:
#   VALIK_VLLM="vllm==0.6.3.post1" VALIK_TRANSFORMERS="transformers>=4.46,<4.47" bash setup.sh
#   VALIK_EXCLUDE= bash submit_all.sh --limit 1000
python -m pip install --upgrade "${VALIK_VLLM:-vllm==0.7.3}"

# Freeze what vLLM chose, so a later dependency that wants a newer torch fails here
# with a resolver error instead of swapping in a CUDA build the nodes cannot run.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python - > "$CONSTRAINTS" <<'PY'
import torch
print("torch==" + torch.__version__.split("+")[0])
print("numpy<2")
PY
echo "=== constraints: $(tr '\n' ' ' < "$CONSTRAINTS")"

# The rest, minus the upstream requirements.txt entries this reproduction does not
# use (clip-interrogator pins an old open_clip and a conflicting torch; CLIP is
# reached through transformers instead).
#
# opencv is capped at <4.12 because 4.12+ declares numpy>=2, which fights the
# numpy<2 pin below. Nothing in repro/ imports cv2 - only the upstream
# src/Image_to_Text scripts do - so this is just keeping pip's resolver quiet.
#
# transformers is bounded both ways: 4.48.2 is what vLLM 0.7.3 requires (and >=4.46
# is where qwen2_vl maps to AutoModelForImageTextToText, which caption.py uses), and
# the ceiling is because vLLM calls transformers internals. An older vLLM needs a
# lower ceiling - see $VALIK_TRANSFORMERS above.
python -m pip install -c "$CONSTRAINTS" \
    "${VALIK_TRANSFORMERS:-transformers>=4.48.2,<4.50}" accelerate qwen-vl-utils einops \
    nltk pillow "opencv-python<4.12" \
    nano-vectordb networkx graspologic tiktoken tenacity xxhash \
    openai aiohttp aiofiles pydantic python-dotenv tqdm hf_transfer pipmaster

# numpy 1.x, last: the shared python310 env is full of wheels built against the 1.x
# ABI (pyarrow, pandas, scipy), which die under numpy 2 with "_ARRAY_API not found".
python -m pip install -c "$CONSTRAINTS" "numpy<2"

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Stop here if torch is built for a CUDA the pool cannot run, instead of finding out
# four dead SLURM stages later. This checks the *build*; require_cuda re-checks the
# node at job start. $VALIK_POOL_CUDA = newest driver in the pool (gpu02, CUDA 12.5);
# raise it when the cluster drivers are upgraded.
VALIK_POOL_CUDA="${VALIK_POOL_CUDA:-12.5}" python - <<'PY' || exit 1
import os
import sys

import torch
import transformers

built = torch.version.cuda
pool = os.environ["VALIK_POOL_CUDA"]
print(f"=== torch {torch.__version__} (CUDA {built or 'n/a'}), "
      f"transformers {transformers.__version__}")

ver = lambda s: tuple(int(p) for p in s.split(".")[:2])
if built and ver(built) > ver(pool):
    print(
        f"FATAL: torch is built for CUDA {built}, but the newest driver in the pool\n"
        f"       only supports CUDA {pool}. Every GPU stage would die in require_cuda\n"
        '       with "The NVIDIA driver on your system is too old".\n'
        "       Pin a lower vLLM - it is what drags torch in. See the table above:\n"
        '         VALIK_VLLM="vllm==0.7.3" bash setup.sh\n'
        "       If the drivers were upgraded instead, raise $VALIK_POOL_CUDA.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"    runs on any node whose driver supports CUDA {built}+ "
      f"(pool max is {pool}); anything below that goes in $VALIK_EXCLUDE (env.sh).")
PY
echo "=== vllm log-requests flag: $(vllm_quiet_flag || true) (empty = quiet by default)"

# ---------------------------------------------------------------------- dataset
mkdir -p datasets
cd datasets
[ -d ScienceQA ] || git clone https://github.com/lupantech/ScienceQA

# Images fetched straight from S3, NOT via ScienceQA's tools/download.sh: that
# script cds into a directory it never creates and exits 0 having downloaded and
# then deleted the zips, leaving problems.json and no images.
IMG_DIR="ScienceQA/data/scienceqa/images"
mkdir -p "$IMG_DIR"
for split in train val test; do
    if [ -d "$IMG_DIR/$split" ]; then
        echo "=== images/$split already extracted ($(find "$IMG_DIR/$split" -mindepth 1 -maxdepth 1 -type d | wc -l) dirs)"
        continue
    fi
    echo "=== downloading images/$split.zip"
    wget -q --show-progress -O "$IMG_DIR/$split.zip" \
        "https://scienceqa.s3.us-west-1.amazonaws.com/images/$split.zip"
    unzip -q "$IMG_DIR/$split.zip" -d "$IMG_DIR"
    rm -f "$IMG_DIR/$split.zip"
done
cd ..

python repro/check_data.py

# ---------------------------------------------------------------- model weights
if [ -n "${VALIK_SKIP_PREFETCH:-}" ]; then
    echo "=== skipping weight prefetch (VALIK_SKIP_PREFETCH set)"
else
    echo "=== prefetching weights into $HF_HOME (~53 GB, first run is slow)"
    for model in "$VLM_MODEL" "$KG_MODEL" "$QA_MODEL" "$EMBED_MODEL" "$CLIP_MODEL"; do
        prefetch_model "$model"
    done
    du -sh "$HF_HOME"
fi

echo
echo "Setup done. Next: bash submit_all.sh --limit 1000"
