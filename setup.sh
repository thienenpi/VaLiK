#!/bin/bash
# Run once on the login node. Not an sbatch script.
#
# Installs into the existing python310 conda env and fetches ScienceQA. Model
# weights are NOT pre-downloaded: the compute nodes have internet, so the first
# caption shard pulls Qwen2-VL-7B into $HF_HOME and every later job reuses it.
#
# Usage: bash setup.sh

set -euo pipefail
cd "$(dirname "$0")" || exit 1
source env.sh
source ~/.bashrc
conda activate python310

echo "=== python: $(which python)"
echo "=== HF_HOME: $HF_HOME"

# vLLM first and alone: it pins its own torch build, and installing it after the
# other requirements would silently swap torch out from under them.
#
# Which vLLM you install decides which nodes can run anything, because each vLLM
# pins one torch and each torch ships one CUDA runtime with a minimum driver:
#
#   vllm <= 0.6.3.post1   torch 2.4.0    CUDA 12.1    >= 525.60.13    every node
#   vllm 0.6.4 - 0.7.3    torch 2.5.1    CUDA 12.4    >= 550.54.14    gpu02 only
#   vllm >= 0.8.0         torch >= 2.6.0 CUDA >= 12.4 >= 550.54.14    gpu02 only
#
# The pool is gpu01 525.147.05, gpu04 535.216.03, gpu02 555.42.02. The default here
# stays unpinned and the run is confined to gpu02 via $VALIK_EXCLUDE (env.sh), which
# needs no reinstall - that is what killed the caption and prune shards on gpu01:
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12000)
#
# To get gpu01 and gpu04 back instead, install the CUDA 12.1 stack and clear the
# exclusion. transformers has to come down with it: vLLM 0.6.3 uses transformers
# internals, and 4.46 is the floor for AutoModelForImageTextToText + qwen2_vl, which
# repro/caption.py loads Qwen2-VL through.
#   VALIK_VLLM="vllm==0.6.3.post1" VALIK_TRANSFORMERS="transformers>=4.46,<4.47" bash setup.sh
#   VALIK_EXCLUDE= bash submit_all.sh --limit 1000
python -m pip install --upgrade "${VALIK_VLLM:-vllm>=0.6.3}"

# Freeze what vLLM just chose, and install everything else under it. A dependency
# that wants a newer torch now fails here with a resolver error instead of quietly
# swapping in a CUDA build the nodes cannot run - which is the exact failure this
# pin exists to prevent, and it is much cheaper to hit on the login node.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python - > "$CONSTRAINTS" <<'PY'
import torch
print("torch==" + torch.__version__.split("+")[0])
print("numpy<2")
PY
echo "=== constraints: $(tr '\n' ' ' < "$CONSTRAINTS")"

# The rest, minus the upstream requirements.txt entries this reproduction does not
# use. clip-interrogator==0.6.0 in particular pins an old open_clip and drags in a
# conflicting torch; the minimal pipeline reaches CLIP through transformers instead.
#
# 4.46 is the floor: it is the first release that maps qwen2_vl into
# AutoModelForImageTextToText, which repro/caption.py loads Qwen2-VL through (4.45
# has the class but registers the model under AutoModelForVision2Seq only). An older
# vLLM needs a ceiling to match - see $VALIK_TRANSFORMERS above.
python -m pip install -c "$CONSTRAINTS" \
    "${VALIK_TRANSFORMERS:-transformers>=4.46}" accelerate qwen-vl-utils einops \
    nltk pillow opencv-python \
    nano-vectordb networkx graspologic tiktoken tenacity xxhash \
    openai aiohttp aiofiles pydantic python-dotenv tqdm

# numpy 1.x, last. The python310 env is shared and full of wheels compiled against
# the 1.x ABI (pyarrow, pandas, scipy, contourpy); under numpy 2 they die with
# "AttributeError: _ARRAY_API not found". vLLM 0.6.3 requires numpy<2.0.0 itself, so
# this now only guards the rest of the tree - and keeps working if the vLLM pin is
# ever raised past the release that dropped that requirement.
python -m pip install -c "$CONSTRAINTS" "numpy<2"

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Report what decides where the jobs can run, while there is still a terminal to read
# it in. The login node usually has no GPU, so this is about the *build*, not about
# whether CUDA works here; require_cuda re-checks that on the node itself.
python - <<'PY'
import torch
import transformers

cuda = torch.version.cuda or "n/a"
print(f"=== torch {torch.__version__} (CUDA {cuda}), transformers {transformers.__version__}")
print("    CUDA 12.x wheels need driver >= 525.60.13. If `nvidia-smi` on any node")
print("    reports less, that node goes in $VALIK_EXCLUDE (env.sh) - or pin an older")
print("    vLLM, since a higher pin raises the floor to 550.54.14.")
PY
echo "=== vllm log-requests flag: $(vllm_quiet_flag || true) (empty = quiet by default)"

# ---------------------------------------------------------------------- dataset
mkdir -p datasets
cd datasets
[ -d ScienceQA ] || git clone https://github.com/lupantech/ScienceQA

# Images, NOT via ScienceQA's tools/download.sh. That script does
#   cd data/scienceqa/images   (no mkdir -p)
# and runs without set -e, so on a fresh clone the cd fails, wget drops the zips in
# the wrong directory, unzip fails, and the trailing `rm *.zip` deletes them - all
# while exiting 0. You end up with problems.json (which came from the git clone) and
# no images at all. The archives are plain public S3 objects, so fetch them directly.
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

# Verify the layout every stage depends on. Fails loudly if the images are missing.
python repro/check_data.py

echo
echo "Setup done. Next: bash submit_all.sh --limit 1000"
