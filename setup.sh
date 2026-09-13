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
python -m pip install --upgrade "vllm>=0.6.3"

# The rest, minus the upstream requirements.txt entries this reproduction does not
# use. clip-interrogator==0.6.0 in particular pins an old open_clip and drags in a
# conflicting torch; the minimal pipeline reaches CLIP through transformers instead.
python -m pip install \
    "transformers>=4.49" accelerate qwen-vl-utils einops \
    nltk pillow opencv-python \
    nano-vectordb networkx graspologic "scipy>=1.13" tiktoken tenacity xxhash \
    openai aiohttp aiofiles pydantic python-dotenv tqdm numpy

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# ---------------------------------------------------------------------- dataset
mkdir -p datasets
cd datasets
if [ ! -d ScienceQA ]; then
    git clone https://github.com/lupantech/ScienceQA
    (cd ScienceQA && bash tools/download.sh)
else
    echo "=== ScienceQA already present"
fi
cd ..

python - <<'PY'
import os, sys
sys.path.insert(0, "repro")
from common import SQA_ROOT, load_problems, load_splits, load_sqa_captions
print("SQA_ROOT:", SQA_ROOT)
problems, splits = load_problems(), load_splits()
print("problems:", len(problems))
for k in ("train", "val", "test"):
    pids = splits[k]
    n_img = sum(1 for p in pids if problems[p].get("image"))
    print(f"  {k:5s}: {len(pids):6d} questions, {n_img:6d} with images")
caps = load_sqa_captions()
print("captions.json:", len(caps), "entries" if caps else "MISSING - the no-KG baseline will have no image description")
PY

echo
echo "Setup done. Next: sbatch jobs/10_caption.slurm"
