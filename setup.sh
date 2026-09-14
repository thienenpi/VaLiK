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
    nano-vectordb networkx graspologic tiktoken tenacity xxhash \
    openai aiohttp aiofiles pydantic python-dotenv tqdm

# numpy 1.x, last. vLLM pulls numpy 2, but the python310 env is shared and full
# of wheels compiled against the 1.x ABI (pyarrow, pandas, scipy, contourpy);
# under numpy 2 they die with "AttributeError: _ARRAY_API not found". Everything
# this pipeline needs runs on 1.26, so pin down rather than chase each wheel.
python -m pip install "numpy<2"

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

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
