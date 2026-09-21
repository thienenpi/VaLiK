#!/bin/bash
#SBATCH --job-name=valik-caption
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# Stage 1: turn every ScienceQA image into text (paper Sec 3.1).
#
# Two shards, one GPU each - the cluster runs two jobs at a time. Shards interleave
# (images[i::2]) so both finish together. Captions train and test, not val: the MMKG
# comes from train, but a test image reaches the model only through its own
# description. Requeue-safe via the per-image skip in caption.py.
#
# Usage: sbatch jobs/10_caption.slurm            (defaults: train+test, .txt)
#        sbatch jobs/10_caption.slurm "train"    (train only)

cd "$SLURM_SUBMIT_DIR" || exit 1
source env.sh
SPLITS="${1:-train test}"
LIMIT="${VALIK_LIMIT:-0}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${SLURM_ARRAY_TASK_COUNT:-1}"
LOG="logs/caption-shard${SHARD}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate python310

echo "Job $SLURM_JOB_ID on $(hostname); shard $SHARD/$NSHARD; splits=$SPLITS; limit=$LIMIT; started $(date)"
nvidia-smi || true
require_cuda "Qwen2-VL captioning" || exit 1

# max-new-tokens 384, not upstream's 32768: Appendix F's throughput implies ~240
# tokens, so the bigger budget only buys slower generation.
python repro/caption.py \
    --model "$VLM_MODEL" \
    --splits $SPLITS \
    --shard-id "$SHARD" --num-shards "$NSHARD" \
    --max-new-tokens 384 \
    --limit "$LIMIT" \
    --suffix .txt || exit 1

echo "Finished $(date)"
