#!/bin/bash
#SBATCH --job-name=valik-prune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# Stage 2: Cross-Modal Similarity Verification (paper Sec 3.2), tau = 0.20.
#
# Cheap (~15 min). Reads image.txt, writes image.pruned.txt; keeping both makes the
# -SV ablation free later via build_kg.py --caption-suffix .txt.
#
# Expect roughly half the sentences to survive. Do NOT tune tau to improve the
# downstream number - Table 4 has SV *reducing* image-only accuracy, so reproducing
# that dip is evidence the implementation is faithful.
#
# Usage: sbatch jobs/20_prune.slurm [tau]        (default: $TAU from env.sh)

cd "$SLURM_SUBMIT_DIR" || exit 1
source env.sh
THRESHOLD="${1:-$TAU}"
LIMIT="${VALIK_LIMIT:-0}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${SLURM_ARRAY_TASK_COUNT:-1}"
LOG="logs/prune-shard${SHARD}-t${THRESHOLD}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate python310

echo "Job $SLURM_JOB_ID on $(hostname); shard $SHARD/$NSHARD; tau=$THRESHOLD; limit=$LIMIT; started $(date)"
nvidia-smi || true
require_cuda "CLIP similarity verification" || exit 1

python repro/prune.py \
    --model "$CLIP_MODEL" \
    --splits train test \
    --shard-id "$SHARD" --num-shards "$NSHARD" \
    --threshold "$THRESHOLD" \
    --mode sentence \
    --limit "$LIMIT" \
    --in-suffix .txt --out-suffix .pruned.txt || exit 1

echo "Finished $(date)"
