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
# Cheap - CLIP scores a whole caption in milliseconds, the paper calls the stage
# "negligible" (Appendix F) - but still two shards, matching stage 1 and the
# cluster's two concurrent job slots.
#
# Reads image.txt, writes image.pruned.txt. Keeping both means the ablation rows of
# Table 4 (-SV) cost nothing extra later: point build_kg.py at --caption-suffix .txt
# instead and no image has to be re-captioned.
#
# Expect the log to report roughly half the sentences surviving. Do NOT tune tau to
# make the downstream number look better: Table 4 has SV *reducing* ScienceQA
# image-only accuracy (NAT -0.92, LAN -1.28), so reproducing that dip is evidence the
# implementation is faithful.
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

python repro/prune.py \
    --model "$CLIP_MODEL" \
    --splits train test \
    --shard-id "$SHARD" --num-shards "$NSHARD" \
    --threshold "$THRESHOLD" \
    --mode sentence \
    --limit "$LIMIT" \
    --in-suffix .txt --out-suffix .pruned.txt || exit 1

echo "Finished $(date)"
