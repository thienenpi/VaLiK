#!/bin/bash
#SBATCH --job-name=valik-caption
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-3

# Stage 1 of the VaLiK reproduction: turn every ScienceQA image into text.
#
# One array task per shard, each on its own GPU - four independent 1-GPU jobs, not
# one 4-GPU job, so this stays inside the 2-GPU-per-job limit. Shards interleave
# (images[i::4]) rather than splitting by split/subject, so all four finish together
# instead of one trailing on the large train set.
#
# Captions both train and test. The MMKG is built from train, but a test question's
# image is never in that graph, so at eval time the test image reaches the model only
# through its own description - which is why the upstream
# CLIP_Interrogator_ScienceQA.py also walks train/val/test. val is skipped: nothing
# in Table 3 uses it, and Get_Text_ScienceQA.py comments out the val pids too.
#
# Requeue safety comes from the per-image skip: caption.py writes image.txt
# atomically (tmp + os.replace) and skips any image that already has one, so a
# preempted shard resumes at the image it was on rather than restarting.
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

# max-new-tokens 384, not the 32768 in src/Image_to_Text/Qwen2VL_ScienceQA.py:65.
# The paper's own throughput figure (Appendix F: 60 tok/s, one image every 4 s)
# implies ~240 tokens, so the upstream budget only buys slower generation.
python repro/caption.py \
    --model "$VLM_MODEL" \
    --splits $SPLITS \
    --shard-id "$SHARD" --num-shards "$NSHARD" \
    --max-new-tokens 384 \
    --limit "$LIMIT" \
    --suffix .txt || exit 1

echo "Finished $(date)"
