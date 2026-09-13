#!/bin/bash
#SBATCH --job-name=valik-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-2

# Produce the three Table 3 rows, one array task each, Qwen2.5-7B throughout
# (paper Sec 4.1 uses 7B for retrieval and reasoning; Appendix D measures only a
# 1-5% spread from going larger, at a large cost in time):
#
#   task 0  nokg        Qwen2.5-7B                        paper 74.72
#   task 1  image_only  Qwen2.5-7B (VaLiK Image-only)     paper 78.88
#   task 2  text_image  Qwen2.5-7B (VaLiK Text-Image)     paper 83.16
#
# Task 0 deliberately reads ScienceQA's own captions.json rather than our Qwen2-VL
# descriptions. Table 3 gives the un-augmented baseline 65.79 on the IMG column,
# well above chance, so it is clearly fed some description; feeding it ours would
# raise the baseline and shrink the gain VaLiK is credited with.
#
# The goal is the ordering, not the decimals: 74.72 < 78.88 < 83.16, with the
# 7B+VaLiK row beating native Qwen2.5-72B (78.37). Expect 1-3 points below the paper
# from dropping the BLIP-2/LLaVA cascade and the 70B graph model. A parse rate below
# ~95% in the log means the prompt is wrong, not the KG - fix that before reading
# any accuracy.
#
# Requeue safety: answers stream to <out>.jsonl and finished pids are skipped on
# restart, so a preempted eval resumes mid-split.
#
# Usage: sbatch jobs/40_eval.slurm                          (full test split)
#        sbatch --export=ALL,VALIK_LIMIT=500 jobs/40_eval.slurm   (smoke)

cd "$SLURM_SUBMIT_DIR" || exit 1
source env.sh
LIMIT="${VALIK_LIMIT:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CONFIGS=(nokg image_only text_image)
CFG="${CONFIGS[$TASK]}"
SUFFIX="$( [ "$LIMIT" = "0" ] && echo "full" || echo "n${LIMIT}" )"
OUT="outputs/results/${CFG}-${SUFFIX}"
LOG="logs/eval-${CFG}-${SUFFIX}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate python310

echo "Job $SLURM_JOB_ID on $(hostname); config=$CFG; limit=$LIMIT; started $(date)"
nvidia-smi || true

if [ -f "${OUT}.json" ]; then
    echo "=== skipped: ${OUT}.json already exists"
    cat "${OUT}.json"
    exit 0
fi

start_vllm "$QA_MODEL" || exit 1

if [ "$CFG" = "nokg" ]; then
    python repro/eval_scienceqa.py \
        --mode nokg \
        --base-url "$BASE_URL" --llm-model "$QA_MODEL" \
        --caption-source sqa \
        --split test --limit "$LIMIT" --concurrency 32 \
        --out "$OUT" || exit 1
else
    WORKDIR="outputs/kg_${CFG}$( [ "$LIMIT" != "0" ] && echo "-n${LIMIT}" )"
    if [ ! -f "${WORKDIR}/.build_done" ]; then
        echo "ERROR: ${WORKDIR} not built yet - run jobs/30_build_kg.slurm first" >&2
        exit 1
    fi
    # hybrid: paper Appendix D, "balances the precision of local cues with the
    # breadth of global knowledge".
    python repro/eval_scienceqa.py \
        --mode kg --working-dir "$WORKDIR" \
        --base-url "$BASE_URL" --llm-model "$QA_MODEL" \
        --embed-model "$EMBED_MODEL" \
        --query-mode hybrid \
        --caption-source valik --caption-suffix .pruned.txt \
        --split test --limit "$LIMIT" --concurrency 32 \
        --out "$OUT" || exit 1
fi

echo "Finished $(date)"
