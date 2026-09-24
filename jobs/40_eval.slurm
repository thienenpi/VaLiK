#!/bin/bash
#SBATCH --job-name=valik-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-2%2
# Node exclusions come from $VALIK_EXCLUDE via submit_all.sh; submitting this file
# by hand bypasses them - use  sbatch --exclude="$VALIK_EXCLUDE" jobs/40_eval.slurm
#
# The three Table 3 rows, one array task each, Qwen2.5-7B throughout (%2 = the two
# concurrent job slots):
#
#   task 0  nokg        Qwen2.5-7B                        paper 74.72
#   task 1  image_only  Qwen2.5-7B (VaLiK Image-only)     paper 78.88
#   task 2  text_image  Qwen2.5-7B (VaLiK Text-Image)     paper 83.16
#
# Task 0 reads ScienceQA's own captions.json, not our Qwen2-VL descriptions: the
# paper's baseline scores 65.79 on IMG, so it is clearly fed some description, and
# feeding it ours would shrink the gain VaLiK is credited with.
#
# Judge the ordering, not the decimals - expect 1-3 points below the paper from
# dropping the BLIP-2/LLaVA cascade and the 70B graph model. A parse rate below ~95%
# means the prompt is wrong, not the KG. Requeue-safe: answers stream to <out>.jsonl.
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

WORKDIR="outputs/kg_${CFG}$( [ "$LIMIT" != "0" ] && echo "-n${LIMIT}" )"
if [ "$CFG" != "nokg" ]; then
    if [ ! -s "${WORKDIR}/graph_chunk_entity_relation.graphml" ]; then
        echo "ERROR: no graph in ${WORKDIR} - run jobs/30_build_kg.slurm first" >&2
        exit 1
    fi
    if [ ! -f "${WORKDIR}/.build_done" ]; then
        echo "WARNING: ${WORKDIR}/.build_done is missing - this graph is a partial"
        echo "         build (check the FATAL line in its kg-*.out log for how many"
        echo "         documents failed). Accuracy below the paper may be that, not VaLiK."
    fi
fi

require_cuda "the eval" || exit 1

# This stage only: the 7B has room for 32k, and LightRAG's hybrid prompts need it.
# The KG build keeps 16384 - Qwen2.5-32B-AWQ leaves no KV cache for more.
export VALIK_MAX_MODEL_LEN=32768

start_vllm "$QA_MODEL" || exit 1

if [ "$CFG" = "nokg" ]; then
    python repro/eval_scienceqa.py \
        --mode nokg \
        --base-url "$BASE_URL" --llm-model "$QA_MODEL" \
        --caption-source sqa \
        --split test --limit "$LIMIT" --concurrency 32 \
        --out "$OUT" || exit 1
else
    # hybrid: paper Appendix D, local cues plus global knowledge.
    python repro/eval_scienceqa.py \
        --mode kg --working-dir "$WORKDIR" \
        --base-url "$BASE_URL" --llm-model "$QA_MODEL" \
        --embed-model "$EMBED_MODEL" \
        --query-mode hybrid \
        --caption-source valik --caption-suffix .pruned.txt \
        --max-context-tokens "${VALIK_CONTEXT_TOKENS:-3000}" \
        --split test --limit "$LIMIT" --concurrency 32 \
        --out "$OUT" || exit 1
fi

echo "Finished $(date)"
