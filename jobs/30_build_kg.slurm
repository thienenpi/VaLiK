#!/bin/bash
#SBATCH --job-name=valik-kg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# Stage 3: build the MMKGs (paper Sec 3.3 / Algorithm 1). The long pole.
#
# Two array tasks, two graphs, one GPU each:
#   task 0  image_only  train image captions alone       -> Table 3 row 78.88
#   task 1  text_image  those captions + train QA text   -> Table 3 row 83.16
# text_only (82.98) is skipped - the two above already isolate the visual
# contribution. Submit with --array=2 to add it.
#
# Served through vLLM, not Ollama: same model and prompts, but continuous batching
# turns Appendix F's 196k tokens/hour into minutes. Graph model is Qwen2.5-32B-AWQ
# rather than the paper's DeepSeek-R1-70B (the repo README recommends Qwen2.5, and
# R1's <think> block breaks LightRAG's entity parsing - build_kg.py strips it anyway,
# so KG_MODEL=deepseek-ai/... still works).
#
# Requeue-safe: build_kg.py inserts a list of documents, so LightRAG's doc_status
# checkpoint resumes mid-build; .build_done stops a finished graph being rebuilt.
#
# Usage: sbatch jobs/30_build_kg.slurm [caption_suffix]   (default: .pruned.txt)
#        sbatch --array=0-1 jobs/30_build_kg.slurm .txt   -> the -SV ablation

cd "$SLURM_SUBMIT_DIR" || exit 1
source env.sh
SUFFIX="${1:-.pruned.txt}"
LIMIT="${VALIK_LIMIT:-0}"
MODES=(image_only text_image text_only)
MODE="${MODES[${SLURM_ARRAY_TASK_ID:-0}]}"
# The limit is part of the tag: a smoke graph's .build_done must never skip the
# real build.
TAG="${MODE}$( [ "$SUFFIX" = ".txt" ] && echo "-nosv" )$( [ "$LIMIT" != "0" ] && echo "-n${LIMIT}" )"
WORKDIR="outputs/kg_${TAG}"
LOG="logs/kg-${TAG}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate python310

echo "Job $SLURM_JOB_ID on $(hostname); mode=$MODE; captions=$SUFFIX; limit=$LIMIT; started $(date)"
nvidia-smi || true

if [ -f "${WORKDIR}/.build_done" ]; then
    echo "=== skipped: ${WORKDIR} already built"
    exit 0
fi

require_cuda "the KG build" || exit 1

start_vllm "$KG_MODEL" || exit 1

# max-async 32, not the upstream demo's 160: that was tuned against Ollama, and
# vLLM's scheduler thrashes the KV cache well before it.
python repro/build_kg.py \
    --mode "$MODE" \
    --working-dir "$WORKDIR" \
    --base-url "$BASE_URL" \
    --llm-model "$KG_MODEL" \
    --embed-model "$EMBED_MODEL" \
    --split train \
    --caption-suffix "$SUFFIX" \
    --limit "$LIMIT" \
    --max-async 32 || exit 1

touch "${WORKDIR}/.build_done"
du -sh "$WORKDIR"
echo "Finished $(date)"
