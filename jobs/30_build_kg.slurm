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
#   task 0  image_only  captions of the train images alone      -> Table 3 row 78.88
#   task 1  text_image  those captions + the train QA text      -> Table 3 row 83.16
# text_only (82.98) is not built: with the no-KG baseline and these two, the visual
# contribution is already isolated (74.72 -> 78.88 is vision alone). Add it later by
# submitting with --array=2 if the text/vision decomposition is wanted.
#
# Serving through vLLM rather than Ollama is what makes the stage fit in a day.
# Ollama has no continuous batching, which is why Appendix F measures only 196k
# tokens/hour; one A100 running vLLM does that in minutes. Same model, same prompts,
# same LightRAG - only the server changes, via LightRAG's own OpenAI backend.
#
# Qwen2.5-32B-AWQ builds the graph instead of the paper's DeepSeek-R1-70B: the repo
# README recommends Qwen2.5, the paper itself flags R1 as anomalous (Sec 4.2), and
# R1's <think> block breaks LightRAG's delimiter-based entity parsing. build_kg.py
# strips <think> regardless, so KG_MODEL=deepseek-ai/... still works if wanted.
#
# Requeue safety: build_kg.py inserts a *list* of documents, so LightRAG's
# doc_status checkpoint (lightrag.py:373-390) skips everything already PROCESSED on
# a restart. The upstream demo inserts one giant string, which would throw away the
# whole run. A .build_done marker stops a finished graph from being rebuilt.
#
# Usage: sbatch jobs/30_build_kg.slurm [caption_suffix]   (default: .pruned.txt)
#        sbatch --array=0-1 jobs/30_build_kg.slurm .txt   -> the -SV ablation

cd "$SLURM_SUBMIT_DIR" || exit 1
source env.sh
SUFFIX="${1:-.pruned.txt}"
LIMIT="${VALIK_LIMIT:-0}"
MODES=(image_only text_image text_only)
MODE="${MODES[${SLURM_ARRAY_TASK_ID:-0}]}"
# The limit is part of the tag: a smoke graph must not sit in the full run's
# working dir, where its .build_done marker would skip the real build.
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

# max-async 32, not the 160 of lightrag_ollama_demo.py:23. That number was tuned
# against Ollama; vLLM's scheduler starts preempting and thrashing the KV cache well
# before it.
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
