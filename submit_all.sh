#!/bin/bash
# Chain the four stages with afterok dependencies and print the job ids.
#
# The cluster runs at most two jobs at a time, so every stage is sized to exactly two
# concurrent tasks: two caption shards, two prune shards, the two graphs, and the
# three eval rows throttled with %2. Stages are serialised by afterok anyway, so
# nothing ever competes across stages.
#
# Run setup.sh on the login node first.
#
#   bash submit_all.sh         full run,  ~12h wall-clock
#   bash submit_all.sh 1000    smoke run, ~2h
#
# The limit travels to *every* stage through VALIK_LIMIT, not just to the eval: a
# smoke run captions the first N train and N test problems, builds the KG from that
# same prefix, and answers those N test questions. All four stages slice the same
# prefix of pid_splits.json, so the subset stays coherent.
#
# Smoke outputs are tagged -n<N> (outputs/kg_text_image-n1000,
# outputs/results/nokg-n1000.json), so they never collide with a full run and their
# .build_done marker can never skip the real build.
#
# DO THE SMOKE RUN FIRST. It proves the plumbing - vLLM comes up, LightRAG parses
# entities, answers parse - in 1.5h instead of failing 6h into a full run. Judge it
# on the parse rate in the eval log, NOT on accuracy: a KG built from 1000 problems
# is too small to reproduce the paper's numbers.

set -euo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs outputs/results
LIMIT="${1:-0}"
EXPORT="ALL,VALIK_LIMIT=${LIMIT}"

if [ "$LIMIT" != "0" ]; then
    echo "### SMOKE RUN: first $LIMIT pids of train and test ###"
else
    echo "### FULL RUN ###"
fi

J_CAP=$(sbatch --parsable --export="$EXPORT" jobs/10_caption.slurm)
echo "caption   : $J_CAP  (array 0-1)"

J_PRUNE=$(sbatch --parsable --export="$EXPORT" --dependency=afterok:$J_CAP jobs/20_prune.slurm)
echo "prune     : $J_PRUNE  (array 0-1, after $J_CAP)"

J_KG=$(sbatch --parsable --export="$EXPORT" --dependency=afterok:$J_PRUNE jobs/30_build_kg.slurm)
echo "build_kg  : $J_KG  (array 0-1: image_only, text_image; after $J_PRUNE)"

J_EVAL=$(sbatch --parsable --export="$EXPORT" --dependency=afterok:$J_KG jobs/40_eval.slurm)
echo "eval      : $J_EVAL  (array 0-2%2: nokg, image_only, text_image; after $J_KG)"

echo
echo "watch   : squeue -u \$USER"
echo "logs    : tail -f logs/*-${J_CAP}*.out"
echo "collect : python repro/report.py"
