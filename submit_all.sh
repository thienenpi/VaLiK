#!/bin/bash
# Submit the reproduction pipeline: all four stages chained, a slice of them, or one.
#
#   bash submit_all.sh                       full chain, ~12h wall-clock
#   bash submit_all.sh --limit 1000          smoke chain, ~2h   (do this first)
#   bash submit_all.sh --stage caption       just stage 1, no dependency
#   bash submit_all.sh --from kg             kg + eval, chained
#   bash submit_all.sh --stage kg --kg-suffix .txt    the -SV ablation
#   bash submit_all.sh --dry-run --limit 500 print the sbatch lines, submit nothing
#
# Stage names: caption, prune, kg, eval.
#
# --stage submits exactly one stage with no --dependency, which is what you want when
# an earlier stage already finished (or was fixed and re-run by hand). --from submits
# that stage and everything after it, chained with afterok. Both honour --limit.
#
# The limit travels to *every* stage through VALIK_LIMIT, not just to the eval: a
# smoke run captions the first N train and N test problems, builds the KG from that
# same prefix, and answers those N test questions. All four stages slice the same
# prefix of pid_splits.json, so the subset stays coherent. Smoke artefacts are tagged
# -n<N> (outputs/kg_text_image-n1000, outputs/results/nokg-n1000.json), so they never
# collide with a full run and their .build_done marker can never skip the real build.
#
# The cluster runs at most two jobs at a time, so every stage is sized to exactly two
# concurrent tasks: two caption shards, two prune shards, the two graphs, and the
# three eval rows throttled with %2. Stages are serialised by afterok anyway, so
# nothing ever competes across stages.
#
# DO THE SMOKE RUN FIRST. It proves the plumbing - vLLM comes up, LightRAG parses
# entities, answers parse - in 2h instead of failing 6h into a full run. Judge it on
# the parse rate in the eval log, NOT on accuracy: a KG built from 1000 problems is
# far too small to reproduce the paper's numbers.
#
# Run setup.sh on the login node first, and repro/check_data.py if any stage reports
# finding no images.

set -euo pipefail
cd "$(dirname "$0")" || exit 1

# For $VALIK_EXCLUDE: the nodes whose driver is too old for the installed torch.
source env.sh

STAGES=(caption prune kg eval)
LIMIT=0
ONLY=""
FROM=""
KG_SUFFIX=""
DRY=0

usage() { sed -n '2,34p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

is_stage() {
    local s
    for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done
    echo "unknown stage '$1' (want: ${STAGES[*]})" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        -l|--limit)     LIMIT="$2"; shift 2 ;;
        -s|--stage)     is_stage "$2"; ONLY="$2"; shift 2 ;;
        -f|--from)      is_stage "$2"; FROM="$2"; shift 2 ;;
        --kg-suffix)    KG_SUFFIX="$2"; shift 2 ;;
        -n|--dry-run)   DRY=1; shift ;;
        -h|--help)      usage 0 ;;
        [0-9]*)         LIMIT="$1"; shift ;;   # bare number = --limit, kept for habit
        *)              echo "unknown option '$1'" >&2; usage 2 ;;
    esac
done

if [ -n "$ONLY" ] && [ -n "$FROM" ]; then
    echo "--stage and --from are mutually exclusive" >&2; exit 2
fi

mkdir -p logs outputs/results
EXPORT="ALL,VALIK_LIMIT=${LIMIT}"

# Empty means "no exclusion", which sbatch would reject as an empty --exclude.
EXCL=()
[ -n "${VALIK_EXCLUDE:-}" ] && EXCL=(--exclude="$VALIK_EXCLUDE")

# Which stages to submit, in order.
SELECTED=()
if [ -n "$ONLY" ]; then
    SELECTED=("$ONLY")
else
    # Explicit ifs, not an && chain: under set -e a failing test in the middle of an
    # AND-OR list is only exempt because it is not the list's last command, which is
    # too subtle to rely on.
    started=0
    [ -z "$FROM" ] && started=1
    for s in "${STAGES[@]}"; do
        if [ "$s" = "$FROM" ]; then
            started=1
        fi
        if [ "$started" = 1 ]; then
            SELECTED+=("$s")
        fi
    done
fi

script_for() {
    case "$1" in
        caption) echo "jobs/10_caption.slurm" ;;
        prune)   echo "jobs/20_prune.slurm" ;;
        kg)      echo "jobs/30_build_kg.slurm" ;;
        eval)    echo "jobs/40_eval.slurm" ;;
    esac
}

shape_of() {
    case "$1" in
        caption) echo "array 0-1: 2 shards" ;;
        prune)   echo "array 0-1: 2 shards" ;;
        kg)      echo "array 0-1: image_only, text_image" ;;
        eval)    echo "array 0-2%2: nokg, image_only, text_image" ;;
    esac
}

if [ "$LIMIT" != "0" ]; then
    echo "### SMOKE: first $LIMIT pids of train and test ###"
else
    echo "### FULL RUN ###"
fi
[ "$DRY" = 1 ] && echo "### DRY RUN - nothing will be submitted ###"
echo "stages: ${SELECTED[*]}"
if [ -n "${VALIK_EXCLUDE:-}" ]; then
    echo "exclude: $VALIK_EXCLUDE  (driver too old for the installed torch)"
else
    echo "exclude: none - every node must satisfy the installed torch"
fi
echo

PREV=""
FIRST_ID=""
for s in "${SELECTED[@]}"; do
    script="$(script_for "$s")"
    args=()
    [ "$s" = "kg" ] && [ -n "$KG_SUFFIX" ] && args=("$KG_SUFFIX")

    dep=()
    # No dependency on the first stage submitted: whatever precedes it either already
    # ran or is being skipped on purpose.
    [ -n "$PREV" ] && dep=(--dependency=afterok:"$PREV")

    if [ "$DRY" = 1 ]; then
        echo "sbatch --export=$EXPORT ${EXCL[*]-} ${dep[*]} $script ${args[*]}"
        PREV="<${s}_id>"
        continue
    fi

    id=$(sbatch --parsable --export="$EXPORT" "${EXCL[@]}" "${dep[@]}" "$script" "${args[@]}")
    printf "%-9s : %s  (%s)%s\n" "$s" "$id" "$(shape_of "$s")" \
        "$([ -n "$PREV" ] && echo ", after $PREV")"
    [ -z "$FIRST_ID" ] && FIRST_ID="$id"
    PREV="$id"
done

[ "$DRY" = 1 ] && exit 0

echo
echo "watch   : squeue -u \$USER"
echo "logs    : tail -f logs/*-${FIRST_ID}*.out"
echo "collect : python repro/report.py"
