#!/bin/bash
# Submit the reproduction pipeline: all four stages chained, a slice of them, or one.
#
#   bash submit_all.sh                       full chain, ~12h wall-clock
#   bash submit_all.sh --limit 1000          smoke chain, ~2h   (do this first)
#   bash submit_all.sh --stage caption       just stage 1, no dependency
#   bash submit_all.sh --from kg             kg + eval, chained with afterok
#   bash submit_all.sh --stage kg --kg-suffix .txt    the -SV ablation
#   bash submit_all.sh --dry-run --limit 500 print the sbatch lines, submit nothing
#
# Stage names: caption, prune, kg, eval. Run setup.sh on the login node first.
#
# --limit travels to every stage via VALIK_LIMIT and slices the same prefix of
# pid_splits.json everywhere, so a smoke run stays coherent. Its artefacts are tagged
# -n<N> and never collide with a full run.
#
# Do the smoke run first and judge it on the parse rate in the eval log, not on
# accuracy: a KG built from 1000 problems is far too small for the paper's numbers.

set -euo pipefail
cd "$(dirname "$0")" || exit 1

# For $VALIK_EXCLUDE: nodes whose GPU or driver cannot run the installed stack.
source env.sh

STAGES=(caption prune kg eval)
LIMIT=0
ONLY=""
FROM=""
KG_SUFFIX=""
DRY=0

usage() { sed -n '2,19p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

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
    # Explicit ifs, not an && chain: too easy to trip set -e.
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
    echo "exclude: $VALIK_EXCLUDE  (GPU too old for the vLLM stages, or driver too old for torch)"
else
    echo "exclude: none - every node must satisfy torch AND vLLM's compute-capability floors"
fi
echo

PREV=""
FIRST_ID=""
for s in "${SELECTED[@]}"; do
    script="$(script_for "$s")"
    args=()
    [ "$s" = "kg" ] && [ -n "$KG_SUFFIX" ] && args=("$KG_SUFFIX")

    dep=()
    # The first stage submitted gets no dependency.
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
