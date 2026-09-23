#!/bin/bash
# Pull run artefacts from the HCMUS server into this local repo, mirroring the SAME
# relative paths (outputs/<tag>/, outputs/results/, logs/). Run by hand from the
# local machine - never submitted.
#
# Safe to run repeatedly, e.g. while a job is still building: rsync moves only what
# changed, --partial resumes an interrupted transfer, and there is no --delete, so a
# local copy is never removed just because the server no longer has it.
#
# A KG working dir is only complete once it contains .build_done (30_build_kg.slurm
# writes it last). Syncing one mid-build is fine - you just get that moment's
# LightRAG checkpoint - but don't read accuracy off a graph without that marker.
set -euo pipefail

# --- config: override from the environment if the server moves ---
REMOTE_USER="${REMOTE_USER:-lhbac29}"
REMOTE_HOST="${REMOTE_HOST:-172.29.74.81}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/media/lhbac29/valik}"   # repo checkout on the server

# Local repo root = directory containing this script (keeps the same paths).
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
usage: $0 [-n] [-l] [pattern]

  pattern   which outputs/ subdirs to pull, glob, default '*'
              e.g. 'kg_text_image-n1000'   one graph
                   '*-n1000'               the smoke run
                   'results'               eval JSON only
  -o, --outputs-only  skip logs/, pull outputs/ alone
  -l, --lean      skip kv_store_llm_response_cache.json (~40% of the bytes; it
                  only speeds up a *rebuild on the server*, nothing local)
  -n, --dry-run   list what would transfer, copy nothing

env: REMOTE_USER REMOTE_HOST REMOTE_PORT REMOTE_DIR
EOF
    exit 1
}

DRY_RUN=0
LEAN=0
WITH_LOGS=1
PATTERN=""
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run)      DRY_RUN=1 ;;
        -l|--lean)         LEAN=1 ;;
        -o|--outputs-only) WITH_LOGS=0 ;;
        -h|--help)    usage ;;
        -*)           echo "unknown option: $1" >&2; usage ;;
        *)            [ -n "$PATTERN" ] && { echo "one pattern at a time" >&2; usage; }
                      PATTERN="$1" ;;
    esac
    shift
done
PATTERN="${PATTERN:-*}"

SSH=(ssh -p "${REMOTE_PORT}")
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# -a archive, -z compress (these are JSON/GraphML - they compress hard),
# --partial resume, --info=progress2 one overall progress bar.
RSYNC_OPTS=(-az --partial --info=progress2 -e "ssh -p ${REMOTE_PORT}")
[ "$DRY_RUN" -eq 1 ] && RSYNC_OPTS+=(--dry-run --itemize-changes)
[ "$LEAN" -eq 1 ] && RSYNC_OPTS+=(--exclude='kv_store_llm_response_cache.json')

"${SSH[@]}" -o BatchMode=yes "$REMOTE" true 2>/dev/null \
    || { echo "cannot ssh ${REMOTE} (key auth). Check the VPN / ssh-agent." >&2; exit 1; }

sync_dir() {  # <relative subdir> [rsync filter args...]
    local sub="$1"; shift
    local src="${REMOTE}:${REMOTE_DIR}/${sub}/"
    local dst="${LOCAL_DIR}/${sub}/"

    if ! "${SSH[@]}" "$REMOTE" "[ -d '${REMOTE_DIR}/${sub}' ]"; then
        echo ">> skip ${sub}/ (not present on server)"
        return
    fi
    echo ">> syncing ${sub}/ -> ${dst}"
    mkdir -p "$dst"
    rsync "${RSYNC_OPTS[@]}" "$@" "$src" "$dst"
}

# outputs/ is one dir per graph plus results/, so the pattern selects whole dirs:
# '***' means "this dir and everything under it".
sync_dir outputs --include="${PATTERN}/***" --exclude='*'

# logs/ is flat and tiny (single-digit MB), and the .out files are the only record
# of how many documents failed to insert, so take all of them unless asked not to.
if [ "$WITH_LOGS" -eq 1 ]; then
    sync_dir logs
else
    echo ">> skip logs/ (--outputs-only)"
fi

echo
echo "local artefacts now:"
du -sh "${LOCAL_DIR}"/outputs/* 2>/dev/null || true
[ "$WITH_LOGS" -eq 1 ] && du -sh "${LOCAL_DIR}/logs" 2>/dev/null || true
echo "Done $(date)"
