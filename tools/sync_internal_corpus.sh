#!/bin/bash
# Refresh the private internal-corpus mirror and push it.
#
# The paths below are ignored in this public repository BY DESIGN — they carry
# roadmap direction, commercial framing and build-toolchain detail. Ignoring
# them meant the whole internal record lived in ONE copy, on one node, in a rack
# on a single breaker with no UPS. This script is the second copy.
#
# It is never correct to `git add -f` an ignored path into the public tree
# instead (2026-07-22: history rewrite on both remotes; 2026-09-03: recurrence
# caught before push). Run this instead, whenever an audit, verdict or register
# changes — the corpus is only as durable as its last sync.
#
#   bash tools/sync_internal_corpus.sh          # rebuild, commit, push
#   bash tools/sync_internal_corpus.sh --dry    # rebuild and report, no push
set -euo pipefail

SRC=/home/mlops/NeuroBrix_System
MIRROR=/home/mlops/neurobrix_internal
REMOTE=gitlab
DRY=${1:-}

cd "$SRC"

# --- doc trees and registers (whole; all text) ---------------------------------
for d in docs/audits docs/internal docs/lessons docs/roadmap docs/verdicts; do
    [ -d "$d" ] || continue
    mkdir -p "$MIRROR/$d"
    rsync -a --delete --exclude='.git' "$d/" "$MIRROR/$d/"
done
for f in DETTE.md CLAUDE.md src/neurobrix/CLAUDE.md MODIFICATIONS.md ROADMAP_INTERNAL.md; do
    [ -f "$f" ] || continue
    mkdir -p "$MIRROR/$(dirname "$f")"
    cp "$f" "$MIRROR/$f"
done

# --- validation evidence: TEXT records only ------------------------------------
# Verdicts and dossiers travel; the binary artifacts they cite (renders, frames,
# WAVs, containers) stay on the node — 7.6 GB, and reproducible from the
# configuration each verdict records.
rsync -a --delete --include='*/' --include='*.md' --include='*.txt' --exclude='*' \
      --prune-empty-dirs validation_outputs/ "$MIRROR/validation_outputs/"

# --- gate: no credential may ever enter the mirror -----------------------------
LEAKS=$(grep -rIlE "glpat-|ghp_|github_pat_|hf_[A-Za-z0-9]{34}|-----BEGIN [A-Z ]*PRIVATE KEY" \
        "$MIRROR" --exclude-dir=.git 2>/dev/null || true)
if [ -n "$LEAKS" ]; then
    echo "REFUSING TO PUSH — credential-shaped content found:" >&2
    echo "$LEAKS" >&2
    exit 3
fi

cd "$MIRROR"
FILES=$(find . -type f -not -path './.git/*' | wc -l)
echo "corpus: $FILES files, $(du -sh --exclude=.git . | cut -f1)"

if ! git diff --quiet HEAD -- 2>/dev/null || [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -q -m "corpus sync $(date -u +%Y-%m-%dT%H:%MZ) — $FILES files"
    if [ "$DRY" = "--dry" ]; then
        echo "committed; --dry, not pushed"
    else
        git push -q "$REMOTE" main
        echo "pushed: $(git rev-parse --short HEAD) -> $REMOTE/main"
    fi
else
    echo "no change since the last sync"
fi
