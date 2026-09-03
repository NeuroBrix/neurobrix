#!/bin/bash
# Mirror the engine's internal corpus into the FORGE repository and push it.
#
# WHY FORGE. These paths are ignored in the public engine repository BY
# DESIGN — they carry roadmap direction, commercial framing and Forge detail
# that does not belong in a public tree. Ignoring them meant the whole
# internal record lived in ONE copy, on one node, in a rack on a single
# breaker with no UPS. Forge is already private and confidential, and it is
# already mirrored on both remotes, so the corpus belongs there rather than
# in a repository of its own.
#
# TWO CONVENTIONS THIS SCRIPT EXISTS TO RESPECT (2026-09-03):
#   * GitLab is a MIRROR of GitHub. A repository is never created on GitLab
#     alone and never carries a different name there. Pushes go to GitHub
#     first, GitLab second.
#   * Creating a repository is a decision about the owner's accounts and is
#     the owner's to make, not this script's.
#
# It is never correct to `git add -f` an ignored path into the public tree
# instead (2026-07-22: history rewrite on both remotes; 2026-09-03:
# recurrence caught before push). Run this instead, whenever an audit,
# verdict or register changes — the corpus is only as durable as its last
# sync.
#
#   bash tools/sync_internal_corpus.sh          # rebuild, commit, push both
#   bash tools/sync_internal_corpus.sh --dry    # rebuild and report, no push
set -euo pipefail

SRC=/home/mlops/NeuroBrix_System
FORGE=$SRC/forge
MIRROR=$FORGE/engine_corpus
DRY=${1:-}

[ -d "$FORGE/.git" ] || { echo "Forge repository not found at $FORGE" >&2; exit 2; }
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
        "$MIRROR" 2>/dev/null || true)
if [ -n "$LEAKS" ]; then
    echo "REFUSING TO PUSH — credential-shaped content found:" >&2
    echo "$LEAKS" >&2
    exit 3
fi

FILES=$(find "$MIRROR" -type f | wc -l)
echo "corpus: $FILES files, $(du -sh "$MIRROR" | cut -f1) -> forge/engine_corpus"

cd "$FORGE"
# Stage ONLY the corpus: Forge's own working tree may carry unrelated
# in-progress work, and this script must never sweep it into a commit.
git add -- engine_corpus
if git diff --cached --quiet -- engine_corpus; then
    echo "no change since the last sync"
    exit 0
fi
git commit -q -m "corpus sync $(date -u +%Y-%m-%dT%H:%MZ) — $FILES files" -- engine_corpus

if [ "$DRY" = "--dry" ]; then
    echo "committed $(git rev-parse --short HEAD); --dry, not pushed"
    exit 0
fi

# GitHub first, GitLab as its mirror.
git push -q origin HEAD:main
git push -q gitlab HEAD:main
echo "pushed $(git rev-parse --short HEAD) -> origin(GitHub) + gitlab(mirror)"
