#!/bin/bash
# Exits on the next matrix cell end (or the chain's end) in a campaign's chain.out: one wake per cell, to judge it and
# rewrite the report (supervisor 2026-10-22 12:13). The campaign directory must lie under a writable root (atelier.json).
# usage: watch_cells.sh <campaign dir>
HERE=$(cd "$(dirname "$0")/.." && pwd); D=$(python3 "$HERE/atelier.py" writable "$1") || exit 2
tail -n0 -F "$D/chain.out" 2>/dev/null | while IFS= read -r l; do
  case "$l" in "CELL "*" END "*|*"MATRIX END"*|*" END "[0-9]*|"== "*" END"*) echo "$(date '+%F %T') $l"; pkill -P $$ -x tail; exit 0;; esac
done
