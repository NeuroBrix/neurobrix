#!/bin/bash
# Woken by a MESSAGE (Hocine 2026-10-21 20:36): follows the inbox and the peer file, exits on the first new "## " heading
# (a background task's exit is the agent's wake). Paths from atelier.json (communication, peer). usage: watch_heading.sh
HERE=$(cd "$(dirname "$0")/.." && pwd); AT() { python3 "$HERE/atelier.py" "$@" || exit 2; }
I=$(AT root communication)/_supervisor_inbox.md; P=$(AT root peer)/to_mac.md
tail -n0 -F "$I" "$P" 2>/dev/null | while IFS= read -r l; do
  case "$l" in "## "*) echo "$(date '+%F %T') NEW: $l"; pkill -P $$ -x tail; exit 0;; esac
done
