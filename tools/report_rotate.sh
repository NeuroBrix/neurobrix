#!/bin/bash
# report_rotate.sh [report] [overflow]: keep the report at most MAX lines (default 150) by moving its OLDEST dated lines
# ("- HH:MM ...": the log; the undated standing bullets — the resume point, the recipes — never move) to the overflow journal. A timestamped copy of the report is taken before any change. Nothing is
# removed when no dated line exists (2026-11-08 04:59: an empty line number made `sed -i "d"` erase the whole report).
# Refuses to leave the report shorter than the lines it did not move.
# Roots from the workshop file ~/.neurobrix/atelier.json (communication, registres), never a path written here.
root() { python3 -c "import json,os,sys; print(json.load(open(os.path.expanduser('~/.neurobrix/atelier.json')))[sys.argv[1]])" "$1"; }
COMM=$(root communication) && REG=$(root registres) || { echo "report_rotate: ~/.neurobrix/atelier.json unreadable — refused" >&2; exit 1; }
F=${1:-$COMM/_report.md}
O=${2:-$REG/journaux/report_acks_out_of_report.md}
B=${REPORT_BACKUPS:-$REG/journaux/report_backups}; MAX=${REPORT_MAX_LINES:-150}
[ -s "$F" ] || { echo "report_rotate: $F is empty or missing — refused (restore it first)" >&2; exit 1; }
mkdir -p "$B"; cp -p "$F" "$B/_report.$(TZ=Europe/Madrid date '+%Y-%m-%d_%H%M%S').md"
n=$(wc -l < "$F")
while [ "$n" -gt "$MAX" ]; do
  L=$(grep -n "^- [0-9][0-9]:[0-9][0-9] " "$F" | head -1 | cut -d: -f1)
  [ -n "$L" ] || { echo "report_rotate: $n lines and no dated line left to move — nothing removed" >&2; exit 0; }
  # the entry = its dated line and the indented lines that continue it
  E=$(awk -v l="$L" 'NR>l && !/^  /{print NR-1; exit} END{if(NR>=l) print NR}' "$F" | head -1)
  k=$((E - L + 1))
  sed -n "${L},${E}p" "$F" >> "$O" && sed -i "${L},${E}d" "$F"
  m=$(wc -l < "$F"); [ "$m" -eq $((n - k)) ] || { echo "report_rotate: line count $n -> $m, expected $((n - k)) — stopped" >&2; exit 1; }
  n=$m
done
