#!/bin/bash
# THE WATCHDOG (the owner's rule, 2026-09-21 13:10): no job leaves supervision unless it runs
# under a watchdog proven to kill it. `timeout --kill-after=60 <cap> -- <job>`: TERM at the cap,
# KILL sixty seconds later. Proven on this rack the same day: `timeout --kill-after=2 2 sleep 30`
# → rc 124; a child that traps TERM (`trap "" TERM; sleep 30`) → KILLed, rc 137. Every detached
# launch of this campaign from now on goes through this wrapper with an explicit cap.
#   usage: wd.sh <cap seconds> <label> -- <command...>
cap=$1; label=$2; shift 3
H=/home/mlops/nbx/campaigns/2026_09_20_hub_pass
timeout --kill-after=60 "$cap" "$@"; rc=$?
[ $rc -eq 124 ] || [ $rc -eq 137 ] && echo "== WATCHDOG killed '$label' at ${cap}s (rc=$rc) $(date -u +%H:%M:%S)" >> $H/RUN.md
exit $rc
