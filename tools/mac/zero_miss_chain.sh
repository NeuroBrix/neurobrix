#!/bin/bash
# The Mac's matrix chain (was campagnes/2026_09_22_apple/scripts/zero_miss_chain_v12.sh; history v8-v12 in the run register's
# journals). Every path from the machine's atelier.json (docs/internal/_atelier.md §1, §4) through tools/atelier.py: the
# engine python, the local container cache, the NAS cache (its mount, its host and its remote path), the peer channel; the
# chain's own output OUT must lie under a writable root (refused by name otherwise). A model whose plan does not fit now goes
# to a replay queue replayed smallest container first; a correctness cell is never withheld for host memory (Hocine
# 2026-10-12 22:45); a local copy is checked against the NAS manifest and removed after the model's cells.
# usage: [HT=<harness tree>] zero_miss_chain.sh <tree> <label> <out> <marker regex> <marker file> <model[:modes]...>
W=$1; LABEL=$2; OUT=$3; MARK=$4; MARKFILE=$5; shift 5
HERE=$(cd "$(dirname "$0")/.." && pwd); AT() { python3 "$HERE/atelier.py" "$@" || exit 2; }
OUT=$(AT writable "$OUT"); V=${PYV:-$(AT python engine)}; NAS=$(AT root nas_cache); LC=$(AT root cache); PEER=$(AT root peer)
S=$OUT/chain.out; mkdir -p $OUT/harness $LC
RD=$OUT/readers.$$; mkdir -p $RD; SSHO="ssh -o BatchMode=yes $(AT root nas_host)"; AG=$(AT root nas_agents_remote); RC=$(AT root nas_cache_remote)
export TOOLCHAINS=Metal PYTHONUNBUFFERED=1
HT=${HT:-$W}; DEFER_S=${DEFER_S:-1800}; REPLAY_MAX_S=${REPLAY_MAX_S:-7200}; Q=$OUT/replay_queue.$$
free_gb() { df -g "$LC" | tail -1 | awk '{print $4}'; }
local_gb() { local k=$(du -sk "$LC/$1" 2>/dev/null | cut -f1); echo $(( ${k:-0} / 1048576 )); }
size_gb() { $SSHO "du -sb $RC/$1" | awk '{printf "%d", $1/1073741824 + 1}'; }   # bytes, GNU du on optimus
precheck() { NEUROBRIX_CACHE=$NAS $V $HERE/mac/precheck_keys.py $W "$1" "$2" >> $S 2>&1; }
reader_sync() {   # the reader file lists what $RD holds now; no file when it holds nothing
  if [ -n "$(ls $RD 2>/dev/null)" ]; then
    { echo "host mac pid $$ ($LABEL)"; for f in $RD/*; do echo "reads $RC/$(basename $f): $(cat $f)"; done; } | $SSHO "cat > $AG/cache.readers/mac-$$"
  else $SSHO "rm -f $AG/cache.readers/mac-$$"; fi; }
reader_on() {   # $1 model $2 purpose — waits while a writer holds the lock, then registers
  local n=0
  while $SSHO "test -e $AG/cache.lock"; do
    [ $((n % 10)) = 0 ] && echo "CACHE LOCK held $(date '+%F %T'): $1 ($2) waits — $($SSHO "head -c 300 $AG/cache.lock" | tr '\n' ' ')" >> $S
    n=$((n+1)); sleep 60; done
  echo "$2, since $(date '+%F %T %Z')" > $RD/$1; reader_sync; }
reader_off() { rm -f $RD/$1; reader_sync; }
trap 'rm -rf $RD; $SSHO "rm -f $AG/cache.readers/mac-$$" 2>/dev/null' EXIT
copy() { local t0=$(date +%s) b0=$(du -sk "$LC/$1" 2>/dev/null | cut -f1); b0=${b0:-0}
  reader_on $1 "copy over SSH"
  rsync -a --delete -e "ssh -o BatchMode=yes" "$(AT root nas_host):$RC/$1/" "$LC/$1/" 2>>$S
  a=$(shasum -a 256 $LC/$1/manifest.json | cut -c1-16); b=$($SSHO "sha256sum $RC/$1/manifest.json" | cut -c1-16); fa=$(find $LC/$1 -type f | wc -l | tr -d ' '); fb=$($SSHO "find $RC/$1 -type f | wc -l" | tr -d ' ')
  reader_off $1
  if [ "$a" != "$b" ] || [ "$fa" != "$fb" ]; then echo "COPY $1 REFUSED $(date +%T): manifest $a vs $b, files $fa vs $fb" >> $S; return 1; fi
  local t1=$(date +%s) b1=$(du -sk "$LC/$1" | cut -f1)
  echo "COPY $1 OK $(date '+%F %T') over SSH from optimus, manifest $a files $fa, $(( (b1-b0)/1024 )) MiB in $((t1-t0)) s ($(( (b1-b0)/1024/(t1-t0+1) )) MiB/s), disk $(free_gb)G free" >> $S; }
harness_run() {   # $1 model $2 modes(csv) $3 cache root — the shared harness under the guard
  echo "HARNESS $1 [$2] START $(date '+%F %T') cache $3" >> $S
  ( cd $W && NEUROBRIX_CACHE=$3 $V $HT/tools/regression_matrix.py run --models "$1" --modes "$2" --gpu 0 --out $OUT/harness --timeout ${CELL_TIMEOUT_S:-2400} --src $W/src --defer-after $DEFER_S >> $OUT/harness/run.log 2>&1 ) & local cp=$!
  while kill -0 $cp 2>/dev/null; do sw=$(sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%.0f",v}'); disk=$(df -k "$LC" | tail -1 | awk '{printf "%.1f",$4*1024/1e9}')
    if [ "$sw" -gt 10000 ] || [ "$(echo "$disk < 15" | bc)" = 1 ]; then echo "GUARD KILLED $1 $(date +%T) swap=${sw}MB disk=${disk}GB" >> $S; pkill -TERM -f "neurobrix run --model $1 "; sleep 5; pkill -KILL -f "neurobrix run --model $1 "; break; fi; sleep 5
  done; wait $cp 2>/dev/null; local rc=$?
  python3 - "$OUT/harness" "$1" >> $S <<'PY'
import json, sys, glob
rows=[json.loads(l) for f in glob.glob(sys.argv[1]+"/rows_card*.jsonl") for l in open(f) if l.strip()]
for r in rows:
    if r.get("model")==sys.argv[2]: print(f"CELL {r['model']} {r['mode']} END rc={r.get('rc')} {r.get('wall_s')}s zero_miss={'PASS' if r.get('rc')==0 and not r.get('misses') else ('MISS' if r.get('misses') else 'ERROR')} misses={r.get('misses')} first={str(r.get('first_missing_key'))[:160]} err={str(r.get('error') or r.get('first_error') or '')[:200]} host_peak={r.get('host_peak_rss')}")
PY
  echo "HARNESS $1 END rc=$rc $(date '+%F %T')" >> $S
  if [ $rc = 75 ]; then echo "DEFERRED-MEMORY $1 $(date +%T): the host could not take its cell for ${DEFER_S} s (harness above) — to the replay queue" >> $S; echo "$1:$2" >> $Q; echo "$1:$2" >> $OUT/withheld_memory; fi; }

# 1. after the marker, never beside a certification or another cell
until grep -qE "$MARK" "$MARKFILE" 2>/dev/null; do sleep 30; done
while pgrep -f "neurobrix autotune certify|neurobrix run --model" >/dev/null; do sleep 10; done; sleep 3
# 2. the tree carries what the cells need, checked here (a failed merge stops the chain)
# 2026-10-14: main's history was rewritten (trailers, Dell 20:13); each required fix is checked by its old SHA OR its new one
# from the Dell's map (_agents/peer/trailers_sha_map.tsv: old<TAB>new).
SMAP=$PEER/trailers_sha_map.tsv
has() { git -C $W merge-base --is-ancestor $1 HEAD 2>/dev/null && return 0; local n=$(grep -m1 "^$1" $SMAP 2>/dev/null | cut -f2); [ -n "$n" ] && git -C $W merge-base --is-ancestor $n HEAD 2>/dev/null; }
for c in 5f20f29f 546fedcb 67dfafa5 b22d7f2a_or_a941135a 9762fedb; do
  if [ $c = b22d7f2a_or_a941135a ]; then has a941135a || has b22d7f2a || { echo "== $LABEL REFUSED $(date '+%F %T'): the Metal wrap fix is not in $(git -C $W rev-parse --short HEAD)" >> $S; exit 1; }
  else has $c || { echo "== $LABEL REFUSED $(date '+%F %T'): $c is not in the tree under test $(git -C $W rev-parse --short HEAD)" >> $S; exit 1; }; fi; done
echo "== $LABEL v11 START $(date '+%F %T') tree $(git -C $W rev-parse --short HEAD): $*" >> $S

pass_over() {
specs=("$@"); N=${#specs[@]}; pre_pid=""; pre_model=""
for ((i=0; i<N; i++)); do
  m=${specs[$i]%%:*}; modes=${specs[$i]#*:}; [ "$modes" = "${specs[$i]}" ] && modes="triton"
  root=$LC
  if [ "$pre_model" != "$m" ]; then
    precheck "$m" "$modes"; pc=$?
    # v12 (Hocine 2026-10-12 22:45): a CORRECTNESS cell is never withheld for host memory; it runs at the plan Prism
    # chooses, and a cell that cannot run with the memory there is a Prism/Tiling defect named at the source.
    if [ $pc = 2 ]; then echo "MEMORY-LOW $m $(date +%T): above the memory available now — run anyway at Prism's plan (correctness cell)" >> $S; pc=0; fi
    if [ $pc != 0 ]; then echo "WITHHELD $m $(date +%T): keys missing under the running generator or no plan (above) — no copy" >> $S; echo "${specs[$i]}" >> $OUT/withheld_keys; continue; fi
    # what is still to copy is the container minus what is already here (2026-10-04 07:26: a complete local copy
    # was ignored, the free disk AFTER it compared with the whole size, and the cell ran from the mount)
    need=$(size_gb $m); have=$(free_gb); here=$(local_gb $m)
    if [ $((have + here - need)) -ge 15 ]; then echo "COPY $m START $(date '+%F %T') disk ${have}G free size ${need}G, ${here}G already local" >> $S; copy $m || { rm -rf "${LC:?}/${m:?}"; continue; }
    else root=$NAS; echo "IN PLACE $m $(date '+%F %T'): ${need}G (${here}G already local) against ${have}G free would leave $((have+here-need))G, under the 15 GB floor — run from the mount, nothing copied" >> $S; fi
  else wait $pre_pid; echo "PREFETCH $m joined $(date +%T)" >> $S; [ -f "$LC/$m/manifest.json" ] || { echo "PREFETCH $m left no copy — skipped" >> $S; continue; }; fi
  pre_pid=""; pre_model=""
  if [ $((i+1)) -lt $N ] && [ $root = $LC ]; then nx=${specs[$((i+1))]%%:*}; nxm=${specs[$((i+1))]#*:}; [ "$nxm" = "${specs[$((i+1))]}" ] && nxm="triton"
    need=$(size_gb $nx); have=$(free_gb); here=$(local_gb $nx)
    if precheck "$nx" "$nxm"; then
      if [ $((have + here - need)) -ge 15 ]; then echo "PREFETCH $nx START $(date '+%F %T') (${need}G, ${have}G free) beside $m's cells" >> $S; ( copy $nx ) & pre_pid=$!; pre_model=$nx
      else echo "WALL disk: $nx (${need}G) not prefetched beside $m — ${have}G free would leave $((have-need))G under the 15 GB floor (decided again at its turn)" >> $S; fi
    else echo "PREFETCH $nx SKIPPED $(date +%T): its shadow does not pass now (decided again at its turn)" >> $S; fi
  fi
  if [ $root = $NAS ]; then reader_on $m "in-place cell from the mount"; t_start=$(date '+%F %T'); fi
  harness_run $m $modes $root
  if [ $root = $NAS ]; then
    echo "IN PLACE $m: $($SSHO "find $RC/$m -newermt '$t_start' | wc -l" | tr -d ' ') path(s) of the shared directory changed during the cell (0 expected)" >> $S; reader_off $m
  else rm -rf "${LC:?}/${m:?}"; echo "DELETED local copy $m $(date +%T) disk $(free_gb)G free" >> $S; fi
done
}
rm -f $Q; pass_over "$@"
echo "== $LABEL first pass END $(date '+%F %T'); replay queue: $(tr '\n' ' ' < $Q 2>/dev/null); withheld for keys: $(tr '\n' ' ' < $OUT/withheld_keys 2>/dev/null)" >> $S
t0=$(date +%s); round=0
while [ -s $Q ]; do
  round=$((round+1)); prev=$(sort $Q | tr '\n' ' ')
  order=$(sort -u $Q | while read sp; do echo "$(size_gb ${sp%%:*}) $sp"; done | sort -n | cut -d' ' -f2 | tr '\n' ' ')
  rm -f $Q; echo "REPLAY round $round $(date '+%F %T'), smallest container first: $order" >> $S
  pass_over $order
  [ -s $Q ] || break
  if [ "$(sort $Q | tr '\n' ' ')" = "$prev" ]; then
    if [ $(( $(date +%s) - t0 )) -ge $REPLAY_MAX_S ]; then echo "STILL DEFERRED $(date '+%F %T') after ${REPLAY_MAX_S} s of replay: $(tr '\n' ' ' < $Q)" >> $S; break; fi
    echo "RETRY $(date +%T): none of the queue fits now; next round in 300 s" >> $S; sleep 300
  else t0=$(date +%s)
  fi
done
echo "$LABEL END $(date '+%F %T')" >> $S
