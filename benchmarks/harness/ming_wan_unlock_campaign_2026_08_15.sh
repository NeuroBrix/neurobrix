#!/bin/bash
# Ming double-unlock + Wan warm-fix campaign (drift discipline).
# Ming row (machine config): OFF / replay / replay+cache(0.15, 0.25)
#   — the device-scalar path made the denoiser bucket replay-eligible;
#   the image_gen leg now carries the F1 cache brick.
# Wan row (pinned gpu2): same four arms — P-WARM-TRITON-VIDEO fixed
#   (lazy_sequential CPU-staged input placement), triton column un-FAILs.
# Wan fairness arm: diffusers FirstBlockCache ON at their default
#   threshold 0.05 (their best weapon, documented in pins — clause 6).
# All arms N=5 warm-daemon requests; artifacts in the R29 media tree.
set -uo pipefail
cd /home/mlops/NeuroBrix_System
RB="python3 benchmarks/harness/run_bench.py --repetitions 5 --gpu 2"

run_arms () {  # $1 row ; remaining args = config flag
  row=$1; shift
  cfgflag="$*"
  echo "== $row OFF $(date -Is)"
  $RB --row $row $cfgflag --columns neurobrix_triton --date 2026_08_15_unlock_off > benchmarks/results/logs_unlock_off_$row.log 2>&1
  echo "rc=$?"
  echo "== $row replay $(date -Is)"
  NBX_TRITON_REPLAY=1 $RB --row $row $cfgflag --columns neurobrix_triton --date 2026_08_15_unlock_replay > benchmarks/results/logs_unlock_replay_$row.log 2>&1
  echo "rc=$?"
  for thr in 0.15 0.25; do
    echo "== $row replay+cache thr=$thr $(date -Is)"
    NBX_TRITON_REPLAY=1 NBX_STEP_CACHE_THRESHOLD=$thr NBX_STEP_CACHE_MAX_SKIPS=3 \
      $RB --row $row $cfgflag --columns neurobrix_triton --date 2026_08_15_unlock_cache$thr > benchmarks/results/logs_unlock_cache${thr}_$row.log 2>&1
    echo "rc=$?"
  done
}

run_arms omni_ming_t2i --config machine
run_arms video_wan13b_t2v

echo "== wan diffusers FBC-ON fairness arm $(date -Is)"
BENCH_DIFFUSERS_FBC=0.05 $RB --row video_wan13b_t2v --columns diffusers --date 2026_08_15_unlock_dfbc > benchmarks/results/logs_unlock_dfbc_wan.log 2>&1
echo "rc=$?"

echo "== StepCache evidence:"
grep -rh "StepCache" benchmarks/results/2026_08_15_unlock_cache*/server_*.log 2>/dev/null | sort | uniq -c
echo "== Replay evidence:"
grep -rh "plan frozen\|VERIFIED byte-equal\|UNREPLAYABLE" benchmarks/results/2026_08_15_unlock_replay/server_*.log 2>/dev/null | sort | uniq -c | head
echo "CAMPAIGN DONE $(date -Is)"
