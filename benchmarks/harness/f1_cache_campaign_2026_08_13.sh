#!/bin/bash
# F1 step-cache three-arm campaign (drift discipline, scoping doc "F1").
# Arms per row: OFF-total / replay / replay+cache(thr sweep x2).
# Wan adds the diffusers FBC-ON arm (fairness clause 6 — Wan IS
# CacheMixin at the 0.35.2 pin; Sana is NOT, documented).
# Artifacts land in the R29 media tree via the bench harness; the
# cache arms' PNGs/frames are the eye's evidence (INDEX built after).
set -uo pipefail
cd /home/mlops/NeuroBrix_System
RB="python3 benchmarks/harness/run_bench.py --repetitions 5 --gpu 2"

run_row () {  # $1 row  $2 config flag (e.g. "--config machine" or "")
  row=$1; shift
  cfgflag="$*"
  echo "== $row OFF-total $(date -Is)"
  $RB --row $row $cfgflag --columns neurobrix_triton,neurobrix_pytorch --date 2026_08_13_f1_off > benchmarks/results/logs_f1_off_$row.log 2>&1
  echo "rc=$?"
  echo "== $row replay $(date -Is)"
  NBX_TRITON_REPLAY=1 $RB --row $row $cfgflag --columns neurobrix_triton,neurobrix_pytorch --date 2026_08_13_f1_replay > benchmarks/results/logs_f1_replay_$row.log 2>&1
  echo "rc=$?"
  for thr in 0.15 0.25; do
    echo "== $row replay+cache thr=$thr $(date -Is)"
    NBX_TRITON_REPLAY=1 NBX_STEP_CACHE_THRESHOLD=$thr NBX_STEP_CACHE_MAX_SKIPS=3 \
      $RB --row $row $cfgflag --columns neurobrix_triton,neurobrix_pytorch --date 2026_08_13_f1_cache$thr > benchmarks/results/logs_f1_cache${thr}_$row.log 2>&1
    echo "rc=$?"
  done
}

run_row image_diffusion_sana1024
run_row omni_ming_t2i --config machine
run_row video_wan13b_t2v

echo "== wan diffusers FBC-ON arm (fairness) $(date -Is)"
$RB --row video_wan13b_t2v --columns diffusers --date 2026_08_13_f1_dfbc > benchmarks/results/logs_f1_dfbc_wan.log 2>&1
echo "rc=$? (NOTE: FBC enablement inside diffusers_cell requires the cell to honor a cache flag — check the cell; if absent, this arm re-ran plain and the fairness note documents it)"

echo "== [StepCache] evidence:"
grep -h "StepCache" benchmarks/results/2026_08_13_f1_cache*/server_*.log 2>/dev/null | sort | uniq -c | head
echo "F1 CAMPAIGN DONE $(date -Is)"
