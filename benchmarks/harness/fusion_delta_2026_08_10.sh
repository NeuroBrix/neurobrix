#!/bin/bash
# fusion_vertical delta campaign — N=5 paired ON/OFF on the 2026_08_08_ref
# reference rows, BOTH engines, both arms at CURRENT SOURCE (the kernel
# edit invalidated the autotune disk cache, so comparing against the
# 08-08 walls would mix config-flip noise into the delta — R16 doctrine,
# see docs/internal/optimization_engine_scoping.md Phase 3).
#
# OFF arm extras: vendor voice re-measure (fixed cell — ref_audio +
# hard no-wav gate) and the new R29 answer-text artifacts for VLM rows.
# TinyLlama is the iteration probe only — never a reported number.
#
# Idempotent: run_bench skips cells with an existing ok JSON; a
# power-loss resume re-runs this script verbatim (flightrec wraps it).
set -uo pipefail
cd /home/mlops/NeuroBrix_System
RB="python3 benchmarks/harness/run_bench.py --repetitions 5"
OFF=2026_08_10_fusion_off
ON=2026_08_10_fusion_on

echo "=== OFF arm start $(date -Is)"
# Pinned rows in parallel, one per 32G GPU (parallel-GPUs doctrine).
$RB --row omni_minicpmo_voice --columns vendor_transformers,neurobrix_pytorch,neurobrix_triton --gpu 2 --date $OFF > benchmarks/results/logs_fusion_off_voice.log 2>&1 &
P1=$!
$RB --row vlm_glm41v --columns neurobrix_pytorch,neurobrix_triton --gpu 3 --date $OFF > benchmarks/results/logs_fusion_off_glm.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "OFF pinned rows rc=$R1/$R2 $(date -Is)"
# Machine rows own the whole rig — strictly sequential.
$RB --row omni_ming_t2i --config machine --columns neurobrix_pytorch,neurobrix_triton --gpu 2 --date $OFF > benchmarks/results/logs_fusion_off_ming.log 2>&1
echo "OFF ming rc=$? $(date -Is)"
$RB --row vlm_qwen3vl --config machine --columns neurobrix_pytorch,neurobrix_triton --gpu 2 --date $OFF > benchmarks/results/logs_fusion_off_qwen3vl.log 2>&1
echo "OFF qwen3vl rc=$? $(date -Is)"

echo "=== ON arm start $(date -Is)"
export NBX_OPTIM_FUSION_VERTICAL=1
$RB --row omni_minicpmo_voice --columns neurobrix_pytorch,neurobrix_triton --gpu 2 --date $ON > benchmarks/results/logs_fusion_on_voice.log 2>&1 &
P1=$!
$RB --row vlm_glm41v --columns neurobrix_pytorch,neurobrix_triton --gpu 3 --date $ON > benchmarks/results/logs_fusion_on_glm.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "ON pinned rows rc=$R1/$R2 $(date -Is)"
$RB --row omni_ming_t2i --config machine --columns neurobrix_pytorch,neurobrix_triton --gpu 2 --date $ON > benchmarks/results/logs_fusion_on_ming.log 2>&1
echo "ON ming rc=$? $(date -Is)"
$RB --row vlm_qwen3vl --config machine --columns neurobrix_pytorch,neurobrix_triton --gpu 2 --date $ON > benchmarks/results/logs_fusion_on_qwen3vl.log 2>&1
echo "ON qwen3vl rc=$? $(date -Is)"

echo "=== activation evidence (ON arm daemon logs must show [Optim] lines)"
grep -l "fusion_vertical" benchmarks/results/$ON/server_*.log 2>/dev/null | head
grep -h "\[Optim\] fusion_vertical" benchmarks/results/$ON/server_*.log 2>/dev/null | sort | uniq -c | head -20

echo "=== ON-vs-OFF media byte gate (real-row end-to-end)"
for f in validation_outputs/bench_reference_$OFF/*/*.wav validation_outputs/bench_reference_$OFF/*/*.png; do
  [ -f "$f" ] || continue
  base=${f#validation_outputs/bench_reference_$OFF/}
  onf="validation_outputs/bench_reference_$ON/$base"
  case "$base" in vendor_*|*/vendor_*) continue ;; esac
  if [ -f "$onf" ]; then
    if cmp -s "$f" "$onf"; then echo "BYTE-IDENTICAL: $base"; else echo "BYTE-DIFF: $base"; fi
  fi
done
echo "CAMPAIGN DONE $(date -Is)"
