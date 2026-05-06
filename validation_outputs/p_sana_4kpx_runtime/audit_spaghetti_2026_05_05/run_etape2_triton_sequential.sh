#!/bin/bash
# Étape 2 — triton_sequential Sana 4Kpx with same per-op live tracking
# Mode: NBXTensor + Triton kernels op-by-op (TritonSequentialDispatcher)
# Backend: Triton + NBX raw cudaMalloc
# Diff vs Étape 1: 2 variables change (allocator + kernels).
# If Étape 1 PASS and Étape 2 FAIL, problem isolated to allocator/kernels
# (not the compiled fusion).

set +e
cd /home/mlops/NeuroBrix_System

OUTDIR=/home/mlops/NeuroBrix_System/validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05
mkdir -p "$OUTDIR"

export NBX_LIVE_DUMP_EVERY=10
export NBX_LIVE_DUMP_ON_OOM=1
export NBX_ALLOC_POOL=1
export NBX_GC_ON_OOM=1
# CUDA_VISIBLE_DEVICES not set — let Prism auto-route to a 32 GB V100
# (cuda:2 or cuda:3); filtering breaks Prism's device-ordinal mapping.

START=$(date +%s)
echo "[START] $(date -u)" > "$OUTDIR/etape2_triton_sequential.log"

/home/mlops/ml/venv/bin/neurobrix run \
  --model Sana_1600M_4Kpx_BF16 \
  --triton-sequential \
  --prompt "a red apple" \
  --steps 12 \
  --output "$OUTDIR/etape2_triton_sequential.png" \
  >> "$OUTDIR/etape2_triton_sequential.log" 2>&1

EXIT=$?
END=$(date +%s)
echo "[END] $(date -u)" >> "$OUTDIR/etape2_triton_sequential.log"
echo "[EXIT=$EXIT WALL=$((END - START))s]" >> "$OUTDIR/etape2_triton_sequential.log"
echo "EXIT=$EXIT WALL=$((END - START))s"
