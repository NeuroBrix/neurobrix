#!/bin/bash
# Étape 1 — sequential Sana 4Kpx with parity instrumentation
# Mode: PyTorch eager op-by-op (NativeATenDispatcher)
# Backend: torch + cuDNN/cuBLAS + CachingAllocator
# Reference oracle: if this PASSes, the same DAG fits a single 32 GB V100
# at the torch live watermark — meaning triton_sequential's 26 GB live at
# OOM is NBX accounting/lifecycle, not structural sub-graph pressure.

set +e
cd /home/mlops/NeuroBrix_System

OUTDIR=/home/mlops/NeuroBrix_System/validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05
mkdir -p "$OUTDIR"

export NBX_NATIVE_LIVE_DUMP_EVERY=10
# CUDA_VISIBLE_DEVICES not set — let Prism auto-route to a 32 GB V100
# (cuda:2 or cuda:3); filtering breaks Prism's device-ordinal mapping.

START=$(date +%s)
echo "[START] $(date -u)" > "$OUTDIR/etape1_sequential.log"

/home/mlops/ml/venv/bin/neurobrix run \
  --model Sana_1600M_4Kpx_BF16 \
  --sequential \
  --prompt "a red apple" \
  --steps 12 \
  --output "$OUTDIR/etape1_sequential.png" \
  >> "$OUTDIR/etape1_sequential.log" 2>&1

EXIT=$?
END=$(date +%s)
echo "[END] $(date -u)" >> "$OUTDIR/etape1_sequential.log"
echo "[EXIT=$EXIT WALL=$((END - START))s]" >> "$OUTDIR/etape1_sequential.log"
echo "EXIT=$EXIT WALL=$((END - START))s"
