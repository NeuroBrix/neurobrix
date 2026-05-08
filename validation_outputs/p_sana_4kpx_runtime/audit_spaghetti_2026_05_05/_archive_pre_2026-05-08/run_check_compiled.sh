#!/bin/bash
# Sanity check: compiled mode Sana 4Kpx after Fix B implementation
# Goal: verify no regression — compiled mode SHOULD still produce a
# coherent PNG. If the in-place add interceptor crashes on torch.Tensor
# inputs, this surfaces immediately.

set +e
cd /home/mlops/NeuroBrix_System

OUTDIR=/home/mlops/NeuroBrix_System/validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05
mkdir -p "$OUTDIR"

START=$(date +%s)
echo "[START] $(date -u)" > "$OUTDIR/check_compiled.log"

/home/mlops/ml/venv/bin/neurobrix run \
  --model Sana_1600M_4Kpx_BF16 \
  --compiled \
  --prompt "a red apple" \
  --steps 12 \
  --output "$OUTDIR/check_compiled.png" \
  >> "$OUTDIR/check_compiled.log" 2>&1

EXIT=$?
END=$(date +%s)
echo "[END] $(date -u)" >> "$OUTDIR/check_compiled.log"
echo "[EXIT=$EXIT WALL=$((END - START))s]" >> "$OUTDIR/check_compiled.log"
echo "EXIT=$EXIT WALL=$((END - START))s"
