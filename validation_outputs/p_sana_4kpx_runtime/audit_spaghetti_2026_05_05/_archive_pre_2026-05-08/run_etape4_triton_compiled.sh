#!/bin/bash
# Étape 4 — triton compiled Sana 4Kpx with the 3 fixes from Étape 1+2
# Mode: TritonSequence (NBXTensor + Triton kernels + compiled hot loop)
# The bias broadcast fix and rms_norm import fix are SHARED with
# triton_sequential (via OpLevelTilingEngine interceptors), so this
# is the test of whether those fixes ALSO unblock triton compiled
# (the production target).

set +e
cd /home/mlops/NeuroBrix_System

OUTDIR=/home/mlops/NeuroBrix_System/validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05
mkdir -p "$OUTDIR"

export NBX_LIVE_DUMP_EVERY=10
export NBX_LIVE_DUMP_ON_OOM=1
export NBX_ALLOC_POOL=1
# Trace every NBX malloc/free with caller's tid + stack site.
# At OOM, post-process this tsv to find data_ptr that's allocated
# but never freed = the orphan source.
# Re-enable in-place add (Fix B) — combined with Fix F2 (pixel_shuffle
# broadcast-aware) which should eliminate the 8 GiB clone leak.

START=$(date +%s)
echo "[START] $(date -u)" > "$OUTDIR/etape4_triton_compiled.log"

/home/mlops/ml/venv/bin/neurobrix run \
  --model Sana_1600M_4Kpx_BF16 \
  --triton \
  --prompt "a red apple" \
  --steps 12 \
  --output "$OUTDIR/etape4_triton_compiled.png" \
  >> "$OUTDIR/etape4_triton_compiled.log" 2>&1

EXIT=$?
END=$(date +%s)
echo "[END] $(date -u)" >> "$OUTDIR/etape4_triton_compiled.log"
echo "[EXIT=$EXIT WALL=$((END - START))s]" >> "$OUTDIR/etape4_triton_compiled.log"
echo "EXIT=$EXIT WALL=$((END - START))s"
