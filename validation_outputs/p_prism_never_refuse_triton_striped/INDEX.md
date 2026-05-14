# P-TRITON-VAE-16G-STRIPED — session 2026-05-14 partial

## Acquis this session

1. **JSONL dump infrastructure** (commit `da484ae`) — replaces the prior
   O(N²) read-then-append JSON dump with O(1) per-call JSONL append.
   Mandatory for any bit-diff investigation at the Sana 4Kpx scale
   (~3000 ops): the old format added ~25 min IO overhead per run.

2. **Bit-diff script** (commit `0b1bfa1`) — `scripts/diff_tiled_ops.py`
   reads two NBX_DUMP_TIDS dumps (legacy JSON or new JSONL), sorts by
   graph execution_order, identifies the first divergent op.

3. **32g triton baseline re-validated**: Sana 4Kpx `--triton` on
   v100-32g produces a coherent red apple in 1443.60 s
   (`32g_triton_nodump_redapple.png`). Confirms 32g triton works on
   the current code state — the divergence on 16g is genuine.

4. **Partial 16g triton dump captured**: 821 ops dumped before kill
   (`triton_cmp16g_partial_821ops.jsonl`). Mostly transformer-block
   ops (last entry: `aten.convolution::14 [2, 11200, 128, 128]` —
   FFN intermediate). VAE phase not reached.

## Why this chantier did not close in this session

Empirical timing for the bit-diff approach on this hardware (1×
V100 32g + 1× V100 16g, parallel dumps):
- 32g triton with dump filter on ~7 op_type prefixes: process at
  100% CPU for 35+ min with only ~810 records dumped, GPU util 0%
  (Triton JIT compile + caching contention).
- Same on 16g triton (parallel).
- Killed both after 35 min wall when progress stalled at ~810 lines.

Project memory had cited "Sana 4Kpx 32g triton historically 511 s"
which appears to have been BEFORE the current code state's Triton
cache invalidation. Empirically the run takes 1443 s without dump
and >35 min with dump.

The bit-diff capture is fundamentally bounded by 2× long triton
runs (~30-60 min wall each with dump filter active). Capturing
both 32g and 16g dumps then running the bit-diff to find the first
divergent op exceeded the session budget after multiple parallel
attempts.

## Inference from the partial data + code review

The depthwise conv tile-skip fix (commit `8af7848`, prior session)
removes 20 transformer `block.X.ffn.conv_depth` ops from `tiled_ops`
on 16g. Confirmed via `_detect_op_level_tiling_pairs` reading. So
the bug surface on 16g triton is NOT in those depthwise convs.

The chain wrapper on triton modes is now correctly skipped (commit
`c9d2581`). So no torch-vs-NBX type mismatch fires.

Remaining candidate root causes (priority order for next session):
1. **`tiled_conv2d_spatial` NBX path** (`_tiled_conv2d_spatial_nbx`) —
   activates on 16g for any VAE conv whose workspace estimator
   crosses the lower 16g threshold (e.g. early VAE convs at 1024²
   spatial with high channel count). On 32g threshold higher, runs
   native. Verify by listing `tiled_ops` for the VAE component on
   16g vs 32g.
2. **`tiled_rms_norm_spatial` triton path** — interceptor registered
   for VAE rms_norm but for NBXTensor the wrapper falls back to
   `nbx_rms_norm` (per `tiled_rms_norm_spatial:512+`). So no actual
   tiling happens for NBX. Less likely culprit.
3. **F2a `BroadcastClonePyroxy` pattern** — interaction with the
   tighter 16g memory state might break the broadcast aliasing
   if a downstream consumer races with the source tensor's release.

## Next session entry point

```bash
# Verify the cache is warm:
ls ~/.triton/cache | head -3
# Capture 32g triton dump (one at a time, no parallel — cache
# contention slows JIT compile):
CUDA_VISIBLE_DEVICES=2 NBX_DUMP_TIDS=/tmp/triton_ref32g.jsonl \
  NBX_DUMP_TIDS_FILTER="aten.add::,aten.convolution::,custom.rms_norm::" \
  neurobrix run --model Sana_1600M_4Kpx_BF16 --steps 1 \
  --hardware v100-32g --triton --output /tmp/ref32g.png > log32g 2>&1 &
# Wait ~25 min (cache hot, should be near 1443s baseline)
# Then sequentially:
CUDA_VISIBLE_DEVICES=0 NBX_DUMP_TIDS=/tmp/triton_cmp16g.jsonl \
  NBX_DUMP_TIDS_FILTER="aten.add::,aten.convolution::,custom.rms_norm::" \
  neurobrix run --model Sana_1600M_4Kpx_BF16 --steps 1 \
  --hardware v100-16g --triton --output /tmp/cmp16g.png > log16g 2>&1 &
# Wait ~30 min
# Then:
python3 scripts/diff_tiled_ops.py /tmp/triton_ref32g.jsonl \
  /tmp/triton_cmp16g.jsonl \
  ~/.neurobrix/cache/Sana_1600M_4Kpx_BF16/components/vae/graph.json
# Read the FIRST divergent op_uid and trace it back to its tile
# decision in `_detect_op_level_tiling_pairs`.
```

Narrower filter (`aten.add::,aten.convolution::,custom.rms_norm::`)
should cut total dump records ~3× vs the broader filter used this
session, reducing IO overhead.

## Artefacts in this directory

- `32g_triton_nodump_redapple.png` — confirmed coherent baseline,
  1443.6 s wall.
- `triton_cmp16g_partial_821ops.jsonl` — partial 16g triton dump
  (transformer-only, 821 records). Useful for next session as
  partial reference even though full VAE coverage is needed.
