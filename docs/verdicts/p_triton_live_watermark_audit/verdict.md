# P-TRITON-LIVE-WATERMARK-AUDIT — progress + residual blockers (2026-05-14)

## Outcome

- **L1 instrumentation** (commit `993181f`): NBX_LIVE_WATERMARK_TRACE
  JSONL stream captures `nbx_live`, `nbx_pool_cached`, `driver_free`,
  `driver_used`, `untracked = driver_used - nbx_live - nbx_pool_cached`
  every NBX_LIVE_WATERMARK_EVERY ops. Confirmed ~647 MB untracked
  baseline on Sana 4Kpx 16g triton, mostly Triton kernel cache.

- **L2 per-slot LIVE_DUMP_SLOT** (commit `993181f` extension):
  identifies surviving intermediate slots at OOM with tid, producing
  op_uid, last_use, current_op_idx. Discriminates "liveness bug"
  (last_use ≤ current_op_idx) from "legitimate memory pressure"
  (last_use > current_op_idx).

- **L4 mode-aware chain wrapper** (commits `f8a8ad8`, `f997479`):
  `band_streamed_chain_nbx` (143-line R33-pure mirror of
  `band_streamed_chain_torch`) closes the R30 gap that commit
  `c9d2581` left open. Mode-aware dispatcher in `tiling_engine.py`
  routes the chain wrapper for triton/triton_sequential through
  the NBX variant. NBXTensor isinstance fix unblocks `_pending`
  population. Explicit `_set_device(t_base)` anchors the chain on
  the tensor's device.

  Post-L4 state on Sana 4Kpx 16g triton:
  - **4 chains succeed** (vae::add::66→69, 69→72, 72→75, 76→79)
  - 2 chains fail intermittently with
    "Pointer argument (at 0) cannot be accessed from Triton
    (cpu tensor?)" — sub-chantier P-TRITON-CHAIN-CPU-POINTER.
  - Eventual OOM at add::86 NOT a chain bug: the two surviving
    4 GiB tensors at LIVE_DUMP_SLOT are
    `aten.convolution::62::out_0` and `aten.pixel_shuffle::4::out_0`,
    both with last_use=current_op_idx=660 — they are the LEGITIMATE
    INPUTS of add::86 (the fork). 8 GiB inputs + 4 GiB output ≫
    13.5 GiB free → real overflow.

## Matrix impact

The chain wrapper landing is a **real architectural acquis**:
- R30 dualité chain wrapper restored across both backends.
- 4 of the ~7 VAE residual chains now stream band-by-band on
  triton 16g, reducing per-chain peak from 6.7 GiB to 1.6 GiB.

But the 16g triton cell remains ⏳ because:
- 2 chains still fail intermittently (cpu-tensor pointer error).
- add::86 fork allocates 4 GiB on top of 8 GiB live inputs → OOM.

Estimated work to close 16g triton:
- **P-TRITON-CHAIN-CPU-POINTER**: ~50-150 lines, identify and fix
  the device-sync race between chain N and chain N+1.
- **P-TRITON-FORK-INPLACE-ADD**: ~50-100 lines, ensure
  `plan.inplace_adds` interceptor fires for add::86 on triton mode
  so one 4 GiB input buffer is reused as the add output (compiled
  mode already does this via `add_inplace_nbx`).

## Reproduction

```bash
# L1 instrumented baseline (run to OOM, then read JSONL):
CUDA_VISIBLE_DEVICES=0 \
  NBX_LIVE_WATERMARK_TRACE=/tmp/watermark_16g_triton.jsonl \
  NBX_LIVE_WATERMARK_EVERY=20 \
  NBX_LIVE_DUMP_ON_OOM=1 \
  neurobrix run --model Sana_1600M_4Kpx_BF16 --prompt "a red apple" \
    --steps 1 --hardware v100-16g --triton \
    --output /tmp/sana_4kpx_16g_triton_diag.png

# Read JSONL trace:
python3 -c "
import json
recs = [json.loads(l) for l in open('/tmp/watermark_16g_triton.jsonl')]
print(f'{len(recs)} samples')
print(f'first untracked: {recs[0][\"untracked_mb\"]:.0f} MB')
print(f'last  untracked: {recs[-1][\"untracked_mb\"]:.0f} MB')
"

# Read OOM per-slot dump:
grep "LIVE_DUMP_SLOT" /tmp/sana_4kpx_16g_triton_diag.log
```

## Hocine validation: TODO

- Verify chain wrapper on 32g triton anti-régression still
  produces coherent red apple PNG (it must — the chain
  registration was previously gated to "compiled/sequential"
  modes; now it also fires on triton modes; this commit must not
  regress 32g triton).
- Open the two sub-chantiers in the next mandate cycle if the
  current session does not have budget.
