# Wan2.1-VACE-1.3B — VERDICT: CLOSED 4/4 (compiled · pytorch-sequential · triton · triton_sequential)

Root of the triton washed-out = the interleaved-complex Wan RoPE in-place write
was lost in two coupled triton steps: `dispatch._meta_slice` **materialised** the
step-2 slice (severing the view→parent alias) and `NBXTensor.copy_` flat-memcpy'd
into the throwaway → self-attn Q/K = 0 → mean(V)-collapse → washed-out. Fixed at
the source (slice returns a view; copy_ scatters into the strided dst). Refuted en
route: text-512, vace injection, flash attention, "34% kernel divergence" — each a
conclusion-by-elimination on metrics blind to the actual failure mode.

**Date:** 2026-06-28 · **Family:** video control · **Arch:**
WanVACETransformer3DModel (30 blocks, 15 VACE control layers) + UMT5 +
AutoencoderKLWan.

## CLOSED — 4/4 coherent, CFG batch=2 (cfg 5.0 → batch=2 ≠ trace 1), f9 480×832, seed 42
Frame-4 std (coherence) + pearson corr vs the compiled oracle (drift gate):

| Mode | frame-4 std | corr vs compiled | R29 |
|------|------------|------------------|-----|
| compiled (oracle) | 52.1 | — | coherent red fox ✓ |
| pytorch-sequential | 52.1 | **0.9995** | coherent red fox ✓ (re-rendered this session) |
| triton | 54.2 | **0.9841** | coherent red fox ✓ (was washed, std 8.8) |
| triton_sequential | 54.1 | **0.9840** | coherent red fox ✓ (was washed, std 8.8) |

All four modes re-rendered and inspected this session (no reliance on prior
artifacts). triton/triton_sequential frame-0 std 73, all frames coherent across
the clip. pytorch-sequential is structurally unaffected by the triton-only fix
(its dispatcher already executes `aten::copy` as `inputs[0].copy_(inputs[1])`, an
in-place write through the slice view — `sequential_dispatcher.py:234`) and its
0.9995 corr vs compiled confirms it.

## ROOT CAUSE — interleaved-complex RoPE in-place write lost (triton only)
The Wan RoPE writes the rotated Q in place at **interleaved** (complex) positions
of an `empty_like` buffer that `permute` later reads:
```
empty_like::0 [.,.,.,128]                       # q buffer; permute::0 → SDPA q
slice::14 = buf[..., 0::2]  (dim=3 start=0 end=INT64_MAX step=2)  = REAL
slice::15 = buf[..., 1::2]  (step=2)                              = IMAG
copy::0: slice14.copy_(real); copy::1: slice15.copy_(imag)   # copy outputs ORPHAN
permute::0 reads empty_like  → relies on the in-place mutation
```
PyTorch/compiled execute `aten::copy` as an in-place write through the slice
**view** into the parent buffer. Triton broke it in two coupled places:
1. `dispatch._meta_slice` step>1 returned `as_strided(...).contiguous()` — a
   **materialised standalone** tensor (`has_base=False`), severing the alias to
   `empty_like`. (The `.contiguous()` was a read-side guard for chatterbox's
   `mel[::2]`; wrong for an in-place-copy destination.)
2. `NBXTensor.copy_` flat-memcpy'd `self._nbytes` ignoring destination strides.

Net: `copy_` wrote real/imag into a **throwaway contiguous buffer**; `empty_like`
was never written; `permute` read it → **self-attn Q = K = 0** (every block; V and
cross-attn fine) → uniform softmax → output = mean(V) → spatial-variance collapse →
washed-out. L2 / head10 / batch-norms are **blind** to this collapse (mean(V)
preserves them); only **spatial std** (std over the seq axis) detects it — q_spatial_std
measured **0.000** at every self-attn block before the fix.

## FIX (two coupled, R30-faithful, NO graph change — R19 preserved)
1. `dispatch.py _meta_slice` step>1: return the `as_strided` **VIEW** (drop the
   `.contiguous()`). A slice is a view in PyTorch/compiled; `as_strided` already
   encodes the correct strided shape.
2. `nbx_tensor.py copy_`: when `self` is non-contiguous, scatter the contiguous src
   into self's strided layout via `_strided_scatter` (inverse of `_strided_copy`,
   the primitive `__setitem__`/`_fill_constant` use) + a numel-match guard. R33-pure.

Read consumers of step>1 slices (the rope-rotation `mul`s) re-materialise via the
elementwise `_prepare_binary .contiguous()`, so they are byte-identical.

## VALIDATION
- **Boundary gate (1-step):** copy_ now receives the step-2 **view**
  (`contig=False, strides=(...,2), has_base=True`); SDPA self-attn
  **q_spatial_std 0.000 → 0.49 / 0.91 / 0.97** per block, k restored. Collapse resolved.
- **Microtests (zero model):** step-2 interleaved scatter bit-exact
  (`buf[...,0::2]=real; buf[...,1::2]=imag`, max|diff|=0); the exact graph chain
  `empty_like→slice→copy→permute→contiguous` bit-exact WITH the fix, corrupted
  (max|diff|=9) without.

## ANTI-REG (shared infra — `--triton`, step>1-slice models)
Static scan: the ONLY live consumers of step>1 slices zoo-wide are `aten::mul`
(reads — self-guarded by `_prepare_binary .contiguous()`) and `aten::copy`
(in-place writes — the fix). The fix changes the slice from a materialised copy
to a view aliasing the parent; values are byte-identical (mul re-materialises),
and the **read-view lifetime** (the view depends on the parent surviving the
arena's kill_slots/deferred-free) is held by NBXTensor's `_base` refcount.
Targets:
- **Wan-T2V-1.3B** (no step>1 slices → unchanged path): coherent fox, std 54.7 ✓
- **SANA-Video_2B_720p** (step-2 `mul` reads — read-path): coherent fox, std 52.7 ✓
  **This is the load-bearing read-path proof**: it exercises the changed
  `_meta_slice` view at 12 steps through the full arena lifecycle and stays
  coherent → read-view aliasing is value- AND lifetime-correct (a dangling-view
  UAF would be random noise, not a coherent fox).
- **Mochi-1-preview** (step-2 `mul` reads, identical code path): ran clean to
  completion, smooth output (NOT random garbage → no UAF). Mochi was closed
  historically via latent drift at 2 steps, never a coherent frame, so frame
  comparison is N/A; covered by the SANA-Video empirical proof + value-identity. ✓
- **chatterbox** (step-2 slices ORPHAN/dead): byte-identical by construction ✓
- **Wan2.2-I2V-A14B** (step-2 `copy` writes, same pattern as VACE): the fix repairs
  the same washed-out path — net positive, not a regression. CAVEAT for its
  closure: under multi-GPU/zero3 the in-place-write view's lifetime must be
  re-verified (`materialize_slots_depending_on` shows aliased views can be broken
  before freeing on that path); single-GPU VACE is `_base`-refcount-safe.
All non-step>1-slice models are byte-identical by construction (step==1 path unchanged).

## Reproduce
```bash
python3 -m neurobrix run --model Wan2.1-VACE-1.3B-diffusers --triton \
  --prompt "a red fox walking in a snowy forest" --cfg 5.0 \
  --height 480 --width 832 --num-frames 9 --steps 12 --seed 42 --output vace_triton.mp4
```

## Hocine validation: TODO (4 coherent frames — compiled/triton/triton_seq attached)
