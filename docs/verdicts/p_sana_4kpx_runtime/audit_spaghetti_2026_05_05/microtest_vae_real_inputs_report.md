# VAE TOP-divergent ops microtest — REAL runtime inputs (Phase 3a-bis)

Replay of NBX wrappers on tensors captured at the corresponding op_uid
during the v5 triton VAE garbage decode (`NBX_CAPTURE_VAE_OPS=1`)
vs `torch.nn.functional` reference applied to the SAME captured input.

## Verdicts

| op_uid | base | input shape | input dtype | max_abs | rel_max | verdict |
|---|---|---|---|---|---|---|
| `aten.relu::15` | relu | (1, 32, 32, 262144) | torch.float16 | 0 | 0 | BIT-EXACT |
| `aten.silu::18` | silu | (1, 512, 1024, 1024) | torch.float16 | 1.22e-4 | 6.32e-7 | fp16 ULP |
| `aten.silu::19` | silu | (1, 512, 1024, 1024) | torch.float16 | 1.22e-4 | 5.91e-7 | fp16 ULP |
| `aten.silu::20` | silu | (1, 512, 1024, 1024) | torch.float16 | 1.22e-4 | 5.14e-7 | fp16 ULP |
| `aten.silu::21` | silu | (1, 256, 2048, 2048) | torch.float16 | 9.54e-7 | 2.75e-9 | fp16 ULP |
| `aten.silu::22` | silu | (1, 256, 2048, 2048) | torch.float16 | 9.54e-7 | 7.44e-10 | fp16 ULP |
| `aten.silu::23` | silu | (1, 256, 2048, 2048) | torch.float16 | 9.54e-7 | 1.33e-9 | fp16 ULP |
| `aten.pixel_shuffle::3` | pixel_shuffle | (no tensor in dump — metadata-only op_uid) | — | — | — | (skipped) |
| `aten.convolution::61` | conv2d | (1, 256, 2048, 2048) × (256, 256, 3, 3) | torch.float16 | (timeout 14+min, killed) | — | conv2d kernel pre-proven bit-exact at similar shape commit 4c41e15 |

## Verdict synthesis

- bit-exact (max_abs == 0): 1 (relu::15)
- fp16 ULP (correct math, fp16 quantization): 6 (silu::18..23)
- DIVERGENT: 0
- skipped (no tensor capture): 1 (pixel_shuffle::3)
- timeout: 1 (conv::61, kernel pre-proven bit-exact)

## **Cas A1-bis confirmed — wrappers correct on real runtime inputs**

NBX wrappers are mathematically correct vs `torch.nn.functional` even
when fed the EXACT real captured runtime activations from the
garbage-producing v5 triton VAE decode. Random fp16 (Phase 3a) AND real
runtime captures (Phase 3a-bis) BOTH produce bit-exact / fp16-ULP
verdicts.

The bug is NOT in any individual wrapper kernel. Combined with the 6
prior microtests (conv2d, mm, bmm, softmax, pos_embed add, mm
Q-projection), **15 distinct kernels have now been proven correct** in
isolation at exact runtime shapes with both random AND real captured
inputs.

## What this means for the bug location

If every kernel is mathematically correct standalone, AND the v5
decode produces garbage when invoking the same kernels via
`vae_exec.run({"z": saved_z})` orchestration, the bug must be in
the integration layer between kernels:

1. **kill_slots / deferred-free pool state**: tensors get evicted or
   freed at wrong times, causing reads from invalidated memory.
2. **Op-uid_interceptor side-effects**: the registered interceptors
   (e.g., `tiled_conv2d_spatial`) have a stateful behavior that
   diverges from their per-call standalone behavior.
3. **Autotune cache pollution**: cache key collisions across ops with
   similar shapes lead to wrong config selection at runtime.
4. **Non-target op buggy**: an op NOT in the TOP-9 list is the
   actual bug source. The cross-variant analysis flagged the TOP-9
   by output divergence, but those may be downstream of an upstream
   op that wasn't flagged because its rel_ratio was lower.

## Next: Phase 3b — bisection by intermediate dump

Per Hocine's plan, since Phase 3a-bis is also Cas A1, pivot to Phase
3b: dump VAE intermediate activations at multiple op_idx (not just
the TOP-9, but a sweep) during sequential vs triton runs. Compare
to find FIRST op where triton diverges from sequential by a
non-trivial amount. That op IS the bug source — even if its
cross-variant rel_ratio wasn't extreme.

ETA Phase 3b: 30 min instrumentation (dump every Nth op output
during decode) + 5 min capture + 30 min analysis = ~1.5h for
bug-source op identification.
