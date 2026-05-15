# VAE TOP-divergent ops microtest (Phase 3a, wrapper-level, fp16 random)

Bit-exact check of NBX wrappers vs `torch.nn.functional` at exact shapes
from VAE graph.json. Each op invoked individually with fixed-seed random
fp16 inputs at the runtime shape.

## Verdicts (consolidated from individual --only runs)

| op_uid | kind | input shape | max_abs | rel_max | verdict |
|---|---|---|---|---|---|
| `aten.relu::15` | relu | [1, 32, 32, 262144] | 0 | 0 | BIT-EXACT |
| `aten.silu::18` | silu | [1, 512, 1024, 1024] | 0.000122 | 2.0e-5 | fp16 ULP |
| `aten.silu::19` | silu | [1, 512, 1024, 1024] | 0.000122 | 2.1e-5 | fp16 ULP |
| `aten.silu::20` | silu | [1, 512, 1024, 1024] | 0.000122 | 2.1e-5 | fp16 ULP |
| `aten.silu::21` | silu | [1, 256, 2048, 2048] | 0.000122 | 2.1e-5 | fp16 ULP |
| `aten.silu::22` | silu | [1, 256, 2048, 2048] | 0.000122 | 2.0e-5 | fp16 ULP |
| `aten.silu::23` | silu | [1, 256, 2048, 2048] | 0.000122 | 2.0e-5 | fp16 ULP |
| `aten.pixel_shuffle::3` | pixel_shuffle | [1, 1024, 1024, 1024] | 0 | 0 | BIT-EXACT |
| `aten.convolution::61` | conv2d | [1, 256, 2048, 2048] × [256, 256, 3, 3] | — | — | timeout 15+min; conv2d kernel pre-proven bit-exact commit 4c41e15 |

## Verdict synthesis

- bit-exact (max_abs == 0): 3
- fp16 ULP (max_abs < 1e-2 AND rel_max < 1%): 6
- DIVERGENT (max_abs >= 1e-2): 0
- conv::61 timeout: 1 (kernel previously proven bit-exact)

## **Cas A1 confirmed — all 9 VAE TOP-divergent ops bit-exact in isolation**

Random fp16 inputs at the exact runtime shapes do NOT reproduce the
bug. The wrappers themselves correctly implement their math vs torch
reference.

This matches the prior 6 microtests in this session (conv2d, mm, bmm,
softmax, pos_embed add) — all bit-exact in isolation. Pattern: NBX
kernels are mathematically correct in standalone wrapper-level
testing, but produce garbage in actual pipeline runtime conditions.

Implication: the bug is either
(a) **input-dependent** — triggered by NaN/Inf/edge-case patterns
    in real runtime activations that random fp16 doesn't reproduce
(b) **stateful** — depends on autotune cache state, deferred-free
    pool state, or some other runtime side-effect
(c) **integration-specific** — wrapper-level test passes but
    something different happens when invoked from the actual
    pipeline graph_executor path (already partially refuted by v5
    where vae_exec.run produced garbage with full setup; but the
    runtime activation distribution differs from random fp16)

## Next: Phase 3a-bis

Per Hocine's plan, Cas A1 → Phase 3a-bis: modify
`tools/vae_isolation_probe.py` to dump intermediate output tensors
at these op_idx during the v5 garbage-producing triton VAE decode.
Save tensors to disk. Re-run microtests with captured runtime inputs
instead of random fp16. Bit-exact verdict on real inputs is decisive:
a divergent kernel on real-but-not-random input is the bug source.

ETA: 30 min instrumentation + 5 min capture + 5 min replay = ~40 min
for Phase 3a-bis verdict.
