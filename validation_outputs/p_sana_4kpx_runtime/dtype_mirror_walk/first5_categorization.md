# Categorization of the first 5 dtype divergences (by op_idx)

Per Hocine's directive: categorize each as
- **(a)** NBX applies upcast/downcast PyTorch doesn't → wrapper too defensive
- **(b)** NBX misses a cast PyTorch applies via DtypeEngine/autocast → TritonDtypeEngine gap
- **(c)** NBX inherits wrong dtype upstream → op itself innocent

## First 5 divergences (op_idx 0..4)

| op_idx | op_uid | op_type | seq_in / out | tri_in / out | category | wrapper file:line |
|---|---|---|---|---|---|---|
| 0 | aten.unsqueeze::0 | aten::unsqueeze | [fp16, scalar] / fp16 | [fp32, scalar] / fp32 | **(c)** | (no NBX wrapper, dispatcher passthrough) |
| 1 | aten.expand::0 | aten::expand | [fp16, non_tensor] / fp16 | [fp32, non_tensor] / fp32 | **(c)** | (passthrough) |
| 2 | aten.clone::0 | aten::clone | [fp16] / fp16 | [fp32] / fp32 | **(c)** | (passthrough) |
| 3 | aten.view::0 | aten::view | [fp16, non_tensor] / fp16 | [fp32, non_tensor] / fp32 | **(c)** | (passthrough) |
| 4 | aten.convolution::0 | aten::convolution | [fp16, fp16, fp16] / fp16 | [fp32, fp16, fp16] / fp16 | **(c)** + (boundary handling) | `src/neurobrix/kernels/wrappers.py:2397` (`conv2d_wrapper`); narrowing at line ~2467-2473 |

**Common root**: `input::z` (the saved latent fp32 tensor) reaches
the runtime tensor store as **fp16** in sequential mode, but **fp32**
in triton_sequential mode. Both engines' metadata ops
(unsqueeze/expand/clone/view) preserve the input dtype faithfully —
they are innocent. The DIVERGENCE seeds at runtime input loading.

In sequential mode: `RuntimeExecutor._execute_native_op` wraps each
op (including the metadata chain) with `DtypeEngine.amp_cast_inputs`
(`core/dtype/engine.py:455`). For non-AMP ops like view/expand/clone,
this is passthrough. **But the activation tensor was already fp16
before the metadata chain**, indicating the cast happens somewhere
EARLIER (likely at `_resolver.normalize_inputs` or weight/input
loading — needs further trace).

In triton_sequential mode: `TritonSequentialDispatcher.dispatch`
applies `TritonDtypeEngine.wrap_op` (`triton/dtype.py:147`). For
non-AMP ops (metadata chain), passthrough. Input::z arrives fp32
and stays fp32 through the chain.

The conv2d op at op_idx=4 is the FIRST compute op:
- seq: receives all-fp16 inputs (from amp_cast_inputs upstream of conv2d).
- tri: receives [fp32, fp16, fp16]. NBX `conv2d_wrapper`
  (`wrappers.py:2397`) narrows the activation to fp16 internally
  (`narrow-on-mismatch` rule at lines 2467-2473), output fp16. So
  conv::0's OUTPUT is fp16 in both modes. The wrapper handles the
  mismatch correctly.

## Bonus: cleanest kernel-level divergence — `custom.rms_norm::0` (op 58)

| op_idx | op_uid | seq_in / out | tri_in / out | category | wrapper file:line |
|---|---|---|---|---|---|
| 58 | custom.rms_norm::0 | [fp32, fp16] / **fp16** | [fp32, fp16] / **fp32** | **(a)** | `src/neurobrix/kernels/wrappers.py:1025` (`rms_norm`); cast-back gated at `triton/dtype.py:170-180` |

**SAME inputs** in both modes. **DIFFERENT output**:
- seq: outputs fp16 (cast back to compute_dtype)
- tri: outputs fp32 (no cast back unless `activations_fp16_safe=True`)

Sequential's `DtypeEngine` for `rms_norm` (in `AMP_FP32_OPS`) casts
inputs to fp32, runs native, then **casts result back to compute_dtype**.
Looking at `core/dtype/engine.py` `compile_op` and `amp_cast_result`
implementation, the seq cast-back appears unconditional for AMP_FP32 ops.

NBX's `TritonDtypeEngine` has the SAME AMP rules logically, but adds
an OPT-IN gate `activations_fp16_safe` (`triton/dtype.py:49,170`) that
prevents the cast-back unless the model registry sets the flag. Default
False → output stays fp32 → "more conservative" but DEVIATES from seq.

This is the **clearest example of Category (a) — NBX too defensive**.
NBX keeps fp32 output where seq casts back to fp16. PyTorch's behavior
(seq) is the oracle per directive doctrine: NBX should mirror.

## Other notable divergences (in_dts only, output matches)

For mm/bmm ops (op 9, 13, 17, 54, ...): seq inputs `[fp32, fp32]`,
tri inputs `[fp32, fp16]`. Seq's `amp_cast_inputs` for mm in
`_FP16_NEED_FP32` upcasts BOTH inputs to fp32. NBX mm wrapper
(`wrappers.py:1197`) keeps weight fp16 with PROMOTE_B inline cast in
the kernel — fp16 weight, fp32 activation, kernel produces fp32 output.

Both produce fp32 output, but seq materializes fp32 weight in memory
(higher VRAM cost), tri keeps fp16 weight (lower VRAM, equivalent
math via PROMOTE_B). NBX is ARITHMETICALLY equivalent but more
VRAM-efficient. This isn't a defect — it's a deliberate optimization
documented in `wrappers.py:1219-1229`.

By Hocine's directive ("retirer du fp32 défensif inutile dans NBX"),
this is actually the CORRECT direction (NBX matches output, saves
VRAM). The seq-vs-tri input dtype difference is acceptable. Marked
in TSV but not flagged as a fix target.

## Summary table — first 5 + bonus

| op_idx | op_uid | category | fix scope |
|---|---|---|---|
| 0 | unsqueeze::0 | (c) | upstream input::z load; not a wrapper fix |
| 1 | expand::0 | (c) | inherits |
| 2 | clone::0 | (c) | inherits |
| 3 | view::0 | (c) | inherits |
| 4 | convolution::0 | (c) | wrapper handles correctly; inherits propagation |
| 58 | rms_norm::0 | **(a)** | `triton/dtype.py:170-180` cast-back gate or `wrappers.py:1025` rms_norm |

## Awaiting Hocine arbitrage

Per directive: "Une fois le verdict livré, on choisira ensemble par
lequel commencer".

Two distinct fix surfaces emerge:
1. **input::z runtime load cast** for triton mode (touches
   `core/runtime/executor.py` and/or `triton/sequential.py` input
   processing — beyond pure wrapper scope).
2. **rms_norm cast-back default** in TritonDtypeEngine — flip the
   opt-in flag for Sana family (model_registry change), or unconditional
   cast-back in `triton/dtype.py:170-180`.

Both are needed to align tri to seq's PyTorch-mirror behavior. Order
matters: fixing #2 might still leave the upstream cascade. Fixing #1
removes the cascade root.

R30 anti-régression Sana 1024 4-mode obligatoire après chaque fix.
