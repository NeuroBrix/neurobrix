# Janus-Pro-7B `::triton` — numerical anti-correlation at step 0

## Status

- **Runtime unblocked**: Janus-Pro-7B `--triton` now runs end-to-end and writes
  a 384×384 PNG at the default output path. Decode 153 s on `v100-32g`
  (faster than native 192 s at this prompt length).
- **Numerical correctness NOT met**: cosine(native, triton) on step-0
  post-CFG logits is **~0** (top-10 indices disjoint); on a full-vocab
  dump of the pre-CFG conditional logit vector the cosine is **-0.986**
  with a triton L2 magnitude ~120× native's. Parity gate (≥ 0.99) is
  therefore missed by a wide margin and the user-visible image is
  not a cat — it renders as a blue sky / blurred ground texture.

## What was fixed to get the pipeline running at all

Four independent runtime correctness issues blocked Janus on the triton
path. All four landed in this session.

### 1. `NBXTensor.view` silently mis-strode expand views

`src/neurobrix/kernels/nbx_tensor.py`. The method treated every view as
contiguous, re-striding the header to `_contiguous_strides(shape)` but
leaving the underlying allocation at the expand-view backing size.  Janus
LM's RoPE chain `expand → view` produced `shape=(2,64,1) numel=128` with
only 64 fp32 elements of backing memory. The second batched-loop
iteration of `bmm` walked past the allocation and Triton flagged the
pointer as host memory (`Pointer argument (at 0) cannot be accessed from
Triton (cpu tensor?)`). Fix: if the input is not contiguous, `view` now
materializes first (same fallback `reshape` already had).

### 2. `bmm` missing pre-launch `_set_device`

`src/neurobrix/kernels/wrappers.py`. `mm()` syncs the Triton driver to
`a`'s device immediately before `matmul_kernel[grid](...)`; `bmm` did the
sync only at the start of the function and then allocated a new output
tensor, which could cudaSetDevice without updating the Triton driver's
active device. Added `_set_device(a)` mirror.

### 3. `aten::native_group_norm` wrapper signature mismatch

`src/neurobrix/kernels/wrappers.py` + `kernels/dispatch.py`. The graph
emits the 8-arg ATen form `(input, weight, bias, N, C, HxW, num_groups,
eps)`; the dispatch pointed `native_group_norm` at the 5-arg friendly
`group_norm_wrapper(x, num_groups, weight, bias, eps)`. Added
`native_group_norm_wrapper` adapter that forwards to the friendly
wrapper, dropping the redundant scalars (recomputed inside the kernel).

### 4. `triton/flow/autoregressive.py` was LLM-only

No `TritonImageStrategy`, no CFG branch in `_tokenize`, no dispatch on
`gen_type`, no post-decode VQ decoder call. Ported the full
`ImageStrategy` from `core/flow/autoregressive.py`: CFG tokenize,
`_run_head` batch-split + CFG combine, `gen_embed` + `gen_aligner`
decode path, `gen_vision_model.decode_code` post-loop. Also added an
LM + KV-cache cleanup before `gen_vision_model` runs, otherwise the
decoder OOMs on top of the 13 GB bf16 LM + batch-2 KV on `v100-32g`.

These four changes together take Janus triton from "immediate crash at
first bmm" to "runs, produces an image".

## What is still wrong

Collected in `/tmp/janus_parity/` (full-vocab dumps of cond/uncond logits
post gen_head at step 0 for both engines):

```
native cond   argmax= 2122  max=  19.066  L2= 1517.85
native uncond argmax= 3736  max=  23.770  L2= 2140.07
triton cond   argmax=14025  max= -909.93  L2=182662.88
triton uncond argmax= 5725  max=-1068.61  L2=204235.51

cos(native cond,   triton cond)   = -0.986152
cos(native uncond, triton uncond) = -0.995283
cos(native cond,   native uncond) =  0.994400  (baseline)
cos(triton cond,   triton uncond) =  0.999263  (!)
```

Two independent signals, both strong:

1. **Anti-correlation with ~120× magnitude.** `cos(native, triton) ≈ -1`
   means the triton logit vector points opposite to native's. Magnitude
   (L2) is ~120× larger. This is not accumulated numerical drift — it's
   a structural op that's either sign-flipping or magnitude-amplifying a
   value that then cascades through the final addmm. Candidates to audit
   in order of suspicion:
   - Softmax direction or mask polarity inside triton SDPA (if attention
     weights are negated or normalized on the wrong axis, the attention
     output is a weighted average that collapses to something near
     `-V.mean()`, and the residual chain amplifies it over 30 layers).
   - RMSNorm `rsqrt` path on V100 fp16. With
     `compute_dtype=fp16` + `has_native_bf16=False`, the DtypeEngine
     upcasts `pow` and `rsqrt` individually but the accumulator around
     the full RMSNorm may not be. Worth a targeted comparison.
   - Janus uses `polar` + `view_as_complex` for RoPE. CLAUDE.md §13
     documents a double-scaling pattern fixed by
     `_normalize_sdpa_scaling` — confirm the fix actually runs for this
     model's `_scaled_dot_product_efficient_attention` variant on the
     triton path (it runs at `_load_graph_from_dict`, which is shared,
     but the triton SDPA interceptor may bypass the normalized mul
     ops if they were pre-rewritten).

2. **Triton cond ≈ triton uncond** (`cos = 0.9993`). The two CFG batches
   produce nearly-identical logits, while the native baseline keeps them
   at 0.994 — different enough for CFG to do meaningful work. This is
   **cross-batch leakage** in triton's batch-2 prefill. Likely
   localizations:
   - Attention mask shape not broadcasting correctly to `(B=2, H, S, S)`
     (mask at `(1,1,S,S)` may apply to batch 0 only, or the wrong batch
     slot sees the conditional tokens).
   - `TritonAttentionInterceptor` mishandling batch dim when KV cache
     is initialized from the first K tensor (K cached shape carries
     `batch_size=2`, but per-batch KV slots may alias).
   - RoPE cos/sin indexed with a single position_ids row instead of
     both; batch-broadcast cos/sin then gives the same rotation to
     both sequences whose positions should be identical — this alone
     would not explain leakage, but confirms we need to audit every
     attention-time tensor for `(B, …)` vs `(1, …)` consistency.

These are two distinct symptoms; it is possible they share a root
cause (the sign flip could be the same ops that are aliasing across
batches) but the diagnostic above cannot prove it.

## Investigation tasks (DO NOT PERFORM IN THIS DOC; dedicated session)

1. Reproduce the anti-correlation on `TinyLlama-1.1B`'s triton path
   with CFG-style batch=2 synthetic prefill. If TinyLlama is clean, the
   bug is either Janus-specific (graph-level) or only surfaces in
   batch=2 with a particular shape combination. Minimal repro keeps the
   next session from re-loading 13 GB bf16 every iteration.

2. Add a per-layer hidden-state dump on both engines for the first few
   layers of Janus's language_model. The sign flip must appear
   somewhere; localize which layer first produces negative output with
   ~10× magnitude of native. That narrows the suspect op cluster.

3. Audit `TritonAttentionInterceptor.intercept` for explicit batch
   handling — does it pass through `(B, H, S, D)` inputs intact, or
   does it reshape to `(H, S, D)` anywhere? Is the KV cache's batch
   dimension keyed by sample or shared?

4. Verify `_normalize_sdpa_scaling` finds and neutralizes all the
   decomposed `mul(Q, sqrt_scale)` + `mul(K, sqrt_scale)` ops on Janus's
   LM graph (DeepSeek-V1-style decomposition). Log count of mul ops
   removed per component on both modes and compare.

5. Propose a fix. Both sign and magnitude issues should be traceable
   to a single kernel/wrapper once the per-layer dump localizes the
   breaking layer.

## Artefacts from this session

- `/tmp/janus_parity/native_cond.json`, `native_uncond.json` —
  full-vocab logits at step 0, CFG split, native path.
- `/tmp/janus_parity/triton_cond.json`, `triton_uncond.json` — same on
  triton.
- `/tmp/janus_parity/native.png` — native-generated cat (photorealistic).
- `/home/mlops/NeuroBrix_System/output_Janus-Pro-7B.png` — triton-
  generated image (wrong semantics, correct pipeline shape).

## What this session explicitly does NOT change

- Zero-torch rule in `src/neurobrix/triton/` — unchanged.
- Native / other-model triton paths — TinyLlama `--triton` was
  re-verified green after each relevant edit.
- CFG formula itself — it matches native; the CFG-combined logits being
  wrong is downstream of cond and uncond already being wrong.
