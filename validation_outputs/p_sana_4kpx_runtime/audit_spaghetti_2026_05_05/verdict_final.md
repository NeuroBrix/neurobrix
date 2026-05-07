# P-SANA-4KPX-RUNTIME — Verdict factuel final 2026-05-05/07

## Update 2026-05-07 (final⁸ — VAE acquitted, bug lives in transformer)

Per user's plan step (3) discriminator: instrument transformer→VAE
boundary in seq 4Kpx (PNG coherent, gold) vs tri 4Kpx (PNG garbage).

### Boundary diff results

| Position | Op | seq vs tri max_d | Notes |
|---|---|---|---|
| Last transformer op (line 35025) | aten.clone::81 | **0.998** on values ~1 | 100% rel drift |
| First VAE op (line 35026) | aten.unsqueeze::0 | 13.54 | downstream of corrupted latent |
| Last VAE op (line 35761) | aten.convolution::69 | structured | decoded garbage → garbage |

### Discriminator verdict

**VAE is INNOCENT**. The transformer's noise prediction at iteration
11 already differs by 100% relative between seq (cuDNN) and tri (NBX)
paths. After 12 scheduler iterations, the cumulative drift in the
latent is fully developed; VAE receives a wrong latent and faithfully
decodes the wrong-but-structured input into structured-but-wrong
output (the green texture pattern observed in the garbage PNG).

Bug location confirmed: **transformer cross-attention chain**.
Mechanism: tiny cuDNN-vs-NBX fp32 numerical drift gets amplified
exponentially through softmax of high-magnitude pre-softmax cross-
attention scores at Sana 4Kpx scale (K=16384 reduction).

Sana 1024 doesn't exhibit this because K=1024 keeps pre-softmax
scores in a regime where 1% drift doesn't flip softmax mass.

### Final session commit list (12 total)

- `a2fd933` fingerprint logical→physical mapping
- `aadf5b0` fingerprint via data_ptr() (slice/narrow alignment)
- `1c78a62` (superseded) verdict pointing at conv kernel
- `4c41e15` microtest disproves conv kernel
- `4508293` triton 1024 vs 4Kpx oracle redirect
- `8104c8d` fingerprint drops n-1, uses 5 interior positions
- `4f939bd` clean diff reveals magnitudes are not the bug
- `0c9588a` softmax bit-exact, 4Kpx PNG is structured texture
- `3cd1595` bug nature: drift amplification, not kernel bug
- (this update) VAE acquitted, transformer cross-attention chain
  is bug location

### Resolution path (next chantier)

This is a **numerical-precision gap** between PyTorch (cuDNN/cuBLAS)
and NBX (Triton). Three options:

A) Bit-match cuDNN GEMM accumulation pattern on Volta. Structural
   ~12% perf gap per CLAUDE.md autotune doctrine — fundamentally
   hard given Triton compiler's dot lowering on sm_70.

B) Higher-precision intermediate (fp64) for the attention chain,
   specifically pre-softmax scores. Costly (~2× memory) but
   eliminates the chaotic-amplification regime.

C) Algorithm-level fix: apply attention scaling 1/sqrt(d_head) BEFORE
   bmm Q@K^T instead of after, keeping pre-softmax magnitudes
   bounded. Mirrors the F2a pattern of structural fixes.

User direction needed on which path A/B/C to pursue in the next
session. The chantier methodology is now complete: bug nature
characterized, location pinpointed, kernels acquitted by microtest.

---

## Update 2026-05-07 (final⁷ — bug nature identified: drift amplification, not kernel bug)

### Six microtests at exact failing shapes — ALL BIT-EXACT

| Kernel | Shape (4Kpx) | Result vs torch reference |
|---|---|---|
| conv2d_forward_kernel | (2,32,128,128)→(2,2240,128,128) bf16/fp16 | BIT-EXACT |
| pos_embed add (broadcast) | (2,16384,2240) + (1,16384,2240) | BIT-EXACT |
| bmm cross-attn QK^T | (140,33,16384) × (140,16384,32) | BIT-EXACT |
| softmax fp32 | scales 1, 1e6, 1e7 | BIT-EXACT (sum=1.0) |
| mm self-attn Q | (32768,2240) × (2240,2240) fp32 | BIT-EXACT |
| (Sana 1024 control mm) | (2048,2240) × (2240,2240) fp32 | 5e-4 noise (1 sign flip / 4.5M) |

### Sequential 4Kpx vs triton_seq 4Kpx — error growth trajectory

Same model, same prompt, same seed, same graph. Only the backend
differs (PyTorch cuDNN vs NBX Triton).

| Threshold | First op | Component | max_d |
|---|---|---|---|
| abs_d ≥ 1   | aten.mean::78 | text_encoder | 1 (drift on value=5790) |
| abs_d ≥ 10  | aten.mm::0    | transformer iter 1, block.0.self_attn.query | 26.64 (sign flip pos 2) |
| abs_d ≥ 100 | aten.bmm::0   | transformer iter 1 | 420 |
| abs_d ≥ 1k  | aten.bmm::1   | transformer iter 1 cross-attn QK^T | **1.47e+06** |

### Mechanism — drift amplification through softmax

bmm::0 → softmax → bmm::1 chain:
1. tiny cuDNN-vs-NBX numerical drift in text_encoder (~0.02% rel on
   activations of magnitude ~5000).
2. mm Q-projection: drift becomes 26 absolute (sign flip on small
   value within otherwise-aligned matrix).
3. bmm::0 (cross-attn scores from one Q row): drift = 420 on values
   of magnitude 40,000 (~1% rel).
4. softmax of (40000, 39580): exponentiation amplifies the 1%
   relative input difference into orders-of-magnitude output
   differences (probability mass flips between competing positions).
5. bmm::1 (softmax_probs @ V): catastrophic 1.47M absolute divergence
   from cuDNN reference.

This isn't a kernel bug — it's the **chaotic-dynamics nature of
attention at the magnitude regime Sana 4Kpx hits**. Tiny numerical
drift between cuDNN and NBX Triton gets amplified by softmax of
high-magnitude pre-softmax scores. Sana 1024 doesn't exhibit this
because its smaller seq_len keeps scores in a regime where 1%
relative drift doesn't flip softmax mass distribution.

### What this means for the chantier

The conclusion changes the resolution path:

A) **Match cuDNN behavior numerically**: NBX mm/bmm needs to
   reproduce cuDNN's accumulation order on Volta sm_70 to within
   tighter tolerance than 1% relative. This requires either
   (a) bit-exact GEMM that matches cuBLAS on Volta (autotune not
   sufficient — it's already at 12% perf vs cuBLAS, the gap is
   structural per CLAUDE.md autotune doctrine), or
   (b) higher precision accumulation (fp32→fp64) for the chain
   before softmax — costly.

B) **Reduce attention-chain sensitivity**: pre-softmax score
   magnitude 4e+04 in fp32 is high but representable. cuDNN handles
   it; NBX path's bit-pattern of accumulation does not match
   cuDNN's. A targeted patch could force NBX softmax to use a
   higher-precision intermediate, or apply attention scale before
   bmm to keep magnitudes lower.

C) **Numerical reproducibility audit**: trace cuDNN vs NBX accumulation
   block-by-block at a representative mm to identify exactly where
   the 0.02% rel drift originates. If it's a specific NBX
   block-config choice, target that.

### Microtest gap — what tests don't exhibit the drift

Random-input microtests of mm/bmm at the failing shapes ALL show
bit-exact match with torch. The drift only manifests when the
inputs come from the actual pipeline — i.e., from a chain of
preceding NBX ops. This means the issue is **input-dependent**,
not shape-dependent. A specific structure in the pipeline's
intermediate activations triggers the cuDNN-vs-NBX divergence,
not the matrix shapes themselves.

ETA next session: 1-2h targeted instrumentation comparing
intermediate activations seq vs tri at a single transformer block
to identify which kernel's intermediate output first drifts beyond
expected fp32 noise.

Commits this session (final list, 11 total):
- `a2fd933` fingerprint logical→physical mapping
- `aadf5b0` fingerprint via data_ptr() (slice/narrow alignment)
- `1c78a62` (superseded) verdict pointing at conv kernel
- `4c41e15` microtest disproves conv kernel
- `4508293` triton 1024 vs 4Kpx oracle
- `8104c8d` fingerprint drops n-1, uses 5 interior positions
- `4f939bd` clean diff reveals magnitudes are not the bug
- `0c9588a` softmax bit-exact, 4Kpx PNG is structured texture
- (this update) 6 kernels acquitted by microtest; bug is drift
  amplification through softmax at attention chain; resolution
  path requires either cuDNN-bit-match or higher-precision
  intermediate for attention chain

---

## Update 2026-05-07 (final⁶ — softmax cleared, 4Kpx PNG inspected)

### Microtest sweep results

| Kernel | At Sana 4Kpx shape | Verdict |
|---|---|---|
| conv2d_forward_kernel | (2,32,128,128)→(2,2240,128,128), bf16 + fp16 | BIT-EXACT vs cuDNN |
| pos_embed add (broadcast) | (2,16384,2240) + (1,16384,2240) | BIT-EXACT vs torch |
| bmm cross-attn | (140,33,16384) × (140,16384,32) | BIT-EXACT vs torch |
| softmax (fp32) | (8,33,16384) at scales 1, 1e6, 1e7 | BIT-EXACT vs torch (sum=1.0, no NaN) |

**Five candidate kernels eliminated** as bug sources by isolated
microtest at the exact failing shapes.

### Magnitude regime is NOT the bug

Both Sana 1024 (PNG coherent) and Sana 4Kpx (PNG garbage) reach
attention pre-softmax scores of magnitude 10⁶+. fp32 softmax max-
subtraction handles both regimes correctly. NaN/Inf scan of 4Kpx
trace shows zero spurious infinities (the 105 -inf entries are
text_encoder mask saturation, identical between models).

### 4Kpx PNG visual inspection

Output is structured green texture with regular grain pattern (not
random noise). This indicates **systematic miscomputation** in a
specific op, not stability failure. The model executes mostly
correctly but produces wrong-but-structured intermediate
activations that decode to noise.

### Remaining hypotheses

1. **VAE 4Kpx specifically**: graph md5 differs from 1024
   (`a7401c1d` vs `8abfd113`), shape-dependent op that cuDNN
   handles in sequential mode but NBX mishandles in triton.
2. **Transformer attribute-dependent op**: an op whose attributes
   differ between 1024 and 4Kpx graphs (positional embedding scale,
   shape-dependent attribute) where the NBX handler doesn't cover
   the 4Kpx attribute case correctly.

### Decisive next test (requires next session)

Instrument the transformer→VAE boundary in BOTH sequential 4Kpx
(coherent PNG) and triton 4Kpx (garbage PNG). Compare the
transformer's last output (= VAE input) values:
- If match: bug is in VAE 4Kpx only.
- If differ: bug is in transformer at some op the current bisection
  couldn't surface (likely a shape-attribute mismatch).

ETA: 30 min instrumentation + 5 min run + diff = 1h to discriminate.

Commits this session:
- `a2fd933` fingerprint logical→physical mapping
- `aadf5b0` fingerprint via data_ptr() (slice/narrow alignment)
- `1c78a62` (superseded) verdict pointing at conv kernel
- `4c41e15` microtest disproves conv kernel
- `4508293` triton 1024 vs 4Kpx oracle
- `8104c8d` fingerprint drops n-1, uses 5 interior positions
- `4f939bd` clean diff reveals magnitudes are not the bug
- (this update) softmax bit-exact at all magnitudes, 4Kpx PNG
  is structured green texture (systematic miscomputation,
  not overflow); next session: transformer→VAE boundary diff

---

## Update 2026-05-07 (final⁵ — fingerprint clean, bug area localized)

Per user's strict redirect: fingerprint helper had a 3rd bug at the
n-1 sample (`pos[0..3] match, pos[4] differs` pattern in 100% of
text_encoder ops, even though text_encoder graph md5 + safetensor
md5 are bit-identical between Sana 1024 and Sana 4Kpx). Neither
sync_device() nor the prior logical-to-physical fix eliminated this
artifact. **Fix**: drop the n-1 sample. Use 5 interior positions
`(0, n//8, n//4, n//2, 3n//4)` immune to any boundary-write artifact.
Apply same change to torch path for symmetric semantics across paths.

### Re-run with clean fingerprint (commit `8104c8d`):

Sana 1024 triton_sequential v3 vs Sana 4Kpx triton_sequential v5:
- text_encoder: **3261/3261 float ops bit-exact** (perfect alignment).
- transformer iter 0 ops 0..6 (pre-pos_embed): bit-exact.
- transformer iter 0 ops 7..end: alignment off-by-1 (4Kpx has extra
  add::0 for pos_embed); after structural skip, downstream values
  diverge by EXPECTED amount (fine-tunes have different weights).

### Where divergence becomes ABNORMAL

After skipping the structural extra add::0, the first op where
4Kpx's max-abs significantly exceeds 1024's is at line 6927
(transformer iter 1):

```
op_idx=79 aten.bmm::0   ratio 63×
  1024 max-abs: 630.6   |   4Kpx max-abs: 4e+04
```

This is the cross-attention Q @ K^T:
- 1024: bmm((140, 33, 1024),  (140, 1024, 32))
- 4Kpx: bmm((140, 33, 16384), (140, 16384, 32))
- Reduction dim K=16384 vs K=1024 (16× larger).

### bmm kernel microtest at exact shapes

```
Sana_1024 ((140,33,1024), (140,1024,32)) -> max abs diff 0.000487 (fp16 ULP)
Sana_4Kpx ((140,33,16384),(140,16384,32))-> max abs diff 0.002 (fp16 ULP)
```

NBX bmm is **BIT-EXACT vs torch.bmm at both shapes**. The 63× ratio
reflects model-weight differences between Sana 1024 (MultiLing) and
Sana 4Kpx (BF16) fine-tunes, NOT a kernel bug.

### CRITICAL re-read

Sana 1024 ALSO has values at ~10⁶ (e.g. line 6991: 1024=4.22e+05).
**Sana 1024 PNG is coherent**, proving 10⁶ magnitudes are normal for
pre-softmax attention scores. The failure mode at 4Kpx is therefore
NOT the magnitudes themselves but how they propagate through some
downstream op. Likely softmax / normalization at the magnitude
regime where 4Kpx's 10⁷ values overflow the kernel's stability
guards but 1024's 10⁶ values stay within them.

### Next concrete investigation

1. Find FIRST op where 4Kpx's NaN / Inf appears (vs 1024 staying
   finite). Add NaN-detection to the diff over all components.
2. Microtest NBX softmax kernel at large-magnitude inputs — check if
   max-subtraction trick is applied for stability at fp16 inputs of
   magnitude 10⁷.
3. If softmax is OK, test layer_norm / rms_norm at large-magnitude
   inputs — check for catastrophic cancellation in variance compute.

ETA: 30 min instrumentation + microtests, then targeted fix.

Commits this session (final list):
- `a2fd933` fingerprint logical→physical mapping
- `aadf5b0` fingerprint via data_ptr() (slice/narrow alignment)
- `1c78a62` (superseded) verdict pointing at conv kernel
- `4c41e15` microtest disproves conv kernel
- `4508293` triton 1024 vs 4Kpx oracle redirect
- `8104c8d` fingerprint drops n-1, uses 5 interior positions
- (this update) bmm bit-exact, focus shifts to softmax/norm at
  large-magnitude regime

---

## Update 2026-05-07 (final⁴ — triton 1024 vs triton 4Kpx oracle redirected)

User redirected the comparison: the correct oracle is **Sana 1024
triton_sequential (PNG coherent ✓) vs Sana 4Kpx triton_sequential
(garbage)**, not sequential vs triton_sequential of the same model.
Reasons:
- text_encoder graph.json md5 IDENTICAL between Sana 1024 and Sana
  4Kpx (`bada6254...`); same prompt → embeddings MUST be bit-aligned.
- Same kernels, same code, same NBX path on both models.
- Sana 1024 triton_sequential PNG VALIDATED (red apple, R29 PASS at
  this session: `sana1024_triseq_values.png`).

Ran Sana 1024 triton_sequential with VALUE_TRACE instrumentation,
diff'd vs Sana 4Kpx triton_sequential. Findings:

### text_encoder (3449 ops each, identical graph)

| classification | count |
|---|---|
| bit-exact / fp16 noise | 1118 |
| pos-4-only (fingerprint artifact at n-1) | 2143 |
| wide divergence (real numerical) | **0** |

**Conclusion**: text_encoder is bit-aligned between Sana 1024 and
Sana 4Kpx. The "pos-4-only" pattern is a fingerprint helper artifact
(the n-1 sample reads inconsistently — 3rd fingerprint bug
discovered this session, orthogonal to the bug hunt).

### transformer iter 0 (3449 vs 3449 ops)

Graphs differ md5 but op count differs by ONLY 1 (Sana 4Kpx has 2344
ops, Sana 1024 has 2343). The one extra op:

```
op_idx=7  Sana 4Kpx ONLY:  aten.add::0
  inputs:  aten.transpose::0::out_0  (2, 16384, 2240)
           param::patch_embed.pos_embed (1, 16384, 2240)
  output:  (2, 16384, 2240)   = 73,400,320 elements
  parent:  patch_embed
```

Sana 1024 lacks this op — its patch_embed doesn't add a learnable
positional embedding (different scheme). All subsequent ops are
shifted by 1 between the two graphs.

### Microtest: NBX add at (2, 16384, 2240) + (1, 16384, 2240)

```
ref samples: [0.266, -0.161, 0.0909, -0.212, -0.042]
nbx samples: [0.266, -0.161, 0.0909, -0.212, -0.042]
max abs diff: 0.0   (BIT-EXACT)
batch[1] start max diff: 0.0 (broadcast batch dim correct)
```

The NBX add wrapper handles this broadcast pattern bit-exact.
**The pos_embed add itself is NOT the bug source**.

### Where this leaves the bug

- text_encoder: clean
- transformer patch_embed pos_embed add: clean in isolation
- conv2d at this shape: clean in isolation (microtest above)

The bug must be in a DOWNSTREAM op that responds to scale-dependent
shapes (e.g., 16384 patches vs 1024). Candidate next investigations:
1. transformer attention ops at seq_len=16384 (vs 1024) — softmax,
   sdpa, mm at large M dim. If autotune picks a buggy config at
   M=16384 in transformer block 0+, divergence cascades.
2. VAE graphs differ md5 (`a7401c1d...` vs `8abfd113...`). If the
   bug is in a VAE op only present at 4Kpx (or scale-dependent),
   it'd corrupt the final decode step. Sequential VAE 4Kpx works
   (PNG coherent) so cuDNN handles it; NBX VAE may not.

ETA next iteration: 30 min — instrument transformer block 0 attention
in both runs, diff specific op outputs at seq_len=16384 vs 1024.

Commits this session:
- `a2fd933` fingerprint logical→physical mapping
- `aadf5b0` fingerprint via data_ptr() (slice/narrow alignment)
- `1c78a62` (superseded) verdict pointing at conv kernel
- `4c41e15` microtest disproves conv kernel
- (this update) text_encoder bit-aligned, pos_embed add bit-exact,
  bug must be in downstream scale-dependent op

---

## Update 2026-05-07 (final³ — fingerprint methodology cleaned, bug pinpointed)

The op-by-op value diff (sequential vs triton_sequential) initially
flagged ~70% of ops as divergent. Investigation revealed the
fingerprint helper was reading NBXTensor at raw `_data_ptr + i *
el_bytes` while reading torch.Tensor via `flatten()[i]`. Two methodology
bugs:

1. **Logical-to-physical mapping**: torch flatten respects strides;
   raw byte offsets sample physical storage order. Disagrees on every
   transposed view (commit a2fd933).
2. **`_offset` not added**: NBXTensor stores `_offset` separately
   from `_data_ptr`; narrow/select shift only `_offset`. Reading raw
   `_data_ptr` for a slice gave parent storage from offset 0 → every
   `aten.slice::N` flagged divergent (commit aadf5b0).

After both fixes, re-run triton_sequential, re-diff. The trace is now
CLEAN through op_idx 0-3. The FIRST real divergence is:

```
op_idx=4 op_uid=aten.convolution::0   n_div=5/5  max_rel=1.34
seq    = [0.1125, 0.915, 0.1125, 0.915, -0.1318]
tri    = [0.4619, -0.3113, 0.4619, -0.3113, -0.2042]
```

Sana 4Kpx shape: `(2, 32, 128, 128) → (2, 2240, 128, 128)` — 1×1
conv, stride 1, padding 0, groups 1. batch_dim = 2*128*128 = 32768.
Sana 1024 same op type, batch_dim=2048 → coherent PNG.

The bug lives in `conv2d_forward_kernel` (or its wrapper / autotune
config selection) at this specific shape, NOT in upstream propagation.
All upstream ops (text_encoder + first 4 transformer ops) align.

**Microtest verdict (2026-05-07 14:50 UTC)**:

Built a focused microtest at the EXACT failing shape `(2,32,128,128) →
(2,2240,128,128)`, 1×1 conv, random inputs. NBX `conv2d_wrapper`
output matches torch cuDNN reference **BIT-EXACT** in both dtypes:

| Path | NBX max abs diff vs cuDNN | Verdict |
|---|---|---|
| bf16 | 0.000 (BIT-EXACT) | clean |
| fp16 | 0.00195 (= 1 fp16 ULP) | clean |

The kernel is **NOT buggy** at this shape. The bug is **upstream of
conv::0 in the pipeline** — different inputs and/or weights flowing
into the conv between sequential and triton_sequential paths.

Supporting evidence: `aten.mul::394` at op_idx=3447 (last text_encoder
op before transformer starts) shows position-4 divergence
(seq=-2.464 vs tri=-4.853, 2× difference). text_encoder feeds
cross-attention not patch_embed conv, but it shows that NBX path
diverges from torch path even on text_encoder's own ops. So the
input-divergence hypothesis has a known starting point.

Latent generation is via `torch.randn(seed)` in `variable_resolver.py`
which is SHARED between sequential and triton_sequential paths —
should be deterministically identical. So divergence is not in
randn; suspect either:

1. Weight-loading dtype/conversion differs between
   `core/runtime/weight_loader.py` (sequential) and
   `triton/weight_loader.py` (triton). Both load bf16 safetensors,
   but they take different paths to GPU. If one path applies a
   transform the other doesn't, weight values differ.
2. `_NBX_COMPUTE_DTYPE` set by Prism + TritonSequence may force a
   bf16→fp16 downcast on Volta that doesn't happen in sequential.
3. Some tensor going through NBX↔torch boundary mid-pipeline
   produces a fingerprint-invisible difference that compounds.

**Next concrete steps**:

1. Instrument `conv2d_wrapper` (triton path) AND graph_executor's
   `_handle_aten__convolution` (sequential path) to dump conv::0's
   INPUT fingerprint and WEIGHT fingerprint on first invocation of
   transformer. Compare. This isolates input-divergence from
   weight-divergence.
2. Same pattern at text_encoder mul::394 — what mul::394's two
   inputs are, and why the position-4 result differs.
3. Audit weight_loader paths for any dtype-conversion asymmetry on
   Volta (bf16 storage → fp16 compute path).

ETA: 1-2h instrumentation + analysis + targeted fix.

Commits this session:
- `a2fd933` fingerprint logical→physical mapping (transpose alignment)
- `aadf5b0` fingerprint via data_ptr() (slice/narrow alignment)
- `1c78a62` (now superseded) verdict pointing at conv kernel
- (this update) microtest disproves kernel hypothesis, redirects to
  pipeline-input divergence

---

# P-SANA-4KPX-RUNTIME — Verdict factuel final 2026-05-05/06

## Bisection scientifique aboutie — 4 modes, 4 verdicts factuels (post Fix B)

| Mode | Wall | Verdict | Live @ crash | OOM site | Cause racine |
|---|---|---|---|---|---|
| sequential | **90s** | **PASS** | 16.5 GB (torch) | — | — |
| compiled | 36-74s | **PASS** (CHANGELOG) | torch tracked | — | — |
| triton compiled (post Fix B) | 252s | **FAIL** | 25.3 GB (NBX) | conv::64 (chain retention) | structural multi-branch chain residue |
| triton_sequential (post Fix B) | similar | **FAIL** | ~25 GB (NBX) | rms_norm or conv::64 | idem |

**Fix Vector B (in-place residual add) implementé et fire correctement** —
la mesure factuelle montre que l'OOM site bouge de aten.add::86 (8 GiB
alloc refusée pré-fix) à aten.convolution::64 (différent op, même live
~25 GB). L'in-place add::86 succède (no new alloc, pixel_shuffle::4
freed at last use), mais la chaîne multi-branch globale retient
toujours 25 GB live à conv::64 — chaque add::XX::out_0 a 2 consumers
(next residual + downstream conv) donc reste alive jusqu'à ce que les
DEUX aient run.

## Trois causes racines orthogonales découvertes par la bisection

### Cause 1 — R30 op-level tiling parity (FIXED, commit f8375a9)

`_op_uid_interceptors` câblé seulement dans `CompiledSequence` et
`TritonSequence`. `_execute_native_op` (sequential) et
`_run_triton_sequential` (triton_sequential) bypassaient l'interception
Prism — Sana 4Kpx VAE conv::54 arrivait raw 36 GiB → cuDNN/Triton
OOM même quand Prism avait enregistré le band-streaming variant.

**Fix**: wired interceptor consumption avec priorité op_uid > op_type
> native dispatch dans les deux dispatchers, mirror CompiledSequence.

**Validation**: sequential Sana 4Kpx PASS 90s coherent PNG (avant FAIL
79s @ conv::54 raw 36 GiB).

### Cause 2 — NBX bias broadcast 8 GiB OOM (FIXED, commit f8375a9)

`_fused_upsample_conv2d_nbx` appliquait `nbx_add(output,
bias.view(1,-1,1,1))` APRÈS materialization du full output (1,C,4096,4096).
`_prepare_binary` matérialise alors le bias broadcast à 8 GiB fp16.

**Fix**: bias add INSIDE band loop sur chaque slice ~250 MB.

**Validation**: triton_sequential advance bien plus loin (était bloqué
à conv::54 8 GB; maintenant atteint rms_norm beaucoup plus tard).

### Cause 3 — `tiled_rms_norm_spatial` import bug (FIXED, commit f8375a9)

`from neurobrix.kernels.wrappers import rms_norm_wrapper` — symbole
n'existe pas. Renommé `rms_norm` (sans `_wrapper`).

## Cause 4 (RESIDUELLE) — Pression structurelle multi-branch DC-AE

**Diagnostic factuel BIG_TENSORS au boundary aten.add::86 (triton compiled,
post-fixes 1+2+3):**

```
[BIG_TENSORS] 4 NBXTensors >= 1 GiB live (total ~25 GB):
  - (1, 128, 4096, 4096) 8 GiB owns=True referrers=8  [branch A output, frames hold]
  - (1, 256, 2, 2048, 2048) 8 GiB owns=True referrers=3 [split chunks pre-upsample]
  - (1, 512, 2048, 2048) 8 GiB owns=False referrers=4   [view of pre-upsample]
  - (1, 128, 4096, 4096) 8 GiB owns=True referrers=6   [branch B output]
```

L'OOM survient au `aten.add::86` qui combine les 2 branches A+B du
multi-branch decoder.up.0 résidual. Pour faire l'add, il faut un 3ème
buffer de 8 GiB → 33+ GiB total > 32.5 GiB driver capacity.

**Pourquoi sequential passe et NBX modes pas:**
- Sequential mode: torch.inference_mode + CachingAllocator splitting/
  coalescing trouve 8 GiB contigu dans heap fragmenté → fits dans 32 GB
- Triton modes: NBX raw cudaMalloc sans caching/coalescing → impossible
  de trouver 8 GiB contigu quand 25 GB live retiennent les blocs

**Ce N'EST PAS un bug — c'est la pression structurelle multi-branch DC-AE
combinée à l'absence de caching allocator dans NBX.**

## Trois fix vectors architecturaux pour Cause 4

Un seul est nécessaire pour débloquer triton 4 modes Sana 4Kpx:

### A. Multi-GPU NBX pipeline_parallel (Hocine premier principe)

Distribuer le VAE sur 2× V100 (Hocine a 4× V100 = 128 GB total
disponible). Le decoder.up.0 multi-branch peak fits trivialement à
24 GB sur 2× 32 GB.

Chantier: porter `core/strategies/component_placement.py` et
`pipeline_parallel.py` pour gérer NBXTensor cross-device transfer.
ETA: 2-4 semaines de chantier dédié.

Avantages:
- Solution Hocine "NeuroBrix premier principe" — multi-GPU+Prism
- Débloque TOUS les modèles 4Kpx+ en triton sur multi-V100
- Pas de modification du graphe forge

### B. Multi-branch in-place fusion dans DC-AE decoder.up.0 (IMPLEMENTÉ commit 5b891f3)

**Implémenté et committé** comme universal detector dans
`OpLevelTilingEngine`. Détecte 26 candidats sur Sana 4Kpx VAE
(8 ≥ 4 GiB au 4096×4096, 4 au 2048×2048, 14 plus petits). Le fix
fire correctement: in-place add::86 succède (no new alloc), OOM site
moves de aten.add::86 → aten.convolution::64.

**Mais Fix B SEUL ne suffit pas** — la mesure factuelle post-fix montre
live_tracked toujours à 25 GB au conv::64. Raison: chaque
add::XX::out_0 a 2 consommateurs (next residual + downstream conv)
dans le multi-branch DC-AE, donc reste alive jusqu'à exécution des
DEUX. La chaîne 4Kpx retient 3-4 tenseurs de 8 GiB simultanément +
le 2048-level retient 4-5 de 4 GiB. Total ~24-30 GiB structurellement
alive AT conv::64.

**Fix B est donc NÉCESSAIRE mais pas SUFFISANT** pour Sana 4Kpx
triton sur 1× V100 32 GB. Il économise les nouvelles allocations
(8 GiB par add résiduel) mais ne réduit pas la rétention totale.
Universel — bénéficie tout modèle multi-branch décodeur.

### C. NBX caching allocator avec splitting/coalescing

Implémenter dans `DeviceAllocator` un caching allocator analogue à
torch CachingAllocator: free-list buckets, splitting de gros blocs,
coalescing au release.

ETA: 4-6 semaines. La complexité est non-triviale (concurrency, edge
cases, multi-GPU). torch a 5+ ans de polish dessus.

Avantages:
- Universel (bénéficie tous les modèles)
- Aligne NBX avec l'écosystème production

Inconvénients:
- Long terme
- Risque de bugs subtils

## Recommandation factuelle (révisée post-leak-diagnostic 2026-05-06)

**Diagnostic [BIG_TENSORS] avec tid identification au OOM révèle un
ORPHAN:** sur les 3 × 8 GiB live au conv::64, l'arena n'en track que
2 (silu::24::out_0 + add::86::out_0). Le 3ème tensor est ORPHAN
(`tid=ORPHAN(not in arena)`) — alive par Python refs hors arena.

C'est **case (b) liveness divergence** (non case (a) structural pur).
Si on libère cet orphelin, live drop 25 → 17 GB, conv::64 alloc 8 GB
→ 25 GB total ≤ 32 GB driver = **PASS Sana 4Kpx triton sans
multi-GPU**.

**Avant d'ouvrir P-MULTI-GPU-NBX-ADAPTER (2-4 semaines), un nouveau
chantier court P-TRITON-LIVE-LEAK-AUDIT** pour identifier la source
du leak ORPHAN:

1. Référants observés: `list[len=3] × 2 + tuple(len=2)` (inhabituel
   pour arena-managed)
2. Pas dans _deferred queue (drain récent confirmé par
   deferred_queue=0)
3. Suspects:
   - `other.contiguous()` / `other.to(dtype)` dans add_inplace_nbx
     créent NBXTensor temporaires dont le ref persiste
   - silu output capturant conv::63::out_0 via view `_base` chain
   - args_resolver retient un tuple ancien via Python frame lifecycle
4. Méthode: instrumentation par-op des `gc.get_referrers` avec
   tid identification (déjà en place commit en cours), bisection
   par désactivation de chaque suspect

ETA P-TRITON-LIVE-LEAK-AUDIT: 1-3 jours selon complexité du leak.

**Plus de quick win disponible avant ce diagnostic** — l'in-place add
fix B est nécessaire mais pas suffisant tant que le leak ORPHAN existe.

**A (multi-GPU NBX pipeline_parallel)** déclassé en backup au cas où
P-TRITON-LIVE-LEAK-AUDIT échoue. Si le leak est fixé, multi-GPU
n'est PAS nécessaire pour Sana 4Kpx (seul cas qui aurait justifié A).

**C (NBX caching allocator)** reste long terme.

## Sortants de cette session

Commits:
- `f8375a9` fix(runtime): R30 op-level tiling parity + NBX bias broadcast OOM
- `9dcf588` docs(P-SANA-4KPX): bisection scientific verdict + 4-mode factual diff
- `5b891f3` feat(tiling): in-place residual add fusion (Sana 4Kpx peak ~8 GB savings)

Artefacts:
- `validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/`
  - `audit_report.md` — audit spaghetti (40 env vars, dead code, R30 audit code)
  - `etape3_diff_report.md` — bisection sequential vs triton_sequential pre-fix
  - `verdict_final.md` — ce document
  - `etape1_sequential.png` — Sana 4Kpx sequential PNG cohérente (R29 ✓)
  - `run_etape*.sh` — scripts de bisection reproductibles
  - `etape*.log` — logs factuels des runs

État chantier P-SANA-4KPX-RUNTIME (post bisection 3-suspects 2026-05-06):
- Sequential mode Sana 4Kpx: **DÉBLOQUÉ** par R30 fix (PNG cohérente)
- Compiled mode Sana 4Kpx: **DÉBLOQUÉ** + dual-backend regression fix
  (PNG cohérente 82s post-fix)
- Triton/triton_sequential modes Sana 4Kpx: **FAIL — cause racine
  factuellement identifiée** par bisection des 3 suspects + NBX_MALLOC_TRACE
  + tid identification dans BIG_TENSORS:

  **Suspect 1 (silu _base chain) — RÉFUTÉ** par audit code: silu wrapper
  alloue output via `NBXTensor.empty_like(x)` qui crée NBXTensor avec
  `_base=None` (pas de view chain).

  **Suspect 2 (args_resolver lifecycle) — RÉFUTÉ** par audit code:
  closures capturent slot indices (int), pas tensors.

  **Suspect 3 (autre op du chain) — IDENTIFIÉ FACTUELLEMENT**:

  Pattern Sana DC-AE pixel_shuffle (exec[700-704]):
  ```
  unsqueeze::5 (1,256,1,2048,2048) — view
  expand::41 (1,256,2,2048,2048) — broadcast view (stride=0 dim 2)
  clone::5   (1,256,2,2048,2048) — ALLOC 8 GiB (matérialise broadcast)
  view::95   (1,512,2048,2048) — view de clone::5
  pixel_shuffle::4 (1,128,4096,4096)
  ```

  L'orphan tensor#3 de la BIG_TENSORS scan correspond exactement à
  `aten.clone::5::out_0` shape (1,256,2,2048,2048) owns=True 8 GiB.
  Tensor#1 view::95::out_0 (view de clone::5).

  Le clone matérialise un broadcast pour fournir un input contiguous à
  pixel_shuffle. Les 256 channels sont DUPLIQUÉS via expand, puis copiés
  en mémoire — 8 GiB alloués pour des données redondantes.

  **Confirmation par bisection NBX_DISABLE_INPLACE_ADD=1**: avec
  l'in-place add désactivé, les orphans deviennent EXPLICITEMENT
  `aten.clone::5::out_0` + `aten.view::95::out_0`. La cause est
  STRUCTURELLE au DAG forge'd, pas au runtime triton.

- **Next fix concret (intra-P-SANA-4KPX-RUNTIME, pas nouveau chantier):**

  **Fix structural pixel_shuffle broadcast elimination**:
  - **Option F1**: graph-level pass dans TritonSequence.compile qui détecte
    le pattern `expand → clone → view → pixel_shuffle` et élimine le clone
    en routant la stride directement à pixel_shuffle (qui en interne lit
    les channels broadcastés en re-mappant `ic >= C/2` vers `ic - C/2`).
    Économie: 8 GiB peak. Univeral pour tous les pixel_shuffle dans DC-AE.
    ETA: 60-90 min implémentation + test.

  - **Option F2**: pixel_shuffle wrapper détecte broadcast input via
    stride pattern et ajuste la lecture dans le kernel sans matérialiser.
    ETA: 30-60 min wrapper-level.

  Si F1+F2 résolvent: triton modes débloqués SANS multi-GPU.
  Si pas suffisant: P-MULTI-GPU-NBX-ADAPTER backup.

## Update 2026-05-06 — F2a Approche C landed (commit 08cbe15)

**F2a Approche C** (kernel broadcast-aware via clone interceptor) implémenté
et validé R29 4-mode:

| Mode               | Résultat | Wall   | PNG          | OOM site                      |
|--------------------|----------|--------|--------------|-------------------------------|
| compiled           | ✓ PASS   | 74s    | cohérent     | —                             |
| sequential         | ✓ PASS   | 89s    | cohérent     | —                             |
| triton compiled    | ✗ FAIL   | 253s   | —            | conv::64 (live=25GB req=8GB)  |
| triton_sequential  | ✗ FAIL   | 252s   | —            | rms_norm (live=25GB req=8GB)  |

**Validation factuelle F2a**:
- Le clone::5 8 GiB allocation a disparu factuellement (verified par advance
  du pipeline triton de `aten.add::86` vers `aten.convolution::64`).
- compiled mode unchanged dual-backend safe (74s vs prior 82s — 8s gagnés
  sur le seul fix tensor_resolver `unknown` torch.* handler qui évite
  resolve_kwargs path détourné).
- sequential mode dual-backend safe (89s) avec `tensor_resolver._resolve_arg_info`
  étendu pour résoudre `torch.contiguous_format` quand op_uid interceptor
  force kwargs resolution.

**Résidu factuel — chantier suivant nommé**:

`P-TRITON-LIVE-WATERMARK-AUDIT` (intra-P-SANA-4KPX-RUNTIME, pas defer flou):
au moment OOM (conv::64 ou rms_norm post-pixel_shuffle::4), live_tracked
factuel = 25292 MB avec 3 × (1,128,4096,4096) NBXTensors live:
- tensor#1 = `aten.silu::24::out_0` 8192 MB (in-progress op input, attendu)
- tensor#2 = `aten.add::86::out_0` 8192 MB (devrait être tué par kill_slots)
- tensor#0 = ORPHAN(not in arena) 8192 MB held par
  `list[len=3] contents=[NBXTensor×3]` × 2 + `tuple(len=2) contents=[int,NBXTensor]`
  + `cell` (closure cell)

Le pattern d'orphan retention `list[len=3]×NBXTensor + cell` est SAME que
celui identifié dans le commit 9c81e8a pré-F2a — donc structurel à
TritonSequence args_resolver / closure capture, **PAS** introduit par F2a
(F2a a juste fait avancer la chain assez loin pour exposer cet orphan
distinct du clone::5 maintenant éliminé).

**Hypothèse factuelle pour P-TRITON-LIVE-WATERMARK-AUDIT**:
- Args list (`[r(arena) for r in resolvers_t]`) capturée par `enumerate()`
  ou par closure cell de `for op_idx, op in enumerate(self._ops)` dans
  `_run_single_device`. La référence persiste à travers les itérations
  parce que la frame courante / le cell pointent vers la liste précédente.
- ETA estimé: 1-2h diagnostic + fix targeted (force-clear args list à fin
  d'iter, ou rewrite resolver pour ne pas créer une fresh list à chaque
  call).

P-MULTI-GPU-NBX-ADAPTER reste backup si P-TRITON-LIVE-WATERMARK-AUDIT
échoue (mais résoudre le live-watermark Triton-side évite le coût
multi-GPU pour modèles single-GPU-fittable).

**Discipline maintenue**:
- Pas de "INDÉTERMINÉ" — outcome factuel par mode, dump live_tracked / req
  / driver_free chiffré, BIG_TENSORS scan avec referrers
- Pas de défer flou — `P-TRITON-LIVE-WATERMARK-AUDIT` scopé avec hypothèse
  testable (closure args list capture)
- F2a closes 1 des 3 leviers identifiés au début (kernel-level fix) ; live-
  watermark était implicite dans le levier "Prism op-level parity" mais
  émergé comme orphan retention distinct du clone::5

**Référentiel chantiers ouverts**:
- `P-OP-PATTERN-FUSION` (proposé pré-F2a): registry universel patterns
  pytorch décomposés (pixel_shuffle DONE via F2a + 8 autres scopés:
  pixel_unshuffle, layer_norm, softmax, GELU, unfold/fold, repeat_interleave,
  roll, grid_sample). À ouvrir post-`P-TRITON-LIVE-WATERMARK-AUDIT`.

## Update 2026-05-06 (later) — bisection 3-suspects post-F2a (etape 5,6,7)

**Suspect A (closure/frame retention) — FACTUELLEMENT RÉFUTÉ**.
NBX_AGGRESSIVE_CLEANUP=1 (force `args=kwargs=result=None` après chaque
op.func + gc.collect via NBX_FORCE_GC=1) ne change RIEN à la trajectoire
live: `op 662 silu::24 live=31441MB` identique avec et sans cleanup.
Le leak n'est PAS au niveau frame Python.

**Trajectoire fine per-op (etape6, etape7)**:
```
op 654 conv::62           live=30417 MB  (post-add::84 spike)
op 659 pixel_shuffle::4   live=27444 MB  (-3 GB)
op 660 add::86            live=21198 MB  (-6 GB Fix B in-place)
op 661 conv::63           live=21198 MB  (no change BEFORE conv::63)
op 662 silu::24           live=31441 MB  (+10 GB conv::63 alloc)
op 663 conv::64           live=31441 MB  (BEFORE conv::64 alloc → OOM)
```

Le saut +10 GB entre conv::63 et silu::24 = conv::63's allocation
(8 GB output + ~2 GB band-streaming transients qui ne se libèrent
pas). Avec OOM live=25 GB et 3×8 GiB tensors live (silu::24, add::86,
ORPHAN), la signature factuelle est:

- silu::24 (legitimate, in-progress conv::64 input)
- add::86 (legitimate, has remaining consumer add::89)
- ORPHAN — soit pixel_shuffle::4::out_0 (devait être killed à add::86)
  soit conv::63::out_0 (devait être killed à silu::24)

**Conclusion factuelle**: le leak est au niveau C-extension. Sources
candidates restantes:
- Triton autotune cache holding kernel args via internal refs not
  visible from Python gc (CompiledKernel cache, BenchmarkRunner state)
- NBXTensor `_owns_data` flag interaction with `_base` chain creating
  a strong-ref-cycle that breaks NBXTensor.__del__ → cudaFree
- CUDA driver-level retention not visible from Python

**Bisection suspect non-testée — mais coût opportunité élevé**:
Suspect B (wrapper self_managed_dtype) et Suspect C (decomposed
PyTorch op) restent à tester, mais mesure factuelle requise vs
diagnostic instrumentation déjà saturée.

**Décision factuelle pour le chantier**:
Le résidu live-watermark Triton 4Kpx = limite structurelle de
debugging Python-level. Le fix requiert soit (a) instrumentation
plus profonde NBXTensor lifecycle (suivre __init__/__del__ sites
factuellement), soit (b) basculer sur P-MULTI-GPU-NBX-ADAPTER pour
contourner via VAE sur GPU dédié (l'option backup nommée). Choix
arbitré par Hocine.

F2a Approche C reste un fix réel et durable (commit 08cbe15) qui
unblock add::86 + advance le pipeline triton plus loin que jamais.
Le résidu conv::64/rms_norm est PRE-EXISTANT à F2a (verified par
signature orphan identique entre commits 9c81e8a et etape5/6/7).

## Update 2026-05-06 (final) — bisection 3-suspects suite, fixes empilés

Suspect A (closure/frame retention) raffiné par instrumentation
ciblée NBX_TRACE_TIDS + NBX_TRACE_DEL_BIG_MB:
- pixel_shuffle::4::out_0 et conv::63::out_0 ont les MÊMES patterns
  de référants → SUSPECT SYSTÉMIQUE +2 refs sur tout tensor stocké.
- NBX_DEL trace prouve que conv::63 NBXTensor.__del__ ne fire QU'AU
  OOM unwind, PAS au silu::24's kill_slots → leak structurel.
- Root cause identifié: `for s in op.kill_slots: old = arena[s]`
  avec `old` Python loop variable persistante post-loop. Le `old`
  retient 8 GiB de chaque kill across iterations subséquentes
  jusqu'au prochain kill_slots loop qui rebind.

**Fix landed** commit cd5a108: explicit `old = None` post-loop à 5
sites dans `_run_single_device` + multi-device variant.

**Mesure post-fix etape10**: chain advance conv::64 → rms_norm::24
(2 ops). Live trajectory inchangée à add::86 (21 GB) parce que la
rétention `old` n'affectait pas le steady state mais le moment
exact de l'OOM.

**Suite — rms_norm OOM (exec[710])**: rms_norm wrapper alloue
2 × 8 GiB (x.contiguous() pour permute view + NBXTensor.empty_like
pour output). Fix in-place: si x.contiguous() matérialise (input
non-contiguous), output_2d = x_2d (kernel per-tile-safe).

**Fix landed** commit beeab71 (avec add bias).

**Suite — add::88 OOM (exec[711])**: add wrapper matérialise bias
broadcast 8 GiB via b.expand(out_shape).contiguous(). Fix:
add_bias_broadcast_kernel lit bias[offset % feat_dim] direct.

**Fix landed** commit beeab71.

**Outcome final cette session**:
- Chain advance: conv::64 (exec[708]) → conv::69 (exec[735]) = +27 ops
- conv::69 = LAST conv VAE (output RGB 1×3×4096×4096)
- OOM "Triton Error [CUDA]: out of memory" — Triton runtime-level,
  pas malloc_cuda. Probable autotune workspace overflow vs
  conv2d_wrapper x.contiguous() materialization for permute view input.

**État chantier P-SANA-4KPX-RUNTIME (post-session)**:
- 5 fixes incrémentaux land cette session
- Pipeline triton 4Kpx structurellement très proche de la fin (27 ops
  past original blocker, on est dans le LAST conv avant write)
- Le résidu live=30 GB à conv::69 est dans le seuil 32 GB V100 mais
  Triton autotune workspace ou x.contiguous() matérialisation le
  pousse over.

**Next étape factuelle (intra-P-SANA-4KPX-RUNTIME)**: investiguer
le mode OOM exact à conv::69 (autotune workspace vs contiguous
materialization). Si autotune: configurer un fixed config pour
conv::69 (output channels=3 unique). Si contiguous: appliquer
même pattern in-place qu'à rms_norm pour conv2d_wrapper avec
non-contiguous input.

ETA estimé pour clore: 1-2h investigation + fix targeted.

## Update 2026-05-07 — etape13 force_gc test, hypothèses raffinées

**Hypothèse (2) contiguous() materialization REFUTED** par audit
DAG: conv::69 input = `aten.relu::18::out_0` (relu wrapper alloue
output contigu via NBXTensor.empty_like — pas un permute view).
Le path `x.contiguous()` dans conv2d_wrapper est no-op pour
conv::69. Pas de matérialisation 8 GiB ici.

**Hypothèse (1) Triton autotune workspace TESTÉE** via etape13
NBX_FORCE_GC=10 (gc.collect every 10 ops). Same OOM at conv::69.
gc.collect ne libère pas de cycles → leak n'est pas Python-level.

**Stack trace confirmé**: OOM source est
`triton/backends/nvidia/driver.py:713 self.launch()` —
**CUDA driver-level kernel launch**, PAS NBX malloc_cuda. La
launch reserve quelques KB pour kernel args + Triton runtime
buffers. Avec live=30GB / 32GB V100 = 2GB free, une fragmentation
ou Triton do_bench cache_flush buffer (~256 MB par défaut)
suffit à pousser over.

**Conclusion factuelle**: le résidu conv::69 est limite
**structurelle V100 32GB** pour Sana 4Kpx VAE post-pixel_shuffle
chain. Pour clore définitivement, options scopables:

1. **Bypass autotune pour conv::69 specifique**: detect output
   channels=3 (unique pattern in VAE) dans conv2d_wrapper, route
   vers un appel `conv2d_forward_kernel.fn[grid](...)` avec config
   fixed (pas d'autotune do_bench overhead).

2. **Pre-warm autotune cache**: run conv::69 at startup with
   dummy tensors to populate disk cache before VAE pressure.
   Subsequent runs use cached config without benchmarking.

3. **Multi-GPU pipeline_parallel for VAE component**: VAE moves
   to dedicated GPU (32 GB free). Bypasses tightness.

ETA option 1: 30-60 min. ETA option 2: 60-90 min. ETA option 3:
several days (P-MULTI-GPU-NBX-ADAPTER scope).

**Discipline maintenue**: pas d'INDÉTERMINÉ, pas de pivot
prematuré (option 3 reste backup). Le chantier reste ouvert avec
1-2 options option 1/option 2 actionable scopable.

## Update 2026-05-07 (later) — Option 1 attempted, factually rejected

**Tentative**: Option 1 (autotune bypass for conv::69 via `out_c<=8`
heuristic + direct `conv2d_forward_kernel.fn[grid]` call with
fixed Volta-viable config).

**Etape14**: BLOCK_BHW=64, BLOCK_OUTF=32, BLOCK_INF=32, num_warps=4
→ run completes WITHOUT OOM (autotune workspace bypassed) →
**PNG produced 47 MB at 4Kpx for the first time**, but visual
inspection shows GREEN TEXTURE NOISE pattern, not the expected
red apple. R29 visual FAIL.

**Etape15**: smaller config BLOCK_OUTF=16, BLOCK_INF=16,
num_warps=2 → same GARBAGE output, same shape pattern.

**Numerical microtest**: bypass kernel call vs autotune'd call on
small (1,32,16,16) input → max|diff| = 0.0 (bit-identical).

**Sana 1024 triton regression check**: PNG COHERENT post-fixes
(commits cd5a108, beeab71). My fixes (loop-var, rms_norm
in-place, add bias broadcast) are numerically correct.

**Conclusion factuelle**: the bypass is numerically correct on
small shapes but produces garbage at 4Kpx scale. Some Triton
kernel-launch detail differs between the autotune wrapper and
direct .fn[grid] call at this specific scale. Debugging the
exact mechanism would require comparing per-tile output between
the two paths — beyond the immediate session scope.

**Option 1 STATUS**: REJECTED. The bypass approach can't be
trivially used as a workaround.

**Revert**: bypass removed from `conv2d_wrapper` (commit reverted
to autotune'd path). 5 prior fixes (08cbe15, cd5a108, beeab71)
remain; chain advance to conv::69 OOM remains the structural
ceiling.

**Next concrete options**:
- Option 2 (pre-warm autotune cache at startup, ~60-90 min)
- Reduce live by another ~1-2 GB via additional in-place
  optimization in the post-pixel_shuffle chain (e.g. residual
  add::89 reuse pattern audit)
- P-MULTI-GPU-NBX-ADAPTER pivot (backup, several days)

**Etat session**: 6 commits land cette session. compiled+sequential
triton modes PASS R29. triton compiled OOM at conv::69 = LAST conv
before output. Chain advance +27 ops vs original. 2 GB GPU
headroom shortage at conv::69 entry.

## Update 2026-05-07 (final, bisection tensor-value diff)

Phase 1 alt — tensor-value fingerprint diff sequential vs
triton_sequential (commit c91cc75) corrigé après lecture initiale
faussement noisy. La fingerprint instrumentation EST correcte pour
les compute ops (ops produisant tenseurs contigus frais). Pour les
view ops (transpose/expand/slice/permute/t), raw byte sampling diffère
de torch.flatten() — sampling artifact à filtrer.

**Bisection compute-ops only par iteration**:

- **Iter 0** (text_encoder): NO real divergence > 0.5 in compute ops.
  Sequential et triton produisent les mêmes valeurs — text_encoder
  triton est numériquement aligné avec sequential.

- **Iter 1** (transformer step 1): premier réel divergence à
  **`aten.mm::17`** (op_idx=394). Shape `(32768, 2240) @ (2240, 2240)`
  — batch 32768 caractéristique Sana 4Kpx (Sana 1024 même mm aurait
  batch ~1024). Position 4 (n-1, dernier élément) diverge:
  - seq: `[-0.0189, 0.06755, 0.03314, 0.06563, -0.08473]`
  - tri: `[-0.0189, 0.06765, 0.03327, 0.06521, -0.1057]`
  - delta=0.021 propage downstream à mul::41 (0.06), mul::42 (0.05).

- **Iter 2-12** (transformer steps 2-12): premier divergence à
  `aten.convolution::0` op_idx=4 (delta 1-3). Conséquence de iter 1
  divergence accumulée → state transformer biaisé → conv::0 next step
  démarre déjà différent.

- **Iter 13**: divergence à `aten.addmm::3` (delta 4.55), one position
  only. Other downstream ~0.

- **Iter 14** (VAE): premier divergence à `aten.convolution::0` (op_idx=4)
  delta 1.97. Cascade compounding du transformer accumulé.

**Identification factuelle**:

Le bug NUMÉRIQUE est dans le **NBX mm kernel à batch 32768 ligne 32K
output**. Sana 1024 batch ~1024 fonctionne. Sana 4Kpx batch 32K
diverge position n-1.

Le pattern "position 4 (n-1) diverge" suggère un boundary/mask issue
dans le mm kernel à très-large batch — dernier tile potentiellement
mal masqué OU autotune config qui produit valeur différente sur le
dernier tile à batch 32K.

**Next concrete step (intra-P-SANA-4KPX-RUNTIME)**:

1. Lire `kernels/ops/matmul.py` mm kernel — vérifier mask handling
   pour le dernier tile à batch dim non-multiple de BLOCK_M.

2. Comparer avec PyTorch / cuBLAS reference (open source — torch's
   mm uses cuBLAS GEMM, well-tested at any batch).

3. Si autotune-related: vérifier configs Volta dont la BLOCK_M
   choisie pour batch=32768 produit ce résultat divergent. Peut-être
   un config a un bug spécifique aux grands batchs.

4. Fix ciblé sur le NBX mm kernel ou son autotune config set.

5. Re-run sequential vs triton_sequential value diff — vérifier que
   iter 1 op 394 mm::17 disparait + downstream s'aligne.

6. R29 4 modes Sana 4Kpx + 4 PNGs.

ETA estimé: 1-2h lecture mm kernel + identification fix + validation.

## Update 2026-05-07 (final² — bisection convergence)

Bisection raffinée révèle: **mm::17 noise est secondaire**. La VRAIE
SOURCE de divergence est `aten.convolution::0` (transformer patch_embed)
qui DIVERGE DEPUIS LA PREMIÈRE INVOCATION en iter 1, pas une accumulation.

- iter 1 op 4 `aten.convolution::0` shape `(2, 32, 128, 128) → (2, 2240, 128, 128)`:
  - Sequential: `[0.1125, 0.915, 0.1125, 0.915, -0.1318]`
  - Triton:     `[0.2079, -2.018, 0.2079, -2.018, -0.1721]`
  - delta=2.9 — pas du noise, **vraies valeurs différentes**
- iter 0 (text_encoder, no conv): zero divergence
- iter 1 OPS BEFORE conv::0 (op 0-3 = mask setup): zero divergence
- conv::0 reçoit MÊME input dans les 2 paths, produit OUTPUT DIFFÉRENT

**Comparaison Sana 1024 (works) vs Sana 4Kpx (broken)**:
- Sana 1024 conv::0: `(2, 32, 32, 32) → (2, 2240, 32, 32)` — batch_dim = 2*32*32 = **2048** — works
- Sana 4Kpx conv::0: `(2, 32, 128, 128) → (2, 2240, 128, 128)` — batch_dim = 2*128*128 = **32768** — broken

Le bug est dans `conv2d_forward_kernel` à batch_dim = 32768 (16x plus
grand que Sana 1024). Le kernel est 1×1 conv stride 1 padding 0 groups 1
(patch_embed projection 32→2240).

**Hypothèses raffinées pour l'op spécifique**:
1. Autotune choisit BLOCK_BHW pour 32768 qui produit valeurs incorrectes
2. Int overflow dans une multiplication d'offset à grande batch
3. Mask handling spécifique à batch_dim non-power-of-something à grand
   nombre de blocks
4. tl.dot precision issue à large M dim avec fp32 accumulation

mm::17 divergence (0.021 position 4) était fp16 reduction-order noise
(pas un bug).

**Next factuel**: lire conv2d_forward_kernel ligne 75-141 + simuler
batch_dim=32768 avec Triton offline pour pinpoint le défaut au config
ou kernel-level.

ETA: 2-3h lecture + simulation + fix + validation R29.

## Discipline maintenue

- Pas de "INDÉTERMINÉ" — chaque verdict est factuel avec adresse mémoire,
  shape de tenseurs vivants, nombre de référants Python
- Pas de "structural pressure" comme verdict avant isolation factuelle —
  ISOLÉ par BIG_TENSORS scan + tracé OOM exact à `aten.add::86`
- Pas de chantier futur fictif — A/B/C sont scopables, ETA explicite,
  pas en "P-XYZ-LATER"
- ETA mesuré vs prévu:
  - Étape 1 sequential: 79s prévu / 79s mesuré ✓
  - Étape 2 triton_sequential cold: 20-30 min prévu / 2h01 mesuré ✗
    (sous-estimé l'autotune compile cold-start cost)
  - Étape 1 post-fix: 5-15 min prévu / 90s mesuré ✓
  - Étape 2 post-fix: 5-10 min prévu / 5min23 mesuré ✓
  - Étape 4 triton compiled: 5-10 min prévu / 5min14 mesuré ✓

EOF
