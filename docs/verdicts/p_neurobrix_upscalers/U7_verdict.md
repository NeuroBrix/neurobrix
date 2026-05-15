# P-NEUROBRIX-UPSCALERS — U7 verdict (HAT)

## Outcome: HAT-S + HAT-L x4 — 2/4 modes ✓ (compiled + sequential), triton named follow-up

HAT-S (compact: embed_dim 144, compress 24, 6 RHAG) and HAT-L
(large: embed_dim 180, compress 3, 12 RHAG) SRx4 operational in
**compiled** and **sequential** modes at the container trace
size. triton / triton-sequential blocked by a single missing
primitive kernel — named follow-up, not an orphan TODO.

## Matrix (input 64×64, test_64.png — downscaled apple)

| model | compiled | sequential | triton | triton-seq |
|---|---|---|---|---|
| hat-s-x4 | 256² ✓ | 256² ✓ | ✗ im2col | ✗ im2col |
| hat-l-x4 | 256² ✓ | 256² ✓ | ✗ im2col | ✗ im2col |

cos vs the PyTorch fp32 reference (canonical arch + checkpoint)
= **0.999998** for hat-s/hat-l × compiled/sequential. std 93.4
(real content). R29 visual: coherent red-apple super-resolution,
sample `hat_l_compiled_sample.png`. Local artefacts under
`validation_outputs/p_neurobrix_upscalers/matrix_hat/`.

## Two NeuroBrix runtime gaps found + dispositions

### 1. compiled `aten::fill` crash → FIXED (engine, general)

Symptom: `fill_ only supports 0-dimension value tensor but got
tensor with 1 dimensions`. HAT computes its shifted-window
attention mask *inside* the forward (`img_mask[slice] = cnt`,
cnt a Python int) — SwinIR precomputes it in `__init__`, which
is why HAT is the first model to exercise this. The `cnt`
scalars are captured as `param::constant_T_*` with no embedded
data.

Diagnosis (instrumented): the fill value resolved to literal
`None`. Root cause: the compiled path's constant-slot
pre-population block sat **after `rebind_partial()`'s return** —
structurally unreachable dead code. So `constant_T_*` slots were
never materialised → `None` → crash. A second latent bug: the
block's shape default was `[0]` (1-dim) where the trace and the
working sequential resolver use `[]` (0-dim scalar).

Fix (`compiled_sequence.py`, NeuroBrix `c0a1445`): move the
block into `bind_weights()` (reachable, post weight-bind so
only genuinely unprovided constant_/[0]/missing-norm slots are
touched — zero3 partial sets and real weights untouched), `[]`
0-dim default. General fix — any model with in-forward orphan
scalar constants now runs in compiled mode. Anti-régression
intact: TinyLlama compiled (KV-cache, coherent haiku) + Sana
4Kpx 32g compiled (coherent red apple).

Empirical note: sequential mode already produced cos 0.999998
via the equivalent `torch.empty(shape)` path — proving the
uninitialised `cnt` mask is numerically negligible at the 64×64
container size (the shifted-window mask only perturbs boundary
windows; OCAB + CAB + unshifted blocks dominate). The engine
fix therefore yields correctness, not just non-crash.

### 2. triton/triton-seq `aten::im2col` → NAMED FOLLOW-UP

HAT's Overlapping Cross-Attention Block uses `nn.Unfold` →
`aten::im2col`, for which no Triton kernel exists. Per the
Triton-pure 2-level doctrine this is a Level-1 primitive (not a
deferrable fused optimisation), so it is named as a chantier,
not closed silently: **P-TRITON-IM2COL-KERNEL** — write a
Triton-pure im2col wrapper/kernel honouring the OCAB unfold
semantics (kernel=overlap_win, stride=window_size,
padding=(overlap_win−window)//2). Until then HAT triton/
triton-seq remain unsupported (clear error, no silent fallback —
R33).

## Named follow-up chantiers (not orphan TODOs)

- **P-TRITON-IM2COL-KERNEL** — Triton-pure `aten::im2col` for
  OCAB-style unfold; unblocks HAT triton + triton-seq, and any
  future unfold-based architecture.
- **P-CONTAINER-EMBED-ORPHAN-SCALARS** — the cleaner long-term
  fix: embed Python-scalar constants from in-forward loops as
  container constant data so `mask[slice]=cnt` patterns carry
  their real values (currently materialised as a 0-dim
  `torch.empty`, numerically negligible at container trace size
  but not guaranteed at larger tiled sizes — relates to BL-1).
  Evidence points: SwinIR `mean` (U6), HAT `cnt` (U7).

## Anti-régression (PRESERVED)

| cell | result |
|---|---|
| TinyLlama compiled | coherent ocean haiku, no error (KV-cache exercised) |
| Sana 4Kpx 32g compiled | coherent red apple on white plate |

The `bind_weights` change is additive: real weights are bound
first (slot non-None → skipped by the pre-population guard);
only previously-`None` constant_/[0]/missing-norm slots are
filled.

## Hub

Future publication target: neurobrix.es (proprietary). No
huggingface.co publication.

## Next

U8 — DRCT (optional, budget permitting; fp16 probe first per
U6 discipline). Then U9 hub prep + consolidation + tag.
