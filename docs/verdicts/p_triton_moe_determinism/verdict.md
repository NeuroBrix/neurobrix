# P-TRITON-MOE-DETERMINISM — verdict (closed on delivered acquis, 2026-05-19)

Branch `p-triton-moe-determinism`. Two sub-chantiers delivered;
end-to-end Qwen3-30B::triton determinism NOT achieved (residual root
cause is elsewhere — handed to a named follow-on chantier). Closure is
honest: it claims only what is done and proven.

---

## Section 1 — Delivered acquis

### Sub-chantier 1 — R30 `_ACCUMULATOR_OPS` parity (commit `dcc7b98`)

The default engine `core/runtime/graph/compiled_sequence.py:50-53`
treats the MoE-aggregation aten ops `{scatter_reduce, scatter_add,
index_add, scatter, index_put}` as NOP-propagation accumulators: when
a deactivated expert feeds `args[0] = None`, the op passes the base
accumulator through instead of nulling the output slot. The triton
mirror `triton/sequence.py:142` (`_ACCUMULATOR_OPS`) only listed
`{add, mul}` → the deactivated-expert path was nulled in `--triton`,
a silent R30 asymmetry vs the reference oracle. Fixed: the triton set
now mirrors the compiled bare-name set (extends, does not replace, the
existing `{add, mul}`).

Experimental backing (counterfactual, prior session): with the R30
fix applied, Qwen3-30B::triton greedy 16-tok run 3× —

- CF1: `Okay, the user is asking about the capital of France. This seems like a`
- CF2: `Okay, the user asked, "The capital of France is" followed by "`
- CF3: `Okay, the user is asking, "The capital of France is..." Hmm,`

3 distinct → the asymmetry fix is real R30 debt (shifts Qwen3 toward
the compiled-correct attractor; the NOP branch WAS exercised and
mishandled) but is NOT the determinism cause. It ships on its own
merit.

Anti-regression: DeepSeek-MoE::triton stays GREEN with the fix
(RC 0, coherent `Hello! How can I`) — no MoE regression on the
obvious co-occurrence.

### Sub-chantier 2 — deterministic `index_add` (commit `feb6103`)

`kernels/ops/index_add.py` used `tl.atomic_add`, whose inter-block
float accumulation order is non-deterministic. Replaced with a
**deterministic output-owner gather**: one program owns each
`(outer, dest, inner-tile)` output cell and computes
`out = x + Σ_{j: index[j]==d} alpha*src` as a 0-initialised
ascending-`j` fp32 *sequential* fold (scalar j-loop — a vectorised
`tl.sum` tree reduce rounds differently and breaks bit-exactness; the
feature dim is vectorised). Determinism is structural: one writer per
cell, fixed order, zero atomics, no sort dependency.

The "segment-sum then add to base" shape (not fold-into-base) was
empirically required: discriminating probe (base=1, src=[1e8,-1e8])
→ torch yields 1.0 (`base + Σ`), not 0.0 (`fold-into-base`).

Tests: `tests/unit/kernels/test_index_add_deterministic.py`, 42/42
pytest + script-mode ALL PASS:
- 5-run byte-identical reproducibility, fp32/fp16/bf16.
- Torch-deterministic parity: fp32 BIT-EXACT for `alpha=1` (the ONLY
  pattern in transformer graphs — Qwen3-30B-A3B model graph =
  6144/6144 `index_add` at default `alpha=1`), fp16/bf16 ≤ 1 ULP.
- `alpha != 1` (extra coverage, no production path): full determinism
  preserved, ≤ 1 ULP fp32 / ≤ 2 ULP bf16; residual fp32 1-ULP traced
  to Triton codegen of the scaled term (bitcast / fma / reassoc
  barriers all no-effect), NOT the gather algorithm (numpy fp32
  sequential fold == torch bit-exact, reproduced bit-exactly here for
  `alpha=1`).
- Coverage: indices sorted / shuffled / many-dup / repeated-N /
  all-unique, dim!=0, alpha!=1.
- RED confirmed first on the pre-fix atomic kernel (non-deterministic
  + non-parity on collision configs; collision-free configs passed).

**Perf cost**: deferred to P-TRITON-MOE-DETERMINISM-RESIDUAL. A
representative wall-clock delta vs the atomic baseline requires
running Qwen3-30B::triton (~110 s/run cold) against a reverted-atomic
build; and the e2e path is still non-deterministic, so the wall-clock
is not a stable measurement yet (> 15 min, would mis-baseline). The
unit suite proves functional correctness; perf characterisation
belongs with the closed e2e fix.

---

## Section 2 — State of the art consulted (R16)

- **Triton #7402** — `tl.atomic_add` return value is incorrect across
  threads (only thread 0 gets the correct value); open upstream bug,
  tested Triton 3.3.1.
  https://github.com/triton-lang/triton/issues/7402
- **Triton #4717** — `tl.atomic_add` is slow due to layout
  conversions (sometimes slower than `tl.store` + inline asm).
  https://github.com/triton-lang/triton/issues/4717
  Together #7402/#4717 doubly justify removing `atomic_add`
  (correctness AND perf), independent of determinism.
- **flagos-ai/FlagGems** — https://github.com/flagos-ai/FlagGems
  (FlagOS umbrella). FlagGems migrated from `FlagOpen/FlagGems`
  (BAAI) to `flagos-ai/FlagGems` late 2025; joined the official
  PyTorch ecosystem; added `index_add_` as an official op. Repo
  `pushed_at` 2026-05-19 (active). Inspected at HEAD:
  `src/flag_gems/ops/index_add.py` STILL uses
  `tl.atomic_add(..., sem='relaxed')`, with an in-code TODO that
  atomic_add does not support bf16; pins `triton==3.3.0`;
  `tests/test_index_add.py` only `gems_assert_close` (tolerance, not
  bit-exact / not determinism). `src/flag_gems/ops/sort.py` carries
  the identical `tl.core.get_int_dtype(num_bits, signed)` call our
  vendored copy uses. **Retained**: nothing vendored — no
  deterministic upstream `index_add` exists (Case (ii) of the
  mandate: upstream has it but uses atomic_add → keep the idea,
  replace the atomic with a deterministic gather). The upstream
  finding directly shaped the implementation choice (gather, not a
  vendor refresh).
- **PyTorch** — `index_add_`/`scatter_add_` are non-deterministic on
  CUDA via atomicAdd; the deterministic path
  (`use_deterministic_algorithms(True)`) is sort + sequential
  segmented reduce. Empirically reverse-engineered (probes): the
  deterministic order = per-destination 0-initialised
  ascending-source-position fp32 sequential fold added once to
  `self` — verified bit-exact fp32 for bucket sizes up to 4096. This
  directly shaped the gather kernel's accumulation order.
  (PyTorch reproducibility docs / forum discussion on index_add_
  CUDA non-determinism.)

---

## Section 3 — Non-closure e2e: factual proof

Post-fix HEAD `feb6103`, model `Qwen3-30B-A3B-Thinking-2507`, prompt
`"The capital of France is"`, `--temperature 0 --max-tokens 16
--triton`.

3 runs, no seed:
- d1: `Okay, the user asked, "The capital of France is" followed by a`
- d2: `Okay, the user asked, "The capital of France is" followed by "`
- d3: `Okay, the user asked, "The capital of France is" and left it`

2 runs, `--seed 42` (greedy):
- s1: `Okay, the user asked, "The capital of France is" followed by a`
- s2: `Okay, the user asked, "The capital of France is" and then left`

Factual conclusions:
- A fixed seed does NOT yield determinism (s1 ≠ s2) → the residual
  non-determinism is in the **forward pass**, not in sampling/seed
  handling. The sampler is ruled out.
- Identical prompt-echo prefix, divergence at the first low-margin
  generated token → the SAME accumulated-bit-noise greedy-argmax-flip
  signature as pre-fix, now around a shifted attractor (the `dcc7b98`
  R30 fix moved Qwen3 toward the compiled-correct basin).
- The deterministic `index_add` (unit-proven) removed one
  non-deterministic op from the path but not the one(s) that drive
  the e2e flip. The prior session's locked single-cause hypothesis
  (atomic_add in index_add = THE cause) was the best-defended call on
  the data then available; its incompleteness was only provable after
  fixing it and observing the drift survive. Healthy science, not a
  method error.

---

## Section 4 — Static triage of the residual cause

Qwen3-30B-A3B model component graph op distribution (top): `aten::t`
18672, `aten::mm` 18672, `aten::mul` 12578, `aten::unsqueeze` 12531,
`aten::index` 12288, `aten::view` 6626, `aten::select` 6144,
`aten::nonzero` 6144, `aten::unbind` 6144, `aten::silu` 6144,
`aten::index_add` 6144, `aten::slice` 579, `aten::_to_copy` 485,
`custom::rms_norm` 193, `aten::add` 192, `aten::scatter` 48,
`aten::scaled_dot_product_attention` 48, `aten::topk` 48, `aten::bmm`
1.

- `index_add` (6144): FIXED `feb6103`, unit-proven deterministic.
- `nonzero` (6144): `nonzero_op.py` = prefix-sum compaction, no
  atomics → deterministic by construction → ELIMINATED.
- `topk` (48): Qwen3 (128 experts) → single-chunk fast path →
  `topk_stage1_kernel` iterative `tl.max`/`tl.argmax`, deterministic
  given deterministic inputs (Triton argmax = consistent lowest-index
  tie-break) → ELIMINATED as an independent source.
- `scatter` (48): `scatter_op.py:75` uses `tl.atomic_add`, but plain
  `aten::scatter` is overwrite → ELIMINATED unless duplicate indices
  collide on the same slot (to verify in the residual chantier).
- **Open suspects** (parallel-reduction non-determinism): `mm`
  (18672 — split-K / reduction order in the autotuned matmul),
  `custom::rms_norm` (193 — parallel reduction), `scaled_dot_product_-
  attention` (48 — flash-attn-style reductions), and any other
  parallel-reduction op. The full op list with counts (above) is the
  starting RED surface for the residual chantier.

---

## Section 5 — New latent observations (backlog)

- **P-TRITON-MOE-DETERMINISM-RESIDUAL** (new named chantier). Goal:
  isolate the residual forward-pass non-determinism in
  Qwen3-30B::triton AFTER the `index_add` fix. Method: P-SANA
  run-to-run op-by-op tensor fingerprinting (differential forward
  pass), starting from the Section-4 op list, ranked suspects `mm` /
  `rms_norm` / `SDPA` / `scatter`. RED entry artefacts: the 3 unseeded
  + 2 seed-fixed Qwen3-30B::triton outputs above (baseline). Estimate:
  multi-day on 30B-A3B (~110 s/run cold).
- **P-TRITON-SORT-TRITON36** (REOPEN-CANDIDATE HIGH). `sort_wrapper`
  (`kernels/ops/sort_op.py`) is non-functional under Triton 3.6:
  `_get_int_t` → `tl.core.get_int_dtype(num_bits, signed)` raises
  `ValueError(... _semantic argument must be provided outside of JIT
  functions)`, chaining to `AttributeError("'dtype' object has no
  attribute 'type'")`. A constexpr bits→`tl.intN` shim is
  insufficient (measured). Production call sites broken TODAY,
  independent of MoE: `src/neurobrix/triton/samplers.py:142`
  (`sort_wrapper(logits, dim=-1, descending=True)` — top-p/top-k
  sampling) and `src/neurobrix/kernels/dispatch.py:589`
  (`"sort": w.sort_wrapper`, `aten::sort`). Own named chantier.
- **P-FLAGGEMS-VENDORING-REFRESH**. `src/neurobrix/kernels/
  triton_kernels_ref/` is vendored from the OLD `FlagOpen/FlagGems`
  (BAAI). Upstream migrated to `flagos-ai/FlagGems` (FlagOS, late
  2025; https://github.com/flagos-ai/FlagGems — active into 2026,
  joined PyTorch ecosystem, added `index_add_` as official op).
  Refresh to be planned (R25 strict per-primitive vendoring, record
  provenance URL + upstream commit). NOTE: upstream `sort.py` still
  has the identical fragile `tl.core.get_int_dtype` call and pins
  `triton==3.3.0` — a refresh ALONE does not fix P-TRITON-SORT-TRITON36
  on our Triton 3.6; the real fix is a Triton-3.6 dtype-API
  adaptation. Operators that evolved upstream and may benefit us:
  `index_add` (now official, still atomic — no gain for determinism),
  and the broader op set to diff at refresh time.

---

## Section 6 — Commit SHAs

- `dcc7b98` — sub-chantier 1: R30 `_ACCUMULATOR_OPS` parity
  (`triton/sequence.py` + CHANGELOG).
- `feb6103` — sub-chantier 2: deterministic output-owner gather
  index_add (`kernels/ops/index_add.py`, `kernels/wrappers.py`) +
  `tests/unit/kernels/test_index_add_deterministic.py` + CHANGELOG.
  (Tests are in `feb6103`, not a separate commit.)
- Verdict commit: the commit that adds this file (SHA reported in the
  closure confirmation; tagged
  `p-triton-moe-determinism-r30-and-index-add-closed`).

---

## Section 7

Hocine validation: TODO
