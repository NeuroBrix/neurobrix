# P-CORRECTNESS-SILENT-FAILURES — Verdict

**Chantier:** eliminate three families of silent-failure bugs from the
residual-debt audit (`docs/audit_dettes_residuelles_2026_05_16.md`,
SHA `3fb4430`, §1.1 / §1.2 / §1.4 / §1.5 / §1.6).
**Branch base:** `main` @ `3fb4430`.
**Date:** 2026-05-16.

> Note on D3.4: it lives in the **private packaging toolchain**,
> which is a **separate git repository** kept strictly decoupled
> from the runtime (runtime↔packaging zero-coupling rule). This
> public verdict describes its user-visible effect only; the
> internal path / symbol names stay in the private companion.

---

## 0. Summary

| Bug | Status | Fix location | Commit |
|---|---|---|---|
| D1 — `linspace` Triton uninitialised memory | FIXED | `src/neurobrix/kernels/dispatch.py:336-368` | `900823a` |
| D2 — `index_put`/`index_put_` identity lambdas | FIXED (reachable → real kernel) | `src/neurobrix/kernels/ops/index_put_op.py` + `wrappers.py` + `dispatch.py:700-701` | `039e60e` |
| D3.1 — bare-except hides corrupt VAE profile | FIXED | `src/neurobrix/core/module/output_processor.py:70-95` | `24a57c7` |
| D3.2 — unknown synthesis method skipped | FIXED | `src/neurobrix/core/runtime/resolution/input_synthesizer.py:247-258` | `da75c84` |
| D3.3 — bare-except in prefetch enqueue | FIXED | `src/neurobrix/core/io/memory.py:12-21,257-272` | `1abe748` |
| D3.4 — bare-except drops container provenance | FIXED | private packaging repo (separate git repo) | `8b9b5c3` (packaging repo) |
| Harness desync (in-scope correction) | FIXED | `tests/regression/{conftest.py,test_all_models.py}` | `59d483d` |

All commits except D3.4 are in the NeuroBrix (runtime) repo. D3.4 is
in the private packaging repository by structural necessity (the
runtime↔packaging zero-coupling rule).

---

## 1. Per-bug before/after matrix

### D1 — `linspace` Triton uninitialised memory (audit §1.1)

- **Before:** `_create_linspace` did
  `out = NBXTensor.empty((steps,), ...)`, imported `fill_kernel`,
  then `# TODO` / `return out` — **no kernel launched**. Reachable:
  `linspace` ∈ `_METADATA_OPS` (`kernels/classification.py:122`
  `"arange", "linspace",`) → `dispatch()` (`dispatch.py:742`) →
  `_create_linspace`. Every `aten::linspace` in a `--triton` /
  `--triton-sequential` DAG (diffusion / video timestep schedules,
  positional grids) received **random GPU memory** — no crash, no
  NaN guarantee, silently wrong.
- **After:** `dispatch.py:336-368` wires the **existing** kernel at
  `kernels/ops/linspace_op.py:18` (`linspace_kernel`, bidirectional:
  forward from start for the first half, backward from end for the
  second → exact endpoints); `steps == 1` special-cased through
  `fill_kernel(start)` (torch returns `[start]`; the bidirectional
  formula would yield `[end]` — this is what the previously-dead
  `fill_kernel` import was for); static `BLOCK_SIZE=128`, no
  autotune (R5/R20), mirroring `_create_arange`.
- **Implementation choice (justification):** the audit said "write a
  dedicated `@triton.jit` kernel". On reading the code, a dedicated
  bidirectional `@triton.jit` kernel **already existed** at
  `kernels/ops/linspace_op.py` (Apache-2.0 lineage), wired nowhere.
  Per the No-Orphan-Code / Search-Before-Write doctrine the correct
  action was to **wire the existing kernel**, not duplicate it. It
  is the dedicated `@triton.jit` the audit asked for; the bug was
  purely a missing dispatch wire. Superior fix (no duplicate kernel,
  exact endpoints by construction).
- **Test:** `tests/unit/kernels/test_linspace.py` — 3 dtypes × 4
  sizes × 2 ranges vs `torch.linspace`. fp32 **bit-exact**
  (`max|d|=0.0` incl. steps 1/2/50/10000), fp16/bf16 within 1 ULP,
  endpoints exact. Validated on **production torch 2.5.1** (script
  mode: ALL PASS) **and torch 2.10** (pytest: 24 passed).
  Empirically proven to FAIL on the pre-fix code (uninitialised
  buffer ≠ `torch.linspace`, `max|d|=1.0`).

### D2 — `index_put` / `index_put_` identity lambdas (audit §1.2)

- **Reachability decision — DECIDED REACHABLE, on facts** (audit
  left it "to confirm"; auto-mode mandate → decided on sources read
  this run):
  - `index_put` ∉ `_METADATA_OPS` (classification.py) and
    `get_execution_type` raises `KeyError` for it, but the triton
    execution paths resolve every op through the **general dispatch
    table regardless**:
    - `triton/sequential.py:170` — `func = kernel_dispatch(base)`,
      `base="index_put"`, no pre-filter excludes it; the
      `if func is None: raise` guard at `:171` **does not fire**
      (identity lambda is not `None`).
    - `triton/sequence.py:1511` — `func = dispatch(op_type)`, same
      non-None-so-no-raise path at `:1512`.
  - `aten::index_put` ∈ `moe_fusion.py:426` (`MOE_OP_TYPES`, V2
    aggregation) and `compiled_sequence.py:52` (`_ACCUMULATOR_OPS`,
    NOP propagation) — the MoE-v2 expert-output aggregation path
    (audio_llm / multimodal / LLM-MoE coverage).
  - **Conclusion:** reachable in `--triton` AND
    `--triton-sequential`; the identity lambda returned the input
    **unmodified** → every scatter write silently dropped → real
    kernel required (not fail-fast-only).
- **Before:** `dispatch.py:700-701` — two identity lambdas.
- **After:** new pure-`@triton.jit` `kernels/ops/index_put_op.py`
  (`index_put_kernel`, `ACCUMULATE` constexpr → `tl.store` vs
  `tl.atomic_add`) + `wrappers.py:index_put_wrapper` (mirrors
  `index_add_wrapper`; functional return, `_` variant realised by
  the graph slot exactly like `index_add`). `dispatch.py:700-701`
  → `w.index_put_wrapper` for both names.
- **Scope (documented boundary):** exactly one non-None **integer**
  index tensor on the **leading dim**, any index shape,
  with/without accumulate — the decomposed-ATen norm (MoE-v2
  aggregation, KV-cache indexed writes, post-`aten::nonzero` masked
  scatter). k≥2 advanced indices, non-leading-dim, **boolean masks**
  (no `nonzero` kernel in the catalogue), un-broadcastable `values`
  raise `NotImplementedError` naming follow-up
  **P-INDEX-PUT-ADVANCED-GENERAL** — visible crash beats silent
  identity (ZERO-FALLBACK).
- **Test:** `tests/unit/kernels/test_index_put.py` via the real
  `dispatch("aten::index_put")` path — 4 configs × fp16+fp32 + 2
  fail-fast cases. Production torch 2.5.1 AND torch 2.10:
  non-accumulate **bit-exact** fp32+fp16, accumulate
  `max|d|≤1.5e-8`, k≥2/bool raise. Proven to FAIL on the pre-fix
  identity lambda.

### D3.1 — bare-except hides corrupt VAE profile (audit §1.4)

- **Before:** `OutputProcessor.from_package` wrapped the VAE profile
  load + registry lookup in `except Exception: pass`. An existing
  but corrupt `components/vae/profile.json` was swallowed → silent
  fallback `clamp_before_normalize=False` → out-of-range VAE image,
  zero diagnostic.
- **After:** `output_processor.py:70-95` — the `.exists()` guard
  keeps an ABSENT profile a legitimate defaults no-op; an existing
  but unreadable profile is narrowed to `(OSError, ValueError)`
  (`json.JSONDecodeError ⊂ ValueError`) and raised as a descriptive
  `RuntimeError`.
- **Reliance check (advisor-flagged):** `grep clamp_before_normalize
  src/ tests/` — all references internal to `output_processor.py`;
  nothing relied on the silent skip.

### D3.2 — unknown synthesis method skipped (audit §1.5)

- **Before:** `input_synthesizer.py` synthesis dispatch ended
  `else: pass  # Unknown synthesis method - skip silently`. An
  unregistered method produced no tensor → downstream component
  consumed an unset slot → silently wrong inference.
- **After:** `input_synthesizer.py:247-258` — `else:` raises the
  file's own established ZERO-FALLBACK `RuntimeError` idiom (mirrors
  the `from_dimensions` raise in the same method), naming the
  method, input, component and the known methods.
- **Reliance check:** `grep synthesis/input_synthesizer
  src/neurobrix/triton/ tests/` — only constructor plumbing and
  scratch probes; nothing relied on the silent skip.

### D3.3 — bare-except in prefetch enqueue (audit §1.6)

- **Before:** `io/memory.py` prefetch enqueue used a bare
  `except: pass` — caught the expected queue-full BUT also
  `KeyboardInterrupt` / `SystemExit` and any genuine error, then
  silently served the component uncached on the warm path.
- **After:** `io/memory.py:257-272` — narrowed to `except Full:`
  with an explicit `logger.warning` (warm-path stale-serve now
  visible, per mandate). Added the stdlib
  `logging.getLogger(__name__)` (module had no logger; convention
  used elsewhere) and `Full` to the queue import. Any other
  exception and Ctrl-C now propagate.
- **Observation (not changed):** queue-full is normal backpressure,
  so `warning` may be noisy under load and could be tuned to
  `debug` later — kept at `warning` per the mandate's explicit
  "propager au minimum en warning".

### D3.4 — bare-except drops container provenance (audit §1.6)

- **Before:** in the private packaging toolchain, the routine that
  extracts model-provenance metadata (vendor / source URL) from the
  source `README.md` wrapped its regex in `except: pass` — a bare
  except swallowing `KeyboardInterrupt`/`SystemExit` and any real
  error, dropping `origin.vendor` / `origin.url` from the immutable
  container with zero signal (irrecoverable at runtime).
- **After:** narrowed to `(OSError, UnicodeDecodeError)` with a
  clear `[WARN]` line. Provenance is best-effort (a secondary
  model-directory-name heuristic still runs) so it logs rather than
  raises; `KeyboardInterrupt`/`SystemExit` and unexpected errors now
  propagate. **User-visible effect:** the packaged container's
  provenance metadata is now either correctly populated or its
  absence is loudly reported at build time, instead of silently
  lost.

---

## 2. Commits (exhaustive)

NeuroBrix runtime repo (two remotes):

| # | SHA | Subject |
|---|---|---|
| 1 | `59d483d` | fix(regression-harness): family-aware CLI inputs (upscaler/multimodal/audio_llm desync) |
| 2 | `900823a` | fix(dispatch): wire linspace Triton kernel — silent uninitialised memory (D1) |
| 3 | `039e60e` | fix(dispatch): index_put — wire real scatter kernel, fail-fast on unwired forms (D2) |
| 4 | `24a57c7` | fix(output-processor): bare-except hid corrupt VAE profile → wrong image (D3.1) |
| 5 | `da75c84` | fix(input-synthesizer): unknown synthesis method silently skipped (D3.2) |
| 6 | `1abe748` | fix(io/memory): bare-except in prefetch enqueue swallowed everything (D3.3) |
| V | _(verdict commit — see §6)_ | docs(verdict): P-CORRECTNESS-SILENT-FAILURES |

Private packaging repo (separate git repo, its own two remotes):

| # | SHA | Subject |
|---|---|---|
| 7 | `8b9b5c3` | D3.4 — bare-except dropped container provenance silently |

"Un commit par bug" spans two repositories for D3.4 by structural
necessity (runtime↔packaging zero-coupling). The chantier tag
`p-correctness-silent-failures-v1-closed` is applied to the verdict
commit in the NeuroBrix runtime repo.

---

## 3. Anti-regression harness

### 3.1 Pre-fix baseline (clean `main` @ `3fb4430`)

`pytest tests/regression/` (fast scope, 31 cached models × 2 modes,
image/video excluded): **25 failed, 13 passed, 12 skipped, 11
xfailed, 1 xpassed in 2140.54s (35m40s)**.

Triage of the 25 failures (each reproduced individually with its
exact error):

| Cells | Cause | Class |
|---|---|---|
| 20 upscaler (real-esrgan×3, swin2SR×3, swinir×2, hat×2, ×2 modes) | harness sent `--prompt` (no `upscaler` branch) — AND a deeper pre-existing runtime gap (§5.1) | harness desync + pre-existing runtime gap |
| 2 Janus-Pro-7B (native+triton) | harness sent no `--mode` (multimodal-strict) | harness desync |
| 1 Voxtral-Mini-3B native | harness sent audio-only (`audio_llm` ∈ STT_FLOWS) | harness desync |
| 2 orpheus-3b (native+triton) | `aten::_scaled_dot_product_efficient_attention` shape `[1,8,3,0,1]` (0-dim), input correct | pre-existing `main` bug |

The 25 reds on a clean `main` are the "harness not run recently"
desync the mandate explicitly scoped in. Commit `59d483d` is the
in-scope correction.

### 3.2 Harness-fix validation (targeted re-baseline, post `59d483d`, pre D1-D3)

`pytest -k "Janus-Pro-7B or Voxtral-Mini-3B-2507"`, 542.66s:

| Cell | Pre-fix | Post-harness-fix |
|---|---|---|
| Janus-Pro-7B::native | red (no `--mode`) | **PASS** (red→green) |
| Voxtral-Mini-3B-2507::native | red (no `--prompt`) | **PASS** (red→green) |
| Voxtral-Mini-3B-2507::triton | xfail | xfail (expected, KNOWN_FAILURES:113) |
| Janus-Pro-7B::triton | red (no `--mode`) | red — 300s **timeout**, empty stdout (pre-existing; never green) |

**TRUE pre-fix green set = 15** (original 13 + Janus::native +
Voxtral::native). The D1-D3 fixes must not regress this set.

### 3.3 Post-fix harness (D1-D3 applied)

Fast scope `pytest tests/regression/` (post D1-D3, 32m25s):
**23 failed, 15 passed, 12 skipped, 11 xfailed, 1 xpassed** (pre-fix
was 25 / 13 / 12 / 11 / 1).

Cell-level transitions vs the pre-fix run:

| Cell | Pre-fix | Post-fix | Cause |
|---|---|---|---|
| Janus-Pro-7B::native | red | **PASS** | harness fix `59d483d` |
| Voxtral-Mini-3B-2507::native | red | **PASS** | harness fix `59d483d` |
| Janus-Pro-7B::triton | red (300s timeout, cold triton cache in the isolated re-baseline) | **PASS** | harness fix + warm shared triton autotune cache in the full run (completed < 300s) |
| Qwen3-30B-A3B-Thinking-2507::triton | green | red ("output drift") | **pre-existing triton-MoE non-determinism — NOT a chantier regression (proof below)** |

The 15 passed = the TRUE pre-fix green set (original 13 +
Janus::native + Voxtral::native), with Janus::triton additionally
green. **No cell in the TRUE 15-green set went red.** The 23
failures = 20 upscaler (pre-existing CLI gap §5.1) + 2 orpheus
(pre-existing SDPA §5.2) + 1 Qwen3-30B::triton (pre-existing
flakiness, below).

**Qwen3-30B::triton green→red — proven NOT a chantier regression:**

- `~/.neurobrix/cache/Qwen3-30B-A3B-Thinking-2507/components/*/graph.json`
  contains **0 `index_put`** and **0 `linspace`** (6144
  `aten::index_add` + 48 `aten::scatter` — Qwen3 MoE-v2 aggregation
  uses index_add / scatter, NOT index_put). → D2 and D1 kernels are
  **never invoked** for this model.
- D3.1 (no VAE in an LLM), D3.2 (would `raise`→exit-1, but the cell
  is exit-0 "output drift"), D3.3 (warm-path prefetch; the LLM
  harness uses the cold path `python -m neurobrix run`) are all
  inert here.
- Harness `_cli_inputs_for` `family == "llm"` branch is
  **byte-identical** pre-fix (`3fb4430` test_all_models.py:114-115)
  and now (:140-141); `_run_neurobrix` only gained an unused-for-llm
  `gen_type` param → Qwen3 receives the **exact same command**.
- Same invocation + provably-identical runtime path ⇒ green→red is
  **pre-existing non-determinism in the triton MoE greedy-decode
  path** (corroborated by memory `project_fused_moe_status`,
  `project_triton_qwen3_status`, R27/R28 MoE data-dependence). The
  Apr-14 golden froze one 5-token greedy output; re-running a
  long-unrun harness surfaced the flakiness.
- **Do NOT `UPDATE_GOLDEN`** (the harness UI suggests it): native —
  the torch reference, D2-unaffected — still passes with
  `Okay, the user said` pre AND post-fix; re-capturing the
  non-deterministic triton output would freeze a wrong truth.
  Candidate follow-up: **P-TRITON-MOE-DETERMINISM** (or a
  MoE-triton golden tolerance / `xfail`-flaky decision — Hocine's
  call, not this chantier).

`--runslow` scope (adds image/video, 1h06m):
**28 failed, 22 passed, 0 skipped, 11 xfailed, 1 xpassed**. The 12
previously-skipped image/video cells now run: 7 pass, 5 fail.

The 5 image/video reds (added on top of the 23 fast reds):

| Cell | Reason | Class |
|---|---|---|
| Flex.1-alpha::triton | exit 1 | pre-existing image-`--triton` blocker (audit follow-ups layer7/8/9) |
| PixArt-Sigma-XL-2-1024-MS::triton | timeout 600s | pre-existing PixArt-triton leak (`project_pixart_triton_leak`) |
| PixArt-XL-2-1024-MS::triton | timeout 600s | pre-existing PixArt-triton leak |
| SANA-Video_2B_720p_diffusers::native | exit 1 | pre-existing — video family "essentially unproven" (audit 3fb4430 §A) |
| SANA-Video_2B_720p_diffusers::triton | exit 1 | pre-existing — video family unproven |

**Proven NOT chantier-caused (same discipline as Qwen3):**

- DAG grep: Flex / PixArt-Sigma / PixArt-XL / SANA-Video each have
  **0 `linspace`** and **0 `index_put`** in `graph.json` → D1 and
  D2 kernels never invoked.
- D3.1 ruled out: SANA-Video and Flex
  `components/vae/profile.json` parse as valid JSON → the narrowed
  `except (OSError, ValueError)` never triggers.
- D3.2 ruled out: diffusion synthesis uses only registered methods;
  the fast run exercised the same `input_synthesizer` across 25
  models with zero D3.2-induced crashes.
- D3.3 ruled out: image/video harness uses the cold path → warm
  prefetch bypassed.
- Harness fix ruled out: image/video flow is not
  upscaler/multimodal/STT/audio_llm/TTS-ref → falls to the final
  `return ["--prompt", "Hello world"]`, byte-identical to pre-fix.
- These are exactly the pre-existing image-`--triton` and
  video-family gaps documented in the audit that motivated this
  chantier (`3fb4430` §A, follow-ups `layer7/8/9`,
  `project_pixart_triton_leak`). Sana 1024 (native+triton), Sana
  4Kpx (native+triton), Flex::native, PixArt-Sigma/XL::native are
  among the 22 passed — D1 did not regress diffusion-native and
  D3.1 did not regress image VAE.

**ANTI-REGRESSION CRITERION — SATISFIED (fast + --runslow).** Every
green→red transition (1 fast: Qwen3-30B::triton; 5 --runslow:
image/video) is proven to lie OUTSIDE all seven changed code paths
(DAG greps + profile-parse + cold-path + byte-identical harness
llm/image branches) = pre-existing, each sourced. No cell in the
TRUE pre-fix green set went red because of D1/D2/D3 or the harness
fix. The harness fix additionally flipped 3 false-reds green
(Janus/Voxtral native + Janus::triton). The remaining reds
(20 upscaler §5.1, 2 orpheus §5.2, 1 Qwen3-MoE-triton
non-determinism §5.6, 5 image/video pre-existing) are all
documented and outside this chantier's scope.

---

## 4. Reachability decision — `index_put` (factual sources)

DECIDED **REACHABLE** in `--triton` and `--triton-sequential`.
Sources read this run:

- `kernels/classification.py:110-153` — `index_put` ∉ `_TRITON_OPS`,
  ∉ `_METADATA_OPS`; `get_execution_type` → `KeyError`;
  `is_metadata_op`/`is_triton_op` swallow it → `False`.
- `triton/sequential.py:135-177` — `dispatch()` resolves every
  non-special-cased op via `kernel_dispatch(base)`; identity lambda
  non-None → `if func is None` false → executed → returns input.
- `triton/sequence.py:1499-1516` — same: `func = dispatch(op_type)`;
  identity lambda non-None → no `RuntimeError` → executed.
- `core/runtime/graph/moe_fusion.py:426` — `aten::index_put` ∈
  `MOE_OP_TYPES` (V2 aggregation `cat + scatter`).
- `core/runtime/graph/compiled_sequence.py:50-53` —
  `aten::index_put` ∈ `_ACCUMULATOR_OPS`.

---

## 5. Latent observations (NOT treated — for future chantiers)

1. **[REOPEN-CANDIDATE — HIGH] `neurobrix run --input-image`
   runtime gap.** `src/neurobrix/cli/commands/run.py:288-309` maps
   `--prompt`/`--audio`/`--steps`/etc. into `inputs["global.*"]`
   but has **no branch for `args.input_image`** (defined
   `src/neurobrix/cli/__init__.py:119`, validated `run.py:201`).
   All 20 upscaler cells fail `neurobrix run` with `ZERO FALLBACK:
   no image tensor output. Available: []` on `main` (`3fb4430`),
   independent of harness inputs. Implies P-NEUROBRIX-UPSCALERS-V1
   (closed 2026-05-15) was validated via a non-CLI path; the
   production CLI is incomplete for the upscaler family.
   Reproduction: `python -m neurobrix run --model real-esrgan-x2
   --temperature 0 --input-image assets/logo_NeuroBrix.png` →
   `RC=1`, `CLI inputs: ['global.temperature']`. Out of scope here
   (escalate, not fourre-tout). **Proposed: dedicated chantier
   P-UPSCALER-CLI-WIRING, or reopen P-NEUROBRIX-UPSCALERS-V1.**

2. **orpheus-3b pre-existing runtime bug.** Both modes fail at
   `aten::_scaled_dot_product_efficient_attention` with shape
   `[1,8,3,0,1]` (a 0 dim) `is invalid for input of size 24`. Input
   (`--prompt`) is correct. Pre-existing on `main`, unrelated to
   D1/D2/D3. Pre-existing red.

3. **R30 compiled↔triton MoE NOP-propagation asymmetry (the audit
   missed this).** `triton/sequence.py:142` defines a LOCAL
   `_ACCUMULATOR_OPS = frozenset({"add", "mul"})` while
   `core/runtime/graph/compiled_sequence.py:50-53` defines a broad
   set including `index_put`, `scatter`, `scatter_add`,
   `index_add`, `scatter_reduce`. When a deactivated MoE expert
   yields `args[0] is None`, compiled mode passes-through-base for
   `index_put` etc.; triton-compiled does NOT (only add/mul) → it
   takes the slot-nulling branch. A real R30 divergence in MoE NOP
   propagation, not caught by the residual-debt audit. Candidate:
   **P-R30-MOE-NOP-ACCUMULATOR-PARITY**.

4. **Janus-Pro-7B::triton timeout-sensitive (harness hygiene).**
   In the isolated targeted re-baseline (cold triton autotune
   cache) it hit the 300s timeout; in the full fast run (warm
   shared autotune cache) it completed < 300s and **PASSED**.
   `FAMILY_TIMEOUT_S` (`tests/regression/conftest.py:73`) has no
   `"multimodal"` entry, so multimodal silently gets the 300s
   generic default — too tight for a 7B triton model on a cold
   cache. NOT speculatively bumped here (out of scope; the audit
   condemns disguised-deferral timeout bumps). Candidate: a
   data-driven multimodal entry in `FAMILY_TIMEOUT_S`
   (harness hygiene, small).

5. **Stale doctrine doc.** `src/neurobrix/CLAUDE.md` §5 references
   `kernels/adapter.py` as the `--triton` ATen translator; the file
   does not exist (the dispatch is `kernels/dispatch.py` +
   `triton/sequence.py`/`sequential.py`). Documentary debt.

6. **[REOPEN-CANDIDATE — MED] Qwen3-30B-A3B::triton MoE
   non-determinism.** Green pre-fix, red post-fix ("output drift",
   exit 0). Proven NOT chantier-caused: its DAG has 0 `index_put`
   and 0 `linspace` (6144 `aten::index_add` + 48 `aten::scatter` —
   MoE-v2 aggregation uses index_add/scatter, not index_put); D3.*
   inert (no VAE, exit-0 not a raise, cold-path); the harness
   `family=="llm"` invocation is byte-identical pre/post
   (`3fb4430` test_all_models.py:114-115 ≡ now :140-141). Same
   command + provably-identical runtime path ⇒ pre-existing
   non-determinism in the triton MoE greedy-decode path
   (corroborated by memory `project_fused_moe_status`,
   `project_triton_qwen3_status`, R27/R28 MoE data-dependence). The
   Apr-14 golden froze one 5-token greedy output; re-running a
   long-unrun harness surfaced the flakiness. **Do NOT
   `UPDATE_GOLDEN`** (the harness UI suggests it): native — the
   torch reference, D2-unaffected — still passes with
   `Okay, the user said` pre AND post-fix; re-capturing the
   non-deterministic triton output would freeze a wrong truth.
   Candidate: **P-TRITON-MOE-DETERMINISM** (determinism fix, or a
   MoE-triton golden-tolerance / `xfail`-flaky decision — Hocine's
   call, not this chantier).

---

## 6. Conclusion

D1, D2, D3.1, D3.2, D3.3, D3.4 fixed and committed (one commit per
bug; D3.4 in the private packaging repo per the runtime↔packaging
zero-coupling rule). The in-scope harness desync corrected and
validated (Janus/Voxtral native flipped red→green). D2 reachability
decided REACHABLE on sources and a real kernel wired (not
fail-fast-only). All unit tests pass on production torch 2.5.1 and
torch 2.10; each test proven to fail on the pre-fix code.
Anti-regression: see §3.3.

The highest-value latent finding — the `neurobrix run --input-image`
runtime gap making the upscaler family non-functional via the
production CLI (§5.1) — is surfaced as a sourced REOPEN-CANDIDATE per
the "no technical debt left behind" directive, deliberately NOT
absorbed into this chantier (scope discipline).

Hocine validation: TODO
