# P-DTYPE-ENGINE-DOCTRINE-RECONCILE — verdict (2026-05-19)

Branch `p-dtype-engine-doctrine-reconcile` from `9e00406` (Ch6
HEAD).

## Section 1 — Goal & debt status

The audit-dettes document of 2026-05-16 (D7 entry) flagged an
apparent contradiction between two early-Feb 2026 debugging
captures (`docs/lessons/002-dtype-engine-prism-master.md`,
`docs/lessons/004-fp16-matmul-overflow-inf-fix.md`) and the live
state of `src/neurobrix/core/dtype/engine.py`. Read as eternal
rules, the lessons forbid AMP wrapping (`AMP_FP32_OPS`,
`_make_fp32_wrapper`, "never clamp") and prescribe a
`_make_inf_fix_wrapper` for matmul overflow; the live code does
the opposite (full AMP, two `_to_copy` clamps, no
`_make_inf_fix_wrapper`). Ch7 is a doctrinal-harmonization
chantier — not a mechanical fix — choosing among (R1) lessons
obsolete / code correct, (R2) code drifted / lessons correct, or
(R3) nuanced middle-ground, based on the factual audit.

Debt status post-audit: confirmed **conditionally LIVE in
documentation, NOT in code** — the contradiction was between
*untracked local development drafts* (`docs/lessons/`,
`docs/internal/`, `docs/roadmap/` are all gitignored — see Section
7) and tracked production source. The live code has been correct
since 2026-02-08; the lessons were never canonized as tracked
project doctrine.

## Section 2 — Audit factuel (Étape 1)

**Lesson 002 (2026-02-02)** — paraphrase of the rule of gold:
"Prism dictates `compute_dtype`. The graph dictates the dtype flow
via emitted `aten::_to_copy` ops. The DtypeEngine only translates
`_to_copy` from half-precision targets to Prism's compute dtype.
Nothing more." Explicit prohibition (Section "Architecture
finale"): no `FP32_OPS / COMPUTE_OPS` classification, no AMP
wrapping, no `enabled` flag, no `get_op_dtype()`, no
`cast_inputs()`. Rationale at the time: "adding AMP casts on top
creates double-cast conflicts because the graph already captures
the internal upcasts/downcasts via `_to_copy`."

**Lesson 004 (2026-02-03, next day)** — Final-state table:
`_make_inf_fix_wrapper` ✅ Active, `_make_fp32_wrapper` ❌
Disabled, with the rule "never use `clamp()` for fp16 overflow
protection". Rationale: clamp on a matmul output corrupts valid
in-range intermediates (the marble-effect image symptom).
**Critically, lines 137-143 of the same lesson** flag AMP as
future plan: "An attempt to implement AMP (FP32_OPS wrapping) the
same session produced catastrophic results (diagonal stripes).
Cause: the implementation performed an immediate fp32→fp16
downcast after every op, whereas PyTorch autocast leaves the
output in fp32 and lets the next op handle the cast. Lesson: AMP
requires an implementation upstream. See
`docs/roadmap/amp-mixed-precision.md` for the future plan."

**Live `engine.py`** — module docstring explicitly cites PyTorch
`aten/src/ATen/autocast_mode.h` as source of truth. Five op
classes: `_to_copy` (Prism-driven dtype remap), `AMP_FP32_OPS`
(numerically sensitive ops upcast to fp32, output STAYS fp32 —
matching PyTorch autocast), `AMP_FP16_OPS` (compute-heavy ops cast
to compute dtype), `_FP16_NEED_FP32 = {mm, bmm, div, addmm}`
(V100-specific fp32 upcast subset), `AMP_PROMOTE_OPS` (promote to
widest input). Two `_to_copy` clamps at lines 420 and 446 narrow
fp32→fp16 when source exceeds ±65 504, with code comment citing
OpenAudio DualAR pre-projection activations + attention bias paths
as the empirical motivation.

**Git archaeology** (`git log --follow` on engine.py and
`git log --all -S`):
- `efe605b` (2026-02-25, v0.1.0-alpha): engine.py already at
  AMP-wired state.
- `0f71a1d` + `4cfa047` (same minute 2026-03-05): "DtypeEngine
  standard AMP + hardware auto-detection" + "**DtypeEngine AMP
  wiring + … + fix grid artifacts**".
- `75b482c` (2026-04-10), `e367981` (2026-04-19): subsequent
  AMP/dtype refinements.
- **`git log --all -S "_make_inf_fix_wrapper"` returns zero commits
  across the entire NBX repository history (any file).** The
  function the lesson marked ✅ Active was never committed under
  that name.

**AMP roadmap doc** (`docs/roadmap/amp-mixed-precision.md`,
Updated 2026-02-08) — six days after the lessons — declares
**status "✅ DONE - DtypeEngine implémenté et fonctionnel"** and
quotes the *current* `_make_fp32_wrapper` body verbatim as
"Solution Implémentée", explicitly fixing the lesson-004
failed-AMP attempt: "**les ops FP32 retournent fp32 sans downcast.
Les ops FP16 suivantes (matmul) castent automatiquement.**"
Outcome: "Stabilité numérique restaurée, zero-corruption
éliminée."

## Section 3 — State of the art consulted (R16)

The live engine cites the authoritative source directly: PyTorch
`aten/src/ATen/autocast_mode.h` (`AT_FORALL_FP32`,
`AT_FORALL_FP32_SET_OPT_DTYPE`, `AT_FORALL_LOWER_PRECISION_FP`,
`AT_FORALL_PROMOTE`). Self-audited 100% match on the FP32 and
PROMOTE sets, 97% on LOWER_PRECISION_FP with documented deviations
(SDPA excluded because `compiled_ops._make_attention` calls
`F.sdpa` directly) and extensions (`polar`, `view_as_complex` →
fp32 because complex32 doesn't exist on CUDA).

Adjacent industry references for context:

- **PyTorch `torch.autocast`** (source of truth above): wraps ops
  by class; FP32 outputs stay fp32; downstream FP16 ops cast back.
  Exactly the pattern lesson 004's failed attempt got wrong
  (immediate downcast) and the AMP roadmap fixed.
- **NVIDIA Transformer Engine**: fp8 per-tensor scaling;
  orthogonal to the V100 fp16 question Ch7 addresses, but
  confirms the principle that mixed-precision frameworks classify
  ops, not models.
- **DeepSpeed mixed-precision**: training loss-scaling; not
  directly applicable to inference, but confirms the
  matmul-overflow-on-fp16 phenomenon as a known industry pattern.
- **JAX bfloat16**: bf16 throughout with no protection. Matches
  the live engine's bf16-hardware branch (zero output protection
  for bf16 hardware because bf16 exponent range = fp32).

R16 considered. The live doctrine matches the industry-canonical
mixed-precision pattern; no external algorithm needs adoption. The
narrow V100-specific refinement (`_FP16_NEED_FP32` upcast subset)
addresses a hardware-class numeric problem PyTorch autocast does
not explicitly handle because most production PyTorch deployments
are A100/H100 with bf16.

## Section 4 — Resolution: **R1** (technical argument)

**Resolution: R1 — supersede both lessons; the live code is the
correct evolved doctrine.**

Three independent decisive factors:

1. **`_make_inf_fix_wrapper` was never committed** to the NBX
   repository under that name (proven by
   `git log --all -S "_make_inf_fix_wrapper"` returning zero).
   There is no code to "re-align onto" — R2 would require
   reverse-engineering a function from a lesson body for a design
   the project documented as superseded six days later.
2. **Lesson 004 itself points forward to the roadmap** (lines
   137-143). The "❌ Disabled — was breaking images" entry in its
   Final-state table captures the state of 2026-02-03 immediately
   *after* the failed AMP attempt, with the lesson explicitly
   announcing the upstream AMP roadmap as the resolution. The
   "Final state" is local to that day, not eternal.
3. **The AMP roadmap status is ✅ DONE since 2026-02-08**,
   quoting the current `_make_fp32_wrapper` verbatim as the
   correctly-designed implementation. The grid-artifacts commit
   (`4cfa047`, 2026-03-05) explicitly bundles "fix grid artifacts"
   — the empirical resolution of lesson-004's "marble/diagonal
   stripes" failure mode — into the AMP wiring commit. The fix
   was correct AMP, not no AMP.

**Lesson-004 "never clamp" sub-nuance (Section 4 of the audit
doc, folded into the supersede notes)**: the prohibition targeted
**matmul-output clamp** which corrupts valid in-range
intermediates. The live engine does **not** clamp matmul outputs
— matmul ops are wrapped by `_make_fp32_wrapper` on V100 fp16
(output stays fp32 — no downcast pressure) or
`_make_lower_precision_wrapper` elsewhere (input-cast only). The
live clamp sits at a fundamentally different site, the
`aten::_to_copy` boundary cast emitted by the graph itself,
narrowing fp32/fp64/bf16 → fp16 when the source exceeds the fp16
range. There the value is required to become fp16 regardless and
fp16 cannot represent anything beyond ±65 504; the choice is
finite-65 504 (preserves graph integrity) vs ±Inf-then-NaN-cascade.
The lesson-004 prohibition does not apply.

R2 is rejected because (1) above. R3 (nuanced middle-ground) is
subsumed into R1's supersede notes — the lesson-004 reasoning
about matmul-output clamp remains correct in its original scope;
the live code respects that scope by clamping elsewhere.

**R23 compliance**: live-prod-is-king. Reverting the engine to
the lesson-era state would empirically re-expose the catastrophic
failure modes the lessons themselves described (NaN, marble
effect, diagonal stripes, black images). The forward validation
in Section 5 empirically confirms that the live doctrine produces
correct outputs on the dtype-sensitive matrix the lessons were
originally written about.

## Section 5 — Implementation + experimental validation

**3 commits on `p-dtype-engine-doctrine-reconcile`** (from `9e00406`):

| Commit | SHA | Content |
|---|---|---|
| 1 | `17e3fae` | `docs/audits/dtype_engine_doctrine_audit.md` — full audit + R1 argument (144 lines). Text-only, no code modification. |
| 2 | `92693ed` | `src/neurobrix/core/dtype/engine.py` — env-gated `NBX_DTYPE_CLAMP_DIAG` diagnostic added at both `_to_copy` clamp sites. Default-off, zero runtime cost. CHANGELOG entry under Added. |
| 3 | `2630193` | `tests/unit/dtype/test_engine_doctrine.py` — 11 doctrine pins (op-set sentinels, exact `_FP16_NEED_FP32`, disjointness, `_to_copy` clamp behaviour at target + passthrough branches, bf16 no-clamp, diagnostic semantics). `.gitignore` extended to track `tests/unit/dtype/` (same pattern as `tests/unit/kernels/`). |

`docs/lessons/{002,004}` were also rewritten on local disk as a
courtesy SUPERSEDED marker pointing to the audit doc and the AMP
roadmap, but `docs/lessons/` is gitignored (`.gitignore:49`) — the
rewrites land locally for any developer reading those files but
produce zero commit. The authoritative tracked record is the
audit doc + `engine.py` module docstring +
`src/neurobrix/CLAUDE.md`.

**Forward experimental validation (R29)** — the dtype-sensitive
matrix the Ch7 mandate names, run under
`NBX_DTYPE_CLAMP_DIAG=1` to empirically witness the clamp:

| Model | Family | Result | clamp hits | Duration |
|---|---|---|---|---|
| openaudio-s1-mini | audio (TTS, dual_ar) | **PASS** (audible WAV) | 0 | 157.0 s |
| Sana_1600M_1024px_MultiLing | image (diffusion fp16) | **PASS** (coherent PNG) | 0 | 15.5 s |
| PixArt-Sigma-XL-2-1024-MS | image (diffusion fp16) | **PASS** (coherent PNG) | 0 | 33.6 s |
| Kokoro-82M | audio (TTS, 82M) | **PASS** (clean WAV) | 0 | 12.9 s |
| Janus-Pro-7B | multimodal (autoregressive image) | **PASS** (coherent PNG, semantic match) | 0 | 192.9 s |

**5/5 PASS** across diffusion, TTS, and multimodal image gen. The
catastrophic failure modes lessons 002+004 cited — NaN, marble
effect, diagonal stripes, black images — **do not reproduce**
under the live doctrine on the model matrix they were originally
written about. The clamp diagnostic fired **0 times across all 5
runs**, including OpenAudio (the explicit motivating model in the
clamp comment) — the `_to_copy` clamp is empirically a tail-path
defensive guard, not a hot-path operation. R29 artefacts under
`validation_outputs/p_dtype_engine_doctrine_reconcile/<model>/`
(output.{wav,png}, prompt.txt, run.log, stats.json, verdict.md,
INDEX.md).

**Counterfactual scope-out** — the lesson-era doctrine
reconstruction (remove the entire AMP machinery, re-add a
never-shipped `_make_inf_fix_wrapper`, remove the `_to_copy`
clamp) is deliberately not run. (1) The function has zero git
presence; (2) R23 forbids destabilizing live prod for a
superseded hypothesis; (3) the forward validation above IS the
discriminator R1 demands. Surfaced explicitly so Hocine can
override.

## Section 6 — Anti-regression

Ch7's only public-runtime change is the default-off env-gated
diagnostic in `engine.py`. The op-set membership, the five
wrappers (`_make_fp32_wrapper`, `_make_lower_precision_wrapper`,
`_make_promote_wrapper`, `_make_safe_softmax`, `_make_to_copy`),
the `_FP16_NEED_FP32` subset, the `_to_copy` clamp behaviour, and
the `convert_constant` policy are **byte-identical** to pre-Ch7.
The default code path is therefore unchanged → cached `.nbx`
models exercise exactly the same execution today as yesterday.

The rigorous anti-reg proof has three layers:

1. **Structural (by construction)**: the diagnostic adds an
   `if not _CLAMP_DIAG_ENABLED or _CLAMP_DIAG_FIRED.get(site):
   return` early-out — with the default `_CLAMP_DIAG_ENABLED =
   False`, the function returns immediately on every invocation;
   no semantic effect on the clamp call site at all.
2. **Forward experimental (Section 5)**: 5/5 PASS on the
   dtype-sensitive matrix, with visual / audible inspection — the
   doctrine being canonized produces correct outputs today on the
   models the lessons were originally about.
3. **Structural doctrine pins**:
   `tests/unit/dtype/test_engine_doctrine.py`, 11/11 PASS,
   covering op-set sentinel membership, exact `_FP16_NEED_FP32`,
   FP32/FP16 disjointness, the `_to_copy` clamp at both target
   and passthrough branches (finite ±65 504, not Inf), bf16
   no-clamp, and the three diagnostic semantics. Any future
   silent doctrine drift trips a specific named test.

**Full `pytest tests/regression/` and targeted `--runslow`
scope-out (with rationale)** — explicit per Ch6 advisor pattern,
surfaced for Hocine override:
the full harness would exercise runtime code Ch7 provably does
not modify; the targeted `--runslow` subset overlaps the R29
forward validation already done (Sana 1024 + PixArt + Kokoro)
and adds Janus-Pro-7B which is also in R29 above. Running it would
produce no Ch7-specific signal beyond what Sections 5 + the
doctrine pins already establish, and would also pull in slow
diffusion / video cells that include the standing-INDETERMINATE
4Kpx model (memory: do **not** relaunch under longer budgets).
The structural + R29 + doctrine-pins triad is the
proportionate-and-rigorous anti-reg control for a default-off
diagnostic chantier.

## Section 7 — Latent observations (D10 — NOT fixed in Ch7)

- **`docs/lessons/`, `docs/internal/`, `docs/roadmap/` are all
  gitignored** (`.gitignore:48-50`). The audit-dettes D7 entry
  read a contradiction between *untracked local development drafts*
  and tracked production code — these are not the same kind of
  object. The lessons were always working notes pointing to a
  roadmap doc that was also untracked. The implication: the
  audit-dettes hygiene chantier (Ch10
  P-VERDICTS-HYGIENE) should distinguish *tracked-doctrine
  contradictions* (real debt) from *untracked-draft
  contradictions* (developer-notes hygiene), and consider grepping
  for `Status:.*DONE` and `SUPERSEDED` markers in untracked-doc
  trees to catch similar cases pre-emptively.
- **The lesson-prescribed `_make_inf_fix_wrapper` design is
  historical artefact**, not a missing feature — the AMP roadmap
  superseded it by design (output stays fp32, downstream FP16
  ops cast back). No follow-up needed.
- **OpenAudio DualAR motivating site empirically dormant** under
  standard prompts. The `_to_copy` clamp is documented defensive
  infrastructure; if a future chantier wants production-level
  empirical confirmation of the cited motivating activations, a
  default-off counter (rather than the current one-shot diag)
  would let real workloads accumulate hit statistics over time —
  not in Ch7 scope.
- **Pre-existing unused-import/var Pyright warnings** in
  `engine.py` (`_kwargs` line 437, `op_type` line 558,
  unused-local on a couple of inner closures) shifted by +22
  lines from the diagnostic header insert. Pre-existing, not
  introduced by Ch7, left untouched (strict scope; flag for a
  future code-hygiene pass).

## Section 8

Hocine validation: TODO. R29 artefacts under
`validation_outputs/p_dtype_engine_doctrine_reconcile/` (5
human-inspectable outputs: 3 PNGs, 2 WAVs).
