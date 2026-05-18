# P-HARNESS-UPSCALER-DISPATCH — Verdict

**Chantier:** route the regression harness's `upscaler` family to the
`nbx upscale` subcommand instead of `nbx run --input-image`, so the
20 upscaler cells go red → green.
**Branch:** `p-harness-upscaler-dispatch` from `1973e01` (HEAD of
`p-correctness-silent-failures-v1`).
**Date:** 2026-05-18.

---

## 1. Why P-UPSCALER-CLI-WIRING was cancelled

P-CORRECTNESS-SILENT-FAILURES verdict §5.1 flagged that
`neurobrix run --input-image` never reaches the runtime. The
follow-up P-UPSCALER-CLI-WIRING was opened to "wire 9 flags into
`global.*`". The Étape 1 audit (before writing any code) found this
premise does not hold:

- Family YAMLs declare inputs by **flag string**
  (`required: ["--input-image"]`), not by `global.*` key — they are
  the family↔flag contract for `validate_required_inputs`, not the
  variable-name source.
- The upscaler container's contract is
  `topology.connections: [{"from": "global.pixel_values", "to":
  "model.pixel_values"}]` — a **tensor**, not a path.
- **Nothing in `core/` produces `global.pixel_values`.** No flow
  handler / resolver / preprocessor loads an image. The audio
  precedent (`core/flow/audio.py` reads `global.audio_path` →
  `core/module/audio/input_processor.py::AudioInputProcessor` →
  `global.input_features`) has **zero image analogue**: no image
  handler in `components/handlers/`, no `core/module/image/`, no
  `Image.open` anywhere in `core/`+`triton/` execution.

Conclusion escalated: wiring the CLI flags is necessary correct CLI
behaviour but provably delivers 0/20 of the objective; the real fix
is an entire missing engine subsystem. Maintainer decision (received
2026-05-18): the upscaler runtime is **already complete** via the
dedicated `nbx upscale` subcommand (validated R32 cosine=1.0 at
P-NEUROBRIX-UPSCALERS-V1, commit `7bba82b`). The actual bug is the
**harness** dispatching `upscaler` to the wrong subcommand.
P-UPSCALER-CLI-WIRING cancelled; P-IMAGE-INPUT-FLOW shelved until a
VLM/img2img is traced (not before). This chantier = harness-only fix.

---

## 2. Fix (harness only — no runtime/src change)

`nbx upscale` (cli/__init__.py:369-389; cli/commands/upscale.py)
already loads the image, feeds the topology's `global.pixel_values`,
and saves the SR output. Flags: `--model --input --output`
(required), `--mode {compiled,sequential,triton,triton-sequential}`
(default compiled). Read-only audit; not modified.

| Site | Before | After |
|---|---|---|
| `test_all_models.py` `_run_neurobrix` | always `nbx run … _cli_inputs_for(…)` ; `--triton` if triton | **family==upscaler →** `nbx upscale --model --input <fixture> --output <tmp> --mode {compiled\|triton}` ; else unchanged |
| `test_all_models.py` `_cli_inputs_for` | `if family=="upscaler": return ["--input-image", str(IMAGE_REF)]` (dead — `nbx run` never consumed it) | branch removed; docstring notes upscaler uses `nbx upscale` |
| `test_all_models.py` `test_model_runs` | non-llm = exit-0 only | **upscaler:** exit 0 **AND** PNG written at exactly input×scale (R29 — inspect artefact; scale parsed from model name; 64×64 fixture is window-safe → no processor padding) |
| `test_all_models.py` `IMAGE_REF` | `assets/logo_NeuroBrix.png` (1983×1536) | `test_upscale_input.png` (64×64 RGB, 138 B). Logo OOMs at x4/x8 (52 GiB conv alloc) — a sizing artefact, not the dispatch under test |
| `conftest.py` `FAMILY_TIMEOUT_S` | no `upscaler` entry → 300 default | explicit `"upscaler": 240` |

Helpers added (dependency-free): `_upscale_out_path`,
`_upscale_scale` (regex `x(\d+)`), `_png_size` (PNG IHDR reader — the
pytest interpreter may lack PIL).

Mode mapping: harness `MODES = ["native","triton"]` →
`nbx upscale --mode`: native↔compiled, triton↔triton. 10 cached
upscaler models × 2 modes = 20 cells. The family is not in the pytest
id (`model::mode`), so selection is by model-name substring. The
**exact-20 selector** (verified via `--collect-only`: 20/62) is:

`-k "real-esrgan or swin2SR or swinir or hat-l-x4 or hat-s-x4"`

A naive `… or hat` over-matches `C**hat**`/`c**hat**terbox`/`**chat**`
(TinyLlama-1.1B-**Chat**, **chat**terbox, deepseek-moe-16b-**chat**);
even `hat-` matches `Chat-v1.0`. Only the full `hat-l-x4`/`hat-s-x4`
tokens are unambiguous. The initial validation run used the naive
selector; the per-cell account below splits the 26.

Probe (worst case that OOM'd on the logo): `nbx upscale --model
real-esrgan-x8 --input test_upscale_input.png --mode compiled` →
`SAVED`, 512×512 (= 64×8) in 1.17 s, RC 0.

---

## 3. Anti-regression matrix

20 upscaler cells (10 models × {native, triton}):
real-esrgan-x2/x4/x8, swin2SR-classical-sr-x2-64/x4-64,
swin2SR-realworld-sr-x4-64-bsrgan-psnr, swinir-classical-x2/x4,
hat-s-x4, hat-l-x4.

Validation run (naive selector, 26 cells = 20 upscaler + 6
over-matched; 564 s): **2 failed, 23 passed, 1 xfailed**. Exact
per-cell account:

| Cells | Pre-fix (P-CORRECTNESS) | Post-fix | Status |
|---|---|---|---|
| real-esrgan-x2/x4/x8 ::native+::triton (6) | red (`nbx run --input-image` → "no image tensor output") | **PASS** | red→green ✓ |
| swin2SR-classical-x2/x4 + realworld ::native+::triton (6) | red | **PASS** | red→green ✓ |
| swinir-classical-x2/x4 ::native+::triton (4) | red | **PASS** | red→green ✓ |
| hat-s-x4 / hat-l-x4 ::native (2) | red | **PASS** | red→green ✓ |
| hat-s-x4 / hat-l-x4 ::triton (2) | red | red (exit 1) | **pre-existing, not a regression** |
| over-match: TinyLlama, deepseek-moe ::native+::triton; chatterbox::native (5) | (n/a) | PASS | unchanged `nbx run` path |
| over-match: chatterbox::triton (1) | (n/a) | xfail | KNOWN_FAILURES (tts_llm ref voice) |

**18 / 20 upscaler cells red → green.** The 2 remaining red
(`hat-s-x4::triton`, `hat-l-x4::triton`, exit 1) are **pre-existing
and NOT caused by this chantier**: P-NEUROBRIX-UPSCALERS-V1 (commit
`7bba82b`; `docs/verdicts/p_neurobrix_upscalers/V1_CLOSURE_verdict.md`
+ `v1_backlog.md`) explicitly closed HAT as **2/4 modes** — HAT
triton/triton-seq is blocked by the tracked follow-up
`docs/follow-ups/p-triton-im2col-kernel.md` (HAT OCAB `unfold` needs
an im2col Triton kernel). The harness fix correctly routes
`hat-*-x4::triton` to `nbx upscale --mode triton`, which then fails
*inside the unmodified, validated `nbx upscale` triton path* for the
documented im2col reason — exactly the "deuxième blocker côté runtime
exécution (pas câblage)" the mandate's Étape 3 anticipated: noted as
a latent observation (§5), not treated (scope discipline). Before
this fix the cell failed earlier at the wrong `nbx run` flag; the
fix lets it reach its real (separately-tracked) blocker.

Non-upscaler regression: 0. The change is strictly
`if family == "upscaler"`-gated in `_run_neurobrix`; every other
family takes the byte-identical `nbx run` branch. The over-matched
TinyLlama / deepseek-moe / chatterbox cells (unchanged path) pass /
xfail exactly as in the P-CORRECTNESS baseline. Full-fast confirmation:

Full-fast post-fix (`pytest tests/regression/`, 30m11s):
**4 failed, 34 passed, 12 skipped, 11 xfailed, 1 xpassed** — vs the
P-CORRECTNESS post-fix-fast baseline (23F / 15P / 12skip / 11xf /
1xp): failed 23 → 4 (−19), passed 15 → 34 (+19).

The 4 failed are all pre-existing, none caused by this chantier:

| Cell | Cause | Source |
|---|---|---|
| hat-s-x4::triton, hat-l-x4::triton | HAT triton blocked (OCAB unfold needs im2col kernel) — HAT was closed 2/4 modes | `p-triton-im2col-kernel.md`, `V1_CLOSURE_verdict.md` |
| orpheus-3b-0.1-ft::native, ::triton | `_scaled_dot_product_efficient_attention` 0-dim shape | P-CORRECTNESS verdict §5.2 (pre-existing) |

`Qwen3-30B-A3B-Thinking-2507::triton` (red in the P-CORRECTNESS
post-fix-fast run) is **green** here — re-confirming the documented
non-deterministic MoE-triton flakiness (P-CORRECTNESS §5.6; its DAG
has no index_put/linspace and the harness llm-invocation is
byte-identical, so it is untouched by every chantier). The +19
passed = 18 upscaler red→green + that flaky cell flipping back.

**No previously-green non-upscaler cell regressed.** The change is
strictly `family=="upscaler"`-gated in `_run_neurobrix`; every other
family takes the byte-identical `nbx run` branch (15-green TRUE
baseline preserved; 12 skip / 11 xfail / 1 xpass unchanged).

---

## 4. Commits

Branch `p-harness-upscaler-dispatch`:

| SHA | Subject |
|---|---|
| `121174a` | fix(regression-harness): route upscaler family to `nbx upscale` subcommand |
| _(verdict)_ | docs(verdict): P-HARNESS-UPSCALER-DISPATCH |

---

## 5. Latent observations

- **P-IMAGE-INPUT-FLOW (deferred, not this chantier).** `nbx run`
  still has no image/video/mask/reference input wiring; the
  image-input subsystem (flow handler + `ImageInputProcessor` +
  preprocessing taxonomy mirroring the audio side) is to be opened
  with design review when the first VLM/img2img is traced.
- `-k upscaler` selects nothing (family absent from the pytest id
  `model::mode`). A family-tagged pytest marker would make
  family-scoped selection ergonomic — harness-hygiene candidate,
  out of scope here.

Hocine validation: TODO
