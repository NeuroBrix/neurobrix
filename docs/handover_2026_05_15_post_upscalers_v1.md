# Handover — 2026-05-15, post P-NEUROBRIX-UPSCALERS-V1

Factual context for the next Claude session. No assumptions —
everything below is read from the current code and git state. A
private companion report (model-packaging side) covers the
sections not appropriate for this public repo.

---

## SECTION 1 — Exact git state (NeuroBrix)

- Branch: `main`, working tree **clean** (only the gitignored
  `.venv-mac/` and `validation_outputs/` are untracked — both
  intentional).
- Remotes both up to date (pushed this session):
  - `origin` → `git@github.com:NeuroBrix/neurobrix.git`
  - `gitlab` → `gitlab.com/neurobrix/neurobrix`
- Recent commits (newest first):
  - `a92ee00` docs(upscalers): P-NEUROBRIX-UPSCALERS-V1 closure
  - `f9bbc20` docs(upscalers): U7 HAT verdict — 2/4 modes
  - `c0a1445` fix(compiled): reachable constant-slot pre-population
  - `bb52f98` docs(upscalers): U6 SwinIR verdict — 8/8 cells
  - `05a39ee` feat(prism): per-component requires_fp32_compute opt-in
  - `c0cff5d` test(upscalers): real-esrgan 12 cells validated
  - `99c0678` docs: nbx upscale user guide + U3/U4 verdict
  - `9c762bb` feat(runtime): roll + reflection_pad2d triton kernels
  - `3a42e46` docs(upscalers): v1 backlog (BL-1)
  - `7bba82b` feat(cli): nbx upscale subcommand + forward_pass fix
- Tags (recent): `p-neurobrix-upscalers-v1-closed`, `v0.2.0`,
  `p-prism-never-refuse-v2-closed`,
  `p-sana-4kpx-runtime-fully-closed`,
  `p-sana-4kpx-runtime-closed`, `v0.1.6` … Confirmed at both
  remotes via `git ls-remote --tags`.

---

## SECTION 3 — NeuroBrix user commands (`neurobrix <cmd>`)

Source: `src/neurobrix/cli/__init__.py` + `cli/commands/`.

| Command | Role | Key args |
|---|---|---|
| `run` | Inference, all 9 families (family read from manifest) | `--model --hardware --prompt --audio --steps --cfg --height --width --output --seed --temperature --repetition-penalty --set KEY=VALUE --max-tokens --input-image --mask-image --reference-image --reference-audio --speaker --video --num-frames --fps --system --mode --chat/--no-chat`; execution: `--compiled`(default)`/--sequential/--triton/--triton-sequential` |
| `upscale` | Image super-resolution | `--model --input --output` (all required), `--mode {compiled,sequential,triton,triton-sequential}` (default compiled), `--hardware` |
| `serve` | Persistent VRAM-resident daemon (all families) | `--model`(req) `--hardware --timeout`(1800) `--foreground`; same 4 execution flags |
| `chat` | Interactive REPL against a running daemon (LLM) | connects to serve socket |
| `stop` | Stop running daemon | — |
| `info` | System status | `--models --hardware --system` |
| `inspect` | Inspect a model/container | — |
| `import` | **Pull** a `.nbx` from `neurobrix.es` → local cache | `model_ref` (`org/name`), `--registry`, `--force` |
| `hub` | **Browse** models on `neurobrix.es` | `--category {IMAGE,VIDEO,AUDIO,SPEECH,LLM,UPSCALER}`, `--search/-s`, `--registry` |
| `list` | List local models | — |
| `remove` | Remove a local model | — |
| `clean` | Clean cache | — |
| `validate` | Validate `.nbx` integrity | `nbx_files…`, `--level {structure,schema,coherence,deep}`, `--strict --json --verbose` |
| `doctor` | Environment diagnostics | — |

Default execution mode is `--compiled` when no flag is passed
(CompiledSequence, cuDNN/cuBLAS). `--triton` is the
NeuroBrix-pure path.

---

## SECTION 4 (NeuroBrix side) — Hub `neurobrix.es`: LIVE PRODUCTION SERVICE

> CORRECTION (2026-05-15): an earlier draft of this report said
> hub deployment was "not determinable / do not assume live".
> That was wrong — written without consulting the architecture
> memory. **The hub is a mature, deployed, operational service.**
> Verified this session by querying it directly.

**Storage layers — do not conflate them (user-visible scope):**

1. **huggingface.co** — upstream SOURCE ONLY. Raw third-party
   models are obtained FROM here. **Never a destination.**
   Publishing `.nbx` to HF is forbidden. HF is not "our"
   registry — it is just where public weights originate.
2. **NeuroBrix local store** — `~/.neurobrix/store/<file>.nbx`,
   where `nbx import` lands a downloaded `.nbx` before extract.
3. **NeuroBrix local cache** — `~/.neurobrix/cache/<model>/`,
   the extracted container the runtime executes.
4. **NeuroBrix Hub** — `neurobrix.es` (proprietary, ours). The
   production registry where the official `.nbx` are hosted.
   `nbx import` / `nbx hub` pull/browse it.

(The upstream→`.nbx` packaging pipeline and its intermediate
working directories are proprietary and documented only in the
private companion report — not in this public file.)

**Hub infrastructure (live, verified — also in the
`architecture_details` memory):**
- `neurobrix.es` → Next.js app at `10.0.0.39:3000`
  (server hostname `NeuroBrix`, reachable via SSH `mlops@10.0.0.39`).
- PostgreSQL at `10.0.0.35:5432` (Prisma).
- MinIO object storage at `10.0.0.36:9000`.
- `hub.neurobrix.es` → public download path
  (Cloudflare → OPNsense HAProxy → MinIO:9000).
- Dual S3 client: internal `s3` for uploads, `s3Public`
  (hub.neurobrix.es) for downloads.
- Auth: GitHub + Google OAuth, GDPR primitives, Brevo SMTP.
  Hub source repo: `github.com/benkelaya/NeuroBrix_Hub`
  (Phase 1 = auth/GDPR/email 2026-04-30; Phase 2 = 9-family
  taxonomy migration; server-side reports
  `~/hub_phase1_report.md`, `~/hub_phase2_cleanup.md` on
  10.0.0.39).
- 9-family `ModelCategory` taxonomy in Prisma matches the
  packaging taxonomy.

**Current hub catalogue (queried live this session, 15 models):**
LLM: TinyLlama-1.1B-Chat, Qwen3-30B-A3B, DeepSeek-MoE-16B-Chat,
Qwen3-30B-A3B-Thinking · IMAGE: PixArt-Sigma-XL-1024,
PixArt-XL-1024, Sana-1600M-MultiLing, Sana-1600M-4Kpx-BF16,
Flex.1-alpha · VIDEO: SANA-Video-2B-720p · STT:
Whisper-V3-Turbo · MULTIMODAL: Janus-Pro-7B · TTS: Chatterbox,
Orpheus-3B · AUDIO_LLM: Voxtral-Mini-3B.

NeuroBrix client side (`cli/commands/registry.py`):
- **Browse** (`cmd_hub`): `GET /api/models?limit=&category=&q=`
  (no auth).
- **Pull** (`cmd_import`): `GET /api/models/{org}/{name}` →
  `GET /api/models/{org}/{name}/download` → stream to
  `~/.neurobrix/store/` → extract to `~/.neurobrix/cache/`.

**The real gap is NOT "activate the hub" — the hub is live.**
The concrete gap: the 10 upscaler `.nbx` built this session
(Real-ESRGAN/SwinIR/HAT/Swin2SR) are **NOT published** — a live
query of `?category=UPSCALER` returns NONE. Publishing them is a
packaging-side `publish` operation (private companion report
Section 4-build), not an infra project.

---

## SECTION 5 — Active technical debt

### docs/follow-ups/ (status read from each file)

| File | Status | Severity / scope |
|---|---|---|
| `janus_triton_anticorrelation.md` | **CLOSED** (root cause fixed) | — |
| `kokoro_cudnn_batch_norm_regression.md` | **OPEN** (symptom-stage) | Kokoro `::native` `aten::cudnn_batch_norm` undefined tensor; medium |
| `layer7-prism-dtype-override.md` | **OPEN** | blocks PixArt-Alpha/Sigma in `--triton`; medium |
| `layer8-computable-buffers-extension.md` | **OPEN** | blocks Sana 4Kpx in `--triton`; medium-large |
| `layer9-sana-4kpx-vae-memory.md` | **OPEN** | blocks `Sana_1600M_4Kpx_BF16` `--triton`; large (VAE memory) |
| `pixart_triton_arena_inter_run_bug.md` | **ARCHIVED** (superseded post Layer 6) | — |
| `qwen3_vram_investigation.md` | **OPEN** (investigation) | Qwen3-30B `v100-16g --triton` ~10.8 GB peak vs target; medium |
| `p-triton-im2col-kernel.md` | **OPEN** (this session) | HAT OCAB `nn.Unfold` → no Triton kernel; blocks HAT triton + triton-seq; scope: 1 Triton-pure kernel + dispatch + equivalence test |
| `p-container-embed-orphan-scalars.md` | **OPEN** (this session) | in-forward scalar constants materialised as 0-dim placeholder; negligible at container trace size, not guaranteed when arbitrary-size tiling unblocks; relates to BL-1 |

### Upscaler v1 backlog (`docs/verdicts/p_neurobrix_upscalers/v1_backlog.md`)

- **BL-1** — container `profile.json` `config` lacks `upscale` +
  `window_size`, so `tiling_engine.from_component_config()`
  returns None → arbitrary-input-size upscaling gated; inputs
  validated only at the container trace size (e.g. 64×64). Root
  is model-packaging (config not propagated into the packaged
  container profile), NOT the runtime tiling logic (correct +
  data-driven). **OPEN**, medium.
- **BL-2** — `manifest.model_type` is `None` for upscaler
  containers. Cosmetic. **OPEN**, low.

### Cross-referenced backlogs (from memory / prior verdicts)

- **Sana 4Kpx `addmm::7` `cudaErrorIllegalAddress`** — pre-existing
  intrinsic crash in `block.0.cross_attn.out.0`; tracked
  separately. **OPEN**, high for Sana 4Kpx triton, isolated.
- **P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP** — triton-cpu has no
  PyPI wheel (upstream OSS issue); documented + escalated, NOT
  internally forked (R25). **OPEN/external**, low actionability
  on our side.
- **P-OP-LEVEL-CROSS-DEVICE-SPLIT (Gap B)** — multi-GPU op-level
  split for NBXTensor (component_placement / pipeline_parallel
  with NBXTensor). **OPEN**, large; lower priority than
  live-watermark items per prior Sana doctrine.
- **P-AUTOTUNE-OFFLINE** — see Section 9.

---

## SECTION 6 — Models present as `.nbx`

`/home/mlops/NeuroBrix_System/models/` (build-output tree):

| Model | Family | Path | State |
|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | llm | models/llm/… | clean |
| real-esrgan-x2/x4/x8 | upscaler | models/upscaler/… | clean (path-leak-scanned 0) |
| swin2SR-classical-sr-x2-64 / -x4-64 / -realworld-sr-x4-64-bsrgan-psnr | upscaler | models/upscaler/… | clean |
| swinir-classical-x2 / -x4 | upscaler | models/upscaler/… | clean |
| hat-s-x4 / hat-l-x4 | upscaler | models/upscaler/… | clean |

All 10 upscaler `.nbx` re-scanned this session for the
`/home/mlops` absolute-path leak → **0 leak files** across
topology.json + profile.json (the prior path-leak chantier's
fix holds for fresh builds).

- `~/.neurobrix/store/` — **empty** (no pulled `.nbx`; expected,
  hub not exercised).
- `~/.neurobrix/cache/` — many extracted models from prior
  sessions (Flex.1-alpha, Janus-Pro-7B, Kokoro-82M, PixArt-*,
  Qwen3-30B, SANA-Video, Sana_1600M_*, TinyLlama, VibeVoice,
  Voxtral, canary-qwen, chatterbox, deepseek-moe, granite-speech,
  hat-s/l-x4, openaudio-s1-mini, orpheus-3b, …). Dev-extracted
  working copies, not authoritative artefacts.

---

## SECTION 8 — TODO / FIXME / XXX / HACK in NeuroBrix code

Total raw matches in `src/neurobrix/`: 217 — but **~203 are in
`src/neurobrix/kernels/triton_kernels_ref/`**, a vendored
third-party reference kernel library (FlagGems, applied_ai,
triton tutorials). Those are upstream comments, not NeuroBrix
debt.

**NeuroBrix-authored** (14, excluding tests / vendored ref /
cosmetic `shard_XXX` format strings):

| File:line | Note | Category |
|---|---|---|
| `kernels/dispatch.py:343` | proper Triton linspace kernel | cosmetic (works via fallback) |
| `kernels/dispatch.py:700-701` | `index_put`/`index_put_` are identity lambdas `# TODO` | **critical-latent** — silent no-op if a model uses index_put on the triton path; verify no traced model hits it |
| `kernels/nbx_tensor.py:1945` | fill kernel for ones | cosmetic |
| `kernels/utils/shape_utils.py:400` | fast div/mod question | cosmetic |
| `kernels/utils/tensor_wrapper.py:60` | `TODO[kunlunxin]` torch upgrade 2025.04 | obsolete (vendored-origin marker) |
| `kernels/utils/random_utils.py:17,43` | frontend cleanup / kunlunxin marker | cosmetic / obsolete |

The single item worth a real look: `dispatch.py:700-701`
`index_put`/`index_put_` mapped to identity — confirm no
runtime-exercised graph relies on scatter/index_put on the
triton path (silent-correctness, not a crash).

---

## SECTION 9 — Deliberately deferred post-dev optimisations

- **Triton autotune** — `@triton.autotune` is AUTHORIZED only
  for `mm/bmm/addmm/conv2d` (Phase 1.5 exception, documented in
  root `CLAUDE.md`). All OTHER optimisation tuning is deferred to
  a dedicated post-development chantier **P-AUTOTUNE-OFFLINE**.
  The Volta sm_70 `mm/bmm/addmm` vs cuBLAS structural gap is
  **not a tuning bug**; closing it needs P-AUTOTUNE-OFFLINE — do
  NOT re-test the already-measured static-config patterns
  (tl.dot 3-arg, tl.assume, FlagGems static, force_fp32 bypass —
  all 0–2% on Sana 1024 mm).
- **Triton-pure Level 2** — monolithic fully-fused `@triton.jit`
  kernels (zero per-launch overhead, no halo over-fetch) are
  deferrable per the 2-level Triton-pure doctrine, provided the
  gap is a named backlog chantier (`P-TRITON-IM2COL-KERNEL` is
  Level-1 → NOT deferred; named and open).
- **Deferred-free drain tuning** (`triton/sequence.py`) — the
  periodic `_deferred` drain thresholds (`NBX_DEFERRED_DRAIN_*`)
  are heuristic defaults; finer policy is post-dev.
- **`DeviceAllocator` caching pool** (`NBX_ALLOC_POOL`) — default
  OFF; pool best-fit policy is explicitly backlog.

---

## SECTION 10 — Recommended next chantiers (Hocine picks)

1. **P-MODEL-COVERAGE-CONTINUATION (primary, user-directed)** —
   *scope: large, the main thrust.* The hub is LIVE and already
   serves 15 models across LLM/IMAGE/VIDEO/STT/MULTIMODAL/TTS/
   AUDIO_LLM. The next body of work is **completing NeuroBrix's
   coverage of the remaining models** — audio / STT / TTS /
   multimodal / video — i.e. packaging them to `.nbx` and
   validating the `nbx run` per-family runtime path, then
   publishing. Precondition Hocine set: first **audit all
   technical debt** from prior sessions to clear the ground
   (this report's Section 5 + the dedicated debt-audit doc).
   Immediate concrete sub-item: the 10 upscaler `.nbx` from this
   session are built but **not yet published** (live
   `?category=UPSCALER` = NONE) — publish them (a routine
   packaging-side `publish`, the hub infra already exists; see
   private companion Section 4-build). This is NOT an
   infrastructure project — the hub is deployed and operational.

2. **P-TRITON-IM2COL-KERNEL** — *scope: small-medium,
   well-bounded.* One Triton-pure `aten::im2col` kernel +
   dispatch + numeric equivalence test. Unblocks HAT
   `triton`/`triton-seq` → HAT reaches 4/4 modes; generalises to
   any future unfold-based architecture. Lowest-risk
   highest-clarity win; closes a named open follow-up.

3. **P-RUNTIME-FAMILY-COVERAGE-AUDIT** — *scope: medium.* `nbx
   run` claims 9 families; many models are extracted in cache
   (audio/STT/TTS/multimodal/video) but their CLI→runtime path
   per family is not freshly validated this cycle. Systematic
   one-cell-per-family smoke (compiled + triton) with R29
   artefacts would map the real coverage matrix and surface the
   next concrete blockers (likely overlaps with layer7-9 triton
   follow-ups for PixArt/Sana).

4. **P-DISPATCH-INDEX-PUT-CORRECTNESS** — *scope: small.*
   Resolve `dispatch.py:700-701` (`index_put`/`index_put_` =
   identity). Audit traced models for scatter/index_put on the
   triton path; if any are exercised this is a silent-correctness
   bug, not cosmetic. Cheap to investigate, potentially
   important.

5. **BL-1 closure (arbitrary-size upscaling)** — *scope: medium.*
   Propagate `upscale` + `window_size` into the packaged
   container profile so `TilingEngine` activates; pairs naturally
   with `P-CONTAINER-EMBED-ORPHAN-SCALARS` (both needed for
   correct shifted-window transformer tiling at arbitrary input
   size). Turns the upscaler tier from "trace-size only" into
   "any input size".

---

## FIN

Working tree clean (NeuroBrix + private companion). Remotes up
to date. This report is the public half; the model-packaging
half is the private companion at the analogous path in that
repo.
