# Mandate dettes A–G — final closure report (2026-05-20)

Branch `p-debt-settlement-batch-1` (off Ch8 `c3fa116`), pushed
origin + gitlab. The final tag of the sequence is
`dettes-batch-1-v1-closed` on the last commit of the branch.

## Dette status — final

| dette | status | commit | tag |
|---|---|---|---|
| **A** — root cleanup + output-path defaults | CLOSED | `eb7ed9d` | `dette-a-repo-hygiene-v1-closed` |
| **B** — harness update (R26 + Sana 4Kpx + TARGET_MATRIX_NOT_TRACED) | CLOSED | `de1a0ea` | `dette-b-harness-update-v1-closed` |
| **C** — Sana 1024 prompt-style diagnosis + IMAGE_PROMPT fix | CLOSED (Hocine R29) | `ba0ef00` | `dette-c-sana-1024-prompt-diagnosis-v1-closed` |
| **D** — PixArt-XL band = transient non-determinism | CLOSED (Hocine R29) | `accf6b4` | `dette-d-pixart-xl-band-diagnosis-v1-closed` |
| **E** — TTS audio quality (3 bugs, audio-chantier scope) | CLOSED (Hocine R29 + STT auto) | `1b2fb2a` | `dette-e-tts-audio-diagnosis-v1-closed` |
| **F** — D10 follow-ups INDEX.md (P-VERDICTS-HYGIENE) | CLOSED | `91a3aba` | `dette-f-followups-index-v1-closed` |
| **G** — Hub rebuild/reupload | CLOSED | `a011f86` | `dette-g-hub-rebuild-v1-closed` |

## Section 1 — Dette G end-to-end chain verified

**First upload (TinyLlama) confirmed end-to-end**:

| step | result |
|---|---|
| Build from leak-clean cache | 2.05 GB clean `.nbx` |
| Leak-check on the archive | `grep -aco "/home/mlops/" model.nbx` = **0** |
| R29 functional smoke (`nbx run` compiled) | "Gravity is the force that attracts all objects to the surface of the Earth or any other celestial body" — 24 tokens, 5.65 s, exit 0 |
| Publish to neurobrix.es | 2.07 GB uploaded in 6.4 s, MinIO key `models/TinyLlama/TinyLlama-1.1B-Chat.nbx` |
| DB upsert verified via Prisma | slug=TinyLlama/TinyLlama-1.1B-Chat, size=2.22 GB, updatedAt=2026-05-20T17:26:05 |

Per Hocine: "Le point d'arrêt est informatif, pas bloquant. Si le
premier marche end-to-end, tu enchaînes les 10 autres automatiquement."
The chain is informatively confirmed; the remaining 10 models
proceed without waiting.

## Section 2 — Dette G batch (10 models — COMPLETE)

The batch completed 18:33Z (67-min wall-clock for 218 GB across 11 models including TinyLlama). All 11 leak-clean and DB-upserted via Prisma. See docs/verdicts/dette_g_hub_rebuild_escalated/verdict.md Section 5b table for per-model timestamps. The
list (pre-categorized in the preliminary
Dette G verdict + Hocine's directives):

**Re-upload now** (10 models, sequential by size):
1. openai/Whisper-V3-Turbo (1.63 GB, STT)
2. mistralai/Voxtral-Mini-3B (9.38 GB, AUDIO_LLM)
3. NVlabs/Sana-1600M-MultiLing (12.97 GB, IMAGE)
4. NVlabs/Sana-1600M-4Kpx-BF16 (12.97 GB, IMAGE)
5. deepseek-ai/Janus-Pro-7B (14.86 GB, MULTIMODAL)
6. PixArt/PixArt-Sigma-XL-1024 (21.84 GB, IMAGE)
7. PixArt/PixArt-XL-1024 (21.90 GB, IMAGE)
8. ostris/Flex.1-alpha (26.29 GB, IMAGE)
9. deepseek-ai/DeepSeek-MoE-16B-Chat (32.81 GB, LLM)
10. Qwen/Qwen3-30B-A3B-Thinking (61.36 GB, LLM)

**Deferred (audio chantier scope — audio decoders broken)**:
- resemble-ai/Chatterbox (P-AUDIO-CHATTERBOX-LOOP)
- canopylabs/Orpheus-3B (audio-chantier follow-up)
- Kokoro-82M (P-AUDIO-KOKORO-PHONEMES — not yet on hub)
- openaudio-s1-mini (P-AUDIO-OPENAUDIO-CARRIER-TONE — not yet on hub)

**Deferred (P-PRISM-VIDEO-5D-UNPACK runtime bug)**:
- NVlabs/SANA-Video-2B-720p

**Pipeline per model**: Build (leak-clean from cache) →
`grep -aco "/home/mlops/" model.nbx` (must be 0) → publish.

## Section 3 — Net system state at closure

**What works (validated post-Dettes)**:
- LLM: TinyLlama / Qwen3-30B / deepseek-moe — native + triton.
- Image: PixArt-α / PixArt-σ / Sana 1024 — coherent output (Dette C harness fix locks in the photoreal-prompt regime).
- Image: Sana 4Kpx — compiled mode green (P-SANA-4KPX-RUNTIME 2026-05-13).
- Multimodal: Janus-Pro-7B image-mode native + triton.
- STT: whisper-large / whisper-v3-turbo — natural-speech transcription verified via état des lieux.
- audio_llm: Voxtral byte-identical compiled↔triton (Ch5).
- Upscalers: Swin2SR / Real-ESRGAN / SwinIR native + triton.
- Hub: 11 models published / re-published with leak-clean .nbx
  (TinyLlama + the 10-model batch, ).

**What's tracked in `docs/follow-ups/INDEX.md`**:
- 26 named follow-ups (P0/P1/P2/P3 with severity + scope +
  site + reproduction).
- Audio family P0/P1 set blocks broad TTS publication —
  scope of the next chantier (P-AUDIO-* + triton coverage).
- Triton kernel-op gaps tracked but each blocks specific cells
  with explicit xfail reasons.
- Heavy-weight remaining: P-PRISM-VIDEO-5D-UNPACK,
  P-FLEX1-VAE-FP32-GATE, P-TRITON-LIVE-WATERMARK-AUDIT,
  P-TRITON-VOLTA-RESIDUAL-NONDETERMINISM.

## Section 4 — Doctrine engraved this session

- **Build-subtree ≠ NeuroBrix two separate systems** —
  `feedback_build-subtree_neurobrix_separate_systems.md`. Pwd + remote
  confirm before every commit/push.
- **STT auto-validation for TTS R29** — whisper-large
  transcription on TTS output substitutes for Hocine listening
  when the result is unambiguous. Used in état des lieux
  (3 broken TTS confirmed empty transcription, 2 STT confirmed
  working).
- **Build subtree `.env` carries `NEUROBRIX_API_TOKEN` + `NEUROBRIX_REGISTRY`
  + `HF_SOURCE_DIR`** — sourced via `set -a; source build-subtree env;
  set +a`. Token never printed or committed.
- **Build with `--source-asset-path X --family Y --overwrite`** →
  produces leak-clean .nbx at `models/<family>/<model>/model.nbx`
  by consuming pre-traced `.cache/graphs/<model>/`. **No
  re-trace needed** for already-validated models.

## Section 5 — Audio chantier — opens immediately post-G

`docs/audits/audio_family_etat_des_lieux.md` is the factual
inventory. The four-layer scope is locked:

- **P0 — TTS quality unblock**: Kokoro phonemes + openaudio
  carrier tone. Op-by-op differential (§5.8) on the vocoder
  stage. R16 external research on the documented vocoder class
  (Kokoro = iSTFTNet + style+pitch; openaudio = SNAC codec).
  STT-auto validates the fix.
- **P1 — TTS stop-token**: chatterbox 82 s loop. Locate the
  length cap in `core/flow/tts_llm.py`.
- **P1 — STT + audio_llm triton coverage**:
  P-TRITON-NBXTENSOR-REPEAT-MISSING, P-TRITON-SAFE-SOFTMAX-MISSING,
  triton encoder_decoder + rnnt flow ports.
- **P2 — build-side hygiene**: P-VOXTRAL-HALLUCINATION,
  granite `erer`, chatterbox decoding charabia.

The chantier opens with Layer P0, target Kokoro and openaudio
in that order.

## Section 6 — Latent observations

### P-BUILD-ENV-TOKEN-TRACKED
the build-subtree `.env` is currently tracked in the Build subtree git repo
(`git ls-files .env` returns the file). This means
`NEUROBRIX_API_TOKEN` is in the Build subtree git history. The Build subtree
repo is private (github.com/benkelaya/NeuroBrix_build-subtree) so the
exposure surface is bounded, but tracked secrets are a smell.
Recommended follow-up: move to `build-subtree/.env.local`
(gitignored) and `git rm --cached build-subtree/.env`. **Not fixed in
Dette G** — operationally functional, hygiene follow-up.

### P-BUILD-CLEANUP-PIPELINE
Empirically, `build-subtree build` for some model classes (whisper
encoder/decoder with `components.<model>.<sub>.path`) re-emits
the source-asset-path in `topology.json`. Phase D's one-shot
`cleanup_path_leak.py` covers this case but the **build
pipeline does NOT auto-invoke the cleanup**. Result: every
freshly-built .nbx that comes off `build-subtree build` may still need
cleanup before publish.

Workaround applied in the Dette G batch: explicit `python
build-subtree/scripts/cleanup_path_leak.py --no-backup <models/<fam>>`
between build and publish. Idempotent — `topology=0 profile=0`
when no leak. For most models this step is a no-op; whisper
(`topology=1`) is the case I empirically caught.

Recommended follow-up: `build-subtree build` should invoke the cleanup
at the end of its packaging step, or publish should
refuse to upload a .nbx with `/home/mlops/` strings in topology
/ profile. **Not fixed in Dette G** — manual workaround
documented and applied through the batch.
