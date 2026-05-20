# Dette G — Hub rebuild / reupload status (2026-05-20)

Branch `p-debt-settlement-batch-1`. Mandate: re-build + re-upload
pre-path-leak .nbx files published on `neurobrix.es`, gated on
C/D/E Hocine R29 OK (received), excluding models whose quality is
not validated.

## Section 1 — Inventory (hub DB Prisma read-only)

15 models currently published in the Postgres `Model` table at
`10.0.0.39:/var/www/neurobrix.es`. SSH-readable read-only via
the existing Prisma client; total ~313 GB across the 15 entries.

Stored on MinIO at `10.0.0.36:9000`, bucket `neurobrix`, key
prefix `models/<org>/<name>.nbx`. The MinIO host has its own SSH
identity (no `mlops` key configured — `Permission denied
(publickey,password)`).

| # | org/name | category | size | `updatedAt` | rebuild target? |
|---|---|---|---|---|---|
| 1 | ostris/Flex.1-alpha | IMAGE | 26.29 GB | 2026-03-25 | YES — pre-Phase-D (path-leak) |
| 2 | NVlabs/SANA-Video-2B-720p | VIDEO | 18.45 GB | 2026-04-02 | YES — pre-Phase-D, **AND** has the standing P-PRISM-VIDEO-5D-UNPACK runtime bug — Hocine may want to defer this one until the video allocator is fixed |
| 3 | deepseek-ai/DeepSeek-MoE-16B-Chat | LLM | 32.81 GB | 2026-04-30 | YES — pre-Phase-D |
| 4 | Qwen/Qwen3-30B-A3B-Thinking | LLM | 61.36 GB | 2026-04-30 | YES — pre-Phase-D |
| 5 | NVlabs/Sana-1600M-MultiLing (Sana 1024) | IMAGE | 12.97 GB | 2026-04-30 | YES — pre-Phase-D. Quality validated R29 (Dette C: prompt-style sensitivity, not broken; "red apple" prompt produces photoreal output). |
| 6 | NVlabs/Sana-1600M-4Kpx-BF16 | IMAGE | 12.97 GB | 2026-04-30 | YES — pre-Phase-D. Compiled mode green (P-SANA-4KPX-RUNTIME closure 2026-05-13); triton standing-INDETERMINATE. Compiled output validated. |
| 7 | deepseek-ai/Janus-Pro-7B | MULTIMODAL | 14.86 GB | 2026-04-30 | YES — pre-Phase-D |
| 8 | **resemble-ai/Chatterbox** | TTS | 2.21 GB | 2026-04-30 | **DEFER** — broken (P-AUDIO-CHATTERBOX-LOOP: 82s saturated output). Leave as-is until audio chantier fixes it. |
| 9 | TinyLlama/TinyLlama-1.1B-Chat | LLM | 2.22 GB | 2026-05-10 | YES — pre-Phase-D |
| 10 | Qwen/Qwen3-30B-A3B | LLM | 61.37 GB | 2026-05-15 | MAYBE — same day as Phase-D (May 15). Verify leak status before deciding. |
| 11 | PixArt/PixArt-Sigma-XL-1024 | IMAGE | 21.84 GB | 2026-05-17 | NO (likely clean) — post-Phase-D. Verify leak status; skip if clean. Quality OK per Hocine Ch7/Ch8 R29 (over-saturation acknowledged-known). |
| 12 | PixArt/PixArt-XL-1024 | IMAGE | 21.90 GB | 2026-05-18 | NO (likely clean) — post-Phase-D. Quality OK per Hocine Ch7/Ch8 R29 (transient band P-PIXART-XL-VOLTA-WHITE-BAND tracked, not deterministic). |
| 13 | **canopylabs/Orpheus-3B** | TTS | 15.17 GB | 2026-04-30 | **DEFER** — both modes fail in the harness (pre-existing). Audio chantier scope. |
| 14 | mistralai/Voxtral-Mini-3B | AUDIO_LLM | 9.38 GB | 2026-04-30 | YES — pre-Phase-D. Voxtral hallucination (P-VOXTRAL-HALLUCINATION) is a build-side processor issue, not runtime — rebuild does NOT fix it. Hocine may want to defer pending the build-side processor patch. |
| 15 | openai/Whisper-V3-Turbo | STT | 1.63 GB | 2026-04-30 | YES — pre-Phase-D |

Re-upload-now candidates (post-validation): 10 models (entries 1-7, 9-10, 14-15).
Defer-until-audio-chantier: 2 (Chatterbox, Orpheus).
Likely-already-clean (skip): 2 (PixArt-σ, PixArt-α).
Plus 2 models flagged for Hocine sign-off before rebuild (SANA-Video pending P-PRISM-VIDEO-5D-UNPACK; Voxtral pending processor patch).

## Section 2 — Path-leak fix mechanism (already-built)

The build subtree already has the resolution tooling:

- the cleanup script (in the build subtree, `scripts/cleanup_path_leak.py`) — idempotent, in-place
  strip of two leak fields:
  - `topology.json::components.<name>.path`
  - `components/<name>/profile.json::component.source_path`
  Handles both extracted directories and packaged .nbx zip
  archives via temp + atomic replace.
- Default scan roots: `~/.neurobrix/cache/`,
  `/home/mlops/NeuroBrix_System/models/`,
  `/home/mlops/NeuroBrix_System/.cache/regression_baselines/`,
  `/home/mlops/NeuroBrix_System/.cache/graphs/`.

Empirically: every Dell-side extracted cache I probed
(`grep -rIl "/home/mlops/"` on TinyLlama / Sana 1024 / PixArt-XL /
Voxtral cache dirs) is **leak-clean** — Phase D's local cleanup
ran successfully across the cached matrix.

The default scan roots do **not** include any path that maps to
MinIO at `10.0.0.36`. The hub-stored .nbx are presumed pre-leak
for models published before 2026-05-15 (Phase D commit
`4a1684e`); models published after that date are likely already
clean.

## Section 3 — Blocker (escalation needed)

Three hub-side operations are systematically blocked in the
agent's auto-mode classifier:

1. **SSH to `10.0.0.36` (MinIO host)** — `Permission denied
   (publickey,password)`. No `mlops` key configured on MinIO.
2. **`mc` (MinIO client)** — not installed on hub `10.0.0.39`.
   Installing requires root.
3. **`neurobrix import <model>` from `neurobrix.es`** — blocked
   by classifier ("Pulling a model .nbx from the public hub to
   a local cache is downloading external code/data to execute").

Without one of these access paths, the rebuild+reupload pipeline
cannot run end-to-end from the agent. The mandate's "premier
upload → ARRÊT + confirme-moi la chaîne end-to-end (MinIO key +
DB upsert + nbx import frais clean + run OK)" requires the
agent to actually upload to MinIO, which needs one of the above.

Per the mandate's escalation rule "Pas de création de compte/token
(escalade Hocine si token manquant ou expiré)" — this is the
legitimate "token-manquant"-equivalent case.

## Section 4 — What Hocine needs to unblock Dette G

One of:

- **Option A (preferred — least change)**: install `mc` on the
  hub (10.0.0.39) and configure an alias with the existing
  MinIO admin credentials from `/var/www/neurobrix.es/.env`
  (`S3_ACCESS_KEY`, `S3_SECRET_KEY`, endpoint
  `http://10.0.0.36:9000`, bucket `neurobrix`). With `mc`
  present, the agent can `mc cp`, `mc mirror`, etc. via SSH.
- **Option B**: add the agent's SSH public key to
  `mlops@10.0.0.36` `~/.ssh/authorized_keys` (read-only suffices
  — agent can SSH-mount the MinIO data dir if file-level access
  is permitted).
- **Option C**: whitelist `python -m neurobrix import` in the
  agent's permission settings so the download path works through
  the existing CLI (the CLI handles auth + signed URLs through
  the hub's web layer).

Any of these unblocks the rebuild loop. Option A is the smallest
change and matches how Hocine probably already operates manually.

## Section 5 — Pipeline once unblocked (one-shot batch)

Once access is available, the agent will:

1. For each "rebuild target = YES" model (10 candidates):
   1. `mc cp` from `neurobrix/models/<org>/<name>.nbx` to
      `/tmp/`.
   2. `grep -l "/home/mlops/" <local>.nbx` — verify the leak
      actually exists in that specific file.
   3. Run the build-subtree cleanup script
      (`scripts/cleanup_path_leak.py /tmp/<...>.nbx`) — in-place strip.
   4. `grep -l "/home/mlops/" <cleaned>.nbx` — verify clean.
   5. Smoke R29: `python -m neurobrix import /tmp/<...>.nbx`
      → `neurobrix run --model …` (mode-dependent: image →
      red-apple R29; audio_llm → STT-auto-transcribed; LLM →
      coherent text per golden).
   6. **PAUSE on the first model**: confirm DB updated_at +
      MinIO presence + fresh import works end-to-end. The
      mandate's "premier upload → ARRÊT".
   7. After Hocine OK, batch the remaining 9.
2. For each "DEFER" model (Chatterbox, Orpheus): no action.
   They stay on the hub with their current .nbx until the
   audio chantier ships a fix.
3. For each "likely-clean" (PixArt-σ, PixArt-α): verify leak
   status; skip rebuild if clean.
4. For each "Hocine sign-off" (SANA-Video, Voxtral): pause for
   Hocine decision (rebuild without addressing underlying issue
   = wasteful; defer = quality state unchanged).

## Section 6 — Disposition

**Dette G is ESCALATED to Hocine** for the access unblock
(Section 4 options A/B/C). Inventory + categorization + pipeline
design are complete and committed. The actual rebuild/reupload
batch is a separate execution step waiting on the unblock; once
Hocine chooses an option and applies it, the batch can run in
a follow-on session without further design work.

The remaining tasks in the mandate (audio family état des lieux)
do **not** require hub access and proceed independently.
