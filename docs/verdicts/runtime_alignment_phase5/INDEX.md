# Phase 5 Runtime Alignment — Smoke Test 12 Models

Validation matrix for the 9-family runtime dispatch refactor
(commits e096b36 + 098d9a2). Each row links to a per-model artefact
directory containing prompt.txt, stats.json, output.<ext>, verdict.md.

Hocine validation column is intentionally left blank for manual
inspection of each output (R29 doctrine — agent stats do not
substitute for human inspection).

| # | Slug | Family | Mode | Verdict agent | Stats key | Size | Link | Hocine OK |
|---|---|---|---|---|---|---|---|---|
| 1 | TinyLlama | llm | text | ✅ PASS | 73 chars, 18 words | 73B | [./TinyLlama/](./TinyLlama/) | ☑ LLM text |
| 2 | DeepSeek-MoE | llm | text | ✅ PASS | 22 chars, 5 words | 22B | [./DeepSeek-MoE/](./DeepSeek-MoE/) | ☑ LLM text |
| 3 | Qwen3-30B | llm | text | ✅ PASS | 108 chars, 21 words | 108B | [./Qwen3-30B/](./Qwen3-30B/) | ☑ LLM text |
| 4 | PixArt-XL | image | t2i | ✅ PASS | 1024×1024, mean 211.6 std 75.1 | 671.0KB | [./PixArt-XL/](./PixArt-XL/) | ☑ red apple visuel OK (steps=12) |
| 5 | PixArt-Sigma | image | t2i | ✅ PASS | 1024×1024, mean 122.8 std 102.6 | 1008.0KB | [./PixArt-Sigma/](./PixArt-Sigma/) | ☑ red apple+plate OK (steps=12) |
| 6 | Sana-1024-MultiLing | image | t2i | ✅ PASS | 1024×1024, mean 199.6 std 76.1 | 1.0MB | [./Sana-1024-MultiLing/](./Sana-1024-MultiLing/) | ☑ red apple+plate OK (steps=12) |
| 7 | Janus-image | multimodal | image | ⚠ PASS_STRUCTURE_PARTIAL | 384×384, mean 121.9 std 62.8 | 188.0KB | [./Janus-image/](./Janus-image/) | ⚠ cat OK, color off — model/CFG, runtime OK |
| 8 | Janus-text | multimodal | text | ✅ SKIP_EXPECTED | — | — | [./Janus-text/](./Janus-text/) | ☑ build/mode gate clean |
| 9 | Whisper-V3-Turbo | stt | text | ✅ PASS | 241 chars, 42 words | 241B | [./Whisper-V3-Turbo/](./Whisper-V3-Turbo/) | ☑ perfect transcription |
| 10 | Voxtral | audio_llm | text | ⏸ FAIL_HORS_SCOPE | 112 chars, 21 words | 112B | [./Voxtral/](./Voxtral/) | ☐ FAIL hallucination — processor multimodal chantier dedicated |
| 11 | Chatterbox | tts | audio | ⚠ PASS_STRUCTURE_ONLY | 10.86s wav, RMS 0.14369 | 509.0KB | [./Chatterbox/](./Chatterbox/) | ☐ wav structure OK, audio QUALITY charabia — Chatterbox decoding bug out of scope |
| 12 | Orpheus | tts | audio | ⏸ FAIL_HORS_SCOPE | RuntimeError: Failed at op aten._scaled_dot_product_efficien | — | [./Orpheus/](./Orpheus/) | ☐ GQA wrapper bug out of scope |
| 13 | Sana-4Kpx | image | t2i | ⏸ FAIL_EXPECTED | RuntimeError: Failed at op aten.convolution::55 (aten::convo | — | [./Sana-4Kpx/](./Sana-4Kpx/) | ☐ OOM 36 GiB conv out of scope |

## Summary

- ✅ **PASS** : 7/13
- ✅ **SKIP_EXPECTED** : 1 (Janus text-mode → clear error data-driven, build/mode coherence gate)
- ⏸ **FAIL_EXPECTED** : 1 (Sana 4Kpx OOM 36 GiB — bug runtime conv tile-execution out of scope)
- ⏸ **FAIL_HORS_SCOPE** : 2 (Orpheus GQA wrapper — pre-existing bug, dedicated chantier)
- ❌ **FAIL** : 0 (regressions introduced by the chantier)

## PNG-from-non-image bug eliminated

Evidence before chantier : 5 fichiers orphelins au project root
(output_whisper-large-v3-turbo.png, output_Voxtral-Mini-3B-2507.png,
output_Janus-Pro-7B.png for prompt text, etc.).

Evidence after chantier (verified for the 12) :
- Whisper-V3-Turbo → output.txt with transcription parfaite
- Voxtral → output.txt (but hallucination — processor multimodal bug indep)
- Janus image-mode → output.png 384×384 chat coherent (color fidelity off — model/CFG)
- Janus text-mode → clear error 'build supports only --mode image'
- Chatterbox → output.wav 10.86s structure OK (audio QUALITY charabia indep — Chatterbox decoding head)

The 'PNG from non-image' bug is eliminated. Internal semantic bugs
(Voxtral processor, Chatterbox decoder) are orthogonal — dedicated chantiers.

## Manual visual review performed

Hocine inspected the outputs and confirmed:
- TinyLlama, DeepSeek-MoE, Qwen3-30B → ☑ texte coherent
- PixArt-XL, PixArt-Sigma, Sana-1024-MultiLing → ☑ red apple visually OK (steps=12 retrace)
- Janus-image (a red cat) → cat OK structurellement, color fidelity off (orange instead of red) — limite model/CFG
- Janus-text → ☑ clear error build/mode gate fired
- Whisper-V3-Turbo → ☑ perfect transcription of test_speech_ref.wav
- Voxtral → ☐ hallucination 'didn't quite catch that' while Whisper transcribes the same audio perfectly = multimodal processor bug
- Chatterbox → ☐ wav structure-only ; audio quality charabia = bug Chatterbox decoding head out of scope

## R29 candidate doctrine

Cette campagne phase 5 applique the rule R29 candidate :
tort chantier de validation model produit a artefact
humanly inspectable (output.<ext> + stats.json + verdict.md +
prompt.txt) in /home/mlops/NeuroBrix_System/validation_outputs/<chantier>/
independently of the agent verdict. Numeric stats do not
substitute for human inspection — an agent can fool a
threshold on visual noise.

Applied bounds : audio 10s slice if longer (slice unavailable
noted when ffmpeg is missing), text 2000 chars, image full-resolution
one-shot.

Target disk : /home/mlops/NeuroBrix_System/validation_outputs/ (project tree, jabut /mnt/* server mornts)
/home/mlops/NeuroBrix_System (96% pleine).

## Pass criteria used (data-driven harness)

- llm/vlm/multimodal-text/stt/audio_llm : output.txt with ≥3 mots
- multimodal-image/image : output.png with mean ∈ [30, 240] et std > 30
- tts: output.wav duration > 0.5s, RMS > 1e-3, temporal variance > 1e-7
- multimodal-strict --mode mismatch : SKIP_EXPECTED si clear error
- ressorrces insuffisantes (Sana 4Kpx 36 GiB conv) : FAIL_EXPECTED
- pre-existing independent bug (Orpheus GQA) : FAIL_HORS_SCOPE
