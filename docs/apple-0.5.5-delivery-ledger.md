# Apple 0.5.5 — delivery ledger

One line per hub-catalogue container. The delivery criterion: every container
that physically fits this machine (M4 Pro, 24 GB unified, disk trigger
22 GiB) runs in all three modes with artefacts judged by an instrument
outside the engine, and the keys those runs demand are certified under the
current pin (`triton 3.8.0+git4a15f415 mps`); every container that does not
fit is refused in writing with the figure that refuses it. A refusal with a
number is a delivery; a model silently skipped is not.

Modes: C = --compiled (plain PyTorch on mps), T = --triton,
S = --triton-sequential. Every run under the 4 GB memory floor, Parallels
VM running. Instruments: LLM = literal text comparison; STT = transcript
against the recording; TTS = speech-to-text reads the wav back, sentence
exact; image = external degeneracy judge (std, distinct values).

| container | size | verdict | figures |
|---|---|---|---|
| llm/granite-3.1-1b-a400m-instruct | 2.7 GB | **DELIVERED** 2026-09-21 | C/T/S all rc=0, "The capital of France is **Paris.**" (LLM judge, exact), 0 autotune misses in all three modes — its shapes were already covered by the 974-key directory; MoE kernels run fixed configs by design. First mixture-of-experts delivery on Apple; the standing MoE refusal lifted through the driver-selection door on this run's judgment, not on a probe. |
| tts/Kokoro-82M | 366 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, ASR reads the sentence exactly in each mode. |
| tts/chatterbox | 2.1 GB | **DELIVERED** 2026-09-20 | C/T/S rc=0, ASR exact; byte-reproducible under NBX_FORCE_RAND_SEED. |
| llm/TinyLlama-1.1B-Chat-v1.0 | 2.1 GB | **DELIVERED** 2026-09-20 | C/T/S rc=0, "Paris." exact, text byte-identical across C and T. |
| stt/whisper-large-v3-turbo | 1.6 GB local | **DELIVERED** 2026-09-20 | C/T/S rc=0, JFK transcript exact. NOTE: the hub's stt/ directory for it is EMPTY — the local copy is the only reachable one; flagged, not evicted. |
| upscaler/swin2SR-classical-sr-x2-64 | 58 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged non-degenerate (std ~104). |
| upscaler/swin2SR-classical-sr-x4-64 | 58 MB | **DELIVERED** 2026-09-20 | same battery, same judge. |
| upscaler/swin2SR-realworld-sr-x4-64-bsrgan-psnr | 58 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| upscaler/real-esrgan-x2 | 67 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| upscaler/real-esrgan-x4 | 67 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| upscaler/real-esrgan-x8 | 67 MB | **DELIVERED at ≤512 px** 2026-09-20 | C/T/S rc=0 at 448/512 px, judged. 1024 px OWED on the per-tile retention defect (handed to the runtime owners); refused with figures until it returns. |
| upscaler/swinir-classical-x2 | 99 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| upscaler/swinir-classical-x4 | 92 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| stt/parakeet-tdt-1.1b | 4.2 GB | **DELIVERED** 2026-09-21 | C/T/S all rc=0 (7 s / 22 s / 15 s), JFK transcript exact in each mode (STT judge), 0 misses both triton modes. Extract from hub 350 s (~12 MB/s sequential). Local copy deleted after judgment. |
| tts/openaudio-s1-mini | 4.1 GB | **DELIVERED** 2026-09-21 | C/T/S all rc=0 (12 s / 367 s / 426 s), ASR reads the sentence exactly in each mode, 0 misses both triton modes. Local copy deleted after judgment. |
| audio_llm/canary-qwen-2.5b | 4.9 GB | **REFUSED — engine defect (Prism)** 2026-09-21 | Both modes tried rc=1 in ~2 s: `ZERO FALLBACK: No allocation for component 'perception_encoder'` — Prism plans no allocation for the perception encoder; mode-independent (planning-time). Re-verified on today's build; first seen 2026-09-19. Prism is the rack side's domain; the model fits this machine by arithmetic (4.9 GB bf16 vs 24 GB) and runs the day the plan allocates its encoder. Local copy deleted. |
| tts/VibeVoice-1.5B | 5.4 GB | **PARTIAL — 2 of 3 modes** 2026-09-21 | C rc=0 11 s and T rc=0 688 s, ASR exact both, 0 misses. S refuses in 4 s by DESIGN: `ZERO FALLBACK: decode branches need the KV cache path (no KV wrapper on this session)` — sequential mode builds no KV wrapper (deliberate O(n) fallback) and the next_token_diffusion flow needs decode branches. Not scaled down silently: the three-mode criterion is the owner's; this line says exactly which mode is short and why. CANDIDATE WALL CLASS: sequential × branch-decode flows — a second instance stops the loop. Local copy deleted. |
| audio_llm/Voxtral-Mini-3B-2507 | 9.4 GB | **PARTIAL — compiled only** 2026-09-21 | C rc=0 27 s, JFK transcript exact — after fixing a real defect this cycle exposed (audio_llm prefix/suffix embeds looked up on the table's device never joined the context device; "Passed CPU tensor to MPS op"). But the machine holds it at the edge: C's lowest was 4360 MB of the 4096 floor at a 14.5 GB ambient, an earlier attempt floor-stopped from a 13.0 GB ambient, and BOTH triton arms floor-stop at 4086/4084 MB — ten megabytes under the floor, 0 misses, after 118 s / 403 s of real work. The figure that refuses: the triton arms need ~10 MB more than the floor leaves at the ambient band's top (10.2-15.2 GB measured with the VM running). Refused for T/S under standing conditions rather than retried onto a lucky ambient. Local copy deleted. |
| upscaler/hat-s-x4 | 51 MB | **REFUSED** (standing) | Blocked by the Prism estimate; stays refused rather than closed by invented demand. |
| upscaler/hat-l-x4 | 182 MB | **REFUSED** (standing) | Floor stop in both triton modes at 1567 MB available with 0 misses after its 8 shapes were certified — the machine genuinely cannot hold it. |

## Anomalies

- Nine hub entries are empty or unreadable directories (no model.nbx, du=0):
  image/Sana_1600M_1024px_MultiLing(+_diffusers), llm/deepseek-moe-16b-chat,
  multimodal/Ming-Lite-Omni-1.5, multimodal/MiniCPM-o-4_5,
  multimodal/Qwen3-Omni-30B-A3B-Instruct, stt/whisper-large-v3-turbo,
  video/Allegro-TI2V, video/SANA-Video_2B_720p_diffusers. A container that
  is not there can be neither run nor refused by arithmetic; these are
  listed as absent, not skipped.

## Link measurement

Hub-mount transfer measured 2026-09-21: 2663 MB in 340 s = **7 MB/s**
(owner's own figure 8.9 MB/s; no cable yet). Re-planned against 7-9 MB/s.
