# S5 — VLM / Multimodal / Omni segment (2026-08-31)

Protocol: locked machine, sequential cells, daemon stopped before campaign,
one pinned GPU per cell, n as stated, cross-engine (compiled + triton) per row.
Gates: VQA = must-phrase containment on the answer; t2i = decodable PNG +
mean band [80,220] + std>5. Artifacts under
`validation_outputs/hub_benchmark_2026_08/s5_*/` (reports, transcripts, PNGs).

## Results (median wall s/req, [min–max], gate)

| Model | Mode | nbx compiled | nbx triton | Gate c/t |
|---|---|---|---|---|
| MiniCPM-o-4.5 | VQA n=5 | 41.43 [40.92–41.89] | 85.63 [85.35–88.19] | PASS/PASS |
| GLM-4.1V-9B-Thinking | VQA n=5 | 54.61 [54.44–54.87] | 129.97 [129.05–130.26] | PASS/PASS |
| Qwen3-VL-30B-A3B-Thinking | VQA n=5 | 501.15 [491.30–512.33] | 1135.54 [1114.65–1138.35] | PASS/PASS |
| Janus-Pro-7B | t2i n=3 | 205.10 [204.67–205.48] | **106.97** [105.30–109.80] | PASS/PASS |
| Ming-Lite-Omni-1.5 | t2i n=3 | 135.87 [134.98–155.32] | 348.89 [346.85–350.36] | PASS-R29 ×2 |
| Qwen3-Omni-30B | speech-to-speech n=3 | 467.21 [450.58–471.01] | 888.78 [878.16–897.45] | PASS/PASS |

## Readings (honest)

- **Janus t2i: triton is 1.9× FASTER than compiled** (106.97 vs 205.10 s) —
  the autoregressive image path is where the SIMT decode family pays off;
  first row where mode 2 beats mode 1 outright.
- **VQA rows: triton 2.1–2.4× slower than compiled** — the vision-prefill
  gap (large-M matmuls) named by the prefill chantier; Qwen3-VL-30B is the
  headline number (501 vs 1136 s).
- **Ming gate-band false-red**: both engines render a photorealistic apple
  (R29 eyeballed); image mean 79.1/78.9 vs anti-black band lower bound 80.
  Cross-engine coherent (std 54.7/55.6). Gate verdict overridden by R29
  inspection; band guard kept (it exists to catch black frames, and a
  1-point near-miss on a dark scene is its known cost).
- **Qwen3-Omni is a full speech-to-speech row**: 11 s spoken question in,
  spoken answer out (thinker 27 tokens + talker 94 codec frames + 7.5 s
  24 kHz waveform). Fidelity gate reads the pipeline's own transcription
  line (contains the JFK phrase, both engines); WAVs kept as R29
  artifacts (rms 0.048/0.040, non-silent). The two first-pass failures
  were HARNESS-side (`--mode audio` omitted, then `.txt` output where
  the speech contract requires `.wav` — both ZERO FALLBACK guards doing
  their job); the runs themselves never failed.

## Capacity axis

All 5 rows above run at HEAD from the hub containers in both engines.
Competitor arms for these models (vLLM/ollama VQA, diffusers t2i for
Janus/Ming) are DNR on this rig: no vLLM/ollama support for these
checkpoints' multimodal graphs at campaign date (sourced in S4/S5 prereg).
