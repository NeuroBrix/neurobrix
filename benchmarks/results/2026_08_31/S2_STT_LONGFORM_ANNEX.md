# S2 — Long-form STT rows (600 s asset, 2026-08-31)

Asset: 600 s concatenated LibriVox chapter (versioned sha in prereg).
Design: sha-only nocturnal cells, n=3 per arm, RTFx = audio_s / wall_s.
Prereg clause honored: "sha-determinism as record, adjudication out of
segment" — and adjudication was needed:

## FINDING — D-STT-LONGFORM-CHUNKING (the naive long-form RTFx table is invalid)

Transcript byte sizes prove the stacks do DIFFERENT WORK on a 600 s input:

| Arm | transcript bytes (turbo / parakeet rows) | Interpretation |
|---|---|---|
| nbx whisper (c/t) | 105 | single 30 s mel window — **truncates**, no chunking loop |
| nbx parakeet (c/t) | 372 | **also truncates** — the rnnt path is bounded by its trace input window |
| transformers (large row) | 16 | near-empty output at this config — invalid cell |
| faster-whisper | 8971 | **full 600 s transcribed** (built-in VAD/chunk loop) |
| nemo (parakeet) | 9046 | **full 600 s transcribed** (streaming loop) |

Therefore RTFx values are NOT comparable across arms — a truncating arm's
"RTFx 172×" is the RTFx of 1/20th of the work. The ONLY valid long-form
throughput cells from this campaign:

| Row | Arm | RTFx median [min–max] n=3 |
|---|---|---|
| whisper-v3-turbo 600 s | faster-whisper | 68.11 [67.70–68.45] |
| whisper-large 600 s | faster-whisper | 22.49 [22.40–22.59] |
| parakeet 600 s | nemo | 68.52 [66.37–69.57] |

**NeuroBrix has ZERO valid long-form cells today** — both the whisper
(encoder_decoder) and parakeet (rnnt) flows transcribe only their trace
window on a 600 s input. This is stated plainly; no nbx long-form number
is published.

sha-determinism record: every arm byte-identical across its 3 reps
(single sha per arm), including the truncated nbx cells — determinism
holds at 600 s, on whatever work each arm performs.

## Named debt

**D-STT-LONGFORM-CHUNKING** (extended to BOTH STT families): neither the
encoder_decoder (whisper-class) nor the rnnt (parakeet) flow has a
long-form loop (window + overlap-merge / streaming). Until it ships, ALL
NeuroBrix STT rows are SHORT-FORM ONLY in public tables (the short rows —
11 s and 3 min assets, fully transcribed and accuracy-gated — remain the
closed reference). The truncation also retro-explains the May "whisper
long audio" anomaly class. Chantier direction: chunk in the stt flow
handler (window + overlap-merge at the flow level, R30 both engines),
never in the mel DSP.
