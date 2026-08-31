# S2 — Long-form STT rows (600 s asset, 2026-08-31)

Asset: 600 s concatenated LibriVox chapter (versioned sha in prereg).
Design: sha-only nocturnal cells, n=3 per arm, RTFx = audio_s / wall_s.
Prereg clause honored: "sha-determinism as record, adjudication out of
segment" — and adjudication was needed:

## FINDING — D-STT-LONGFORM-CHUNKING (the naive long-form RTFx table is invalid)

Transcript byte sizes prove the stacks do DIFFERENT WORK on a 600 s input:

| Arm | turbo transcript bytes | Interpretation |
|---|---|---|
| nbx (c/t) | 105 | single 30 s mel window — **truncates**, no chunking loop |
| transformers | ~1.1k | vendor default chunks partially (30 s window, no long-form algo in this config) |
| faster-whisper | 8971 | **full 600 s transcribed** (built-in VAD/chunk loop) |
| nemo (parakeet) | full | full-file streaming transcription |

Therefore RTFx values are NOT comparable across arms (nbx "172×" is the
RTFx of 1/20th of the work). The only valid long-form throughput cells:

| Row | Arm | RTFx median [min–max] n=3 |
|---|---|---|
| whisper-v3-turbo 600 s | faster-whisper | 68.11 [67.70–68.45] |
| whisper-large 600 s | faster-whisper | 22.49 [22.40–22.59] |
| parakeet 600 s | nemo | 68.52 [66.37–69.57] |
| parakeet 600 s | **nbx compiled** | **124.48** [111.73–130.72] — valid: RNNT path streams the full file (sha matches nemo-arm content class, full transcript) |
| parakeet 600 s | nbx triton | 39.66 [39.53–40.24] — valid, full transcript |

sha-determinism record: every arm byte-identical across its 3 reps
(single sha per arm), including nbx cells — determinism holds at 600 s.

## Named debt

**D-STT-LONGFORM-CHUNKING**: the encoder-decoder (whisper-class) flow
lacks a long-form chunking loop (30 s windows + merge). Until it ships,
whisper-class rows are SHORT-FORM ONLY in public tables; parakeet (rnnt)
is the long-form-capable NeuroBrix STT row today — and it BEATS nemo
124.5× vs 68.5× RTFx on the same 600 s asset.

The 30 s truncation also retro-explains the May "whisper long audio"
anomaly class. Chantier direction: chunk in the stt flow handler
(window + overlap-merge at the flow level, R30 both engines), never in
the mel DSP.
