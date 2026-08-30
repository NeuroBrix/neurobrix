# S2 STT lot-1 annex — three rows, all arms, locked protocol (2026-08-30)

Protocol: GPU 2 locked 1290, machine exclusive, interleaved arms, n=5
after warmup, accuracy gate ACTIVE on every cell (jfk: full-text
word gate, mutation-proved in situ at 01:00 UTC — "martians" turned
the campaign RED, report in s2_mutation_proof/; ref:
phrase-containment, per the 01:35 UTC dated amendment). Every cell
derives from its report.json. RTFx = clip_s / transcribe_s, HIGHER
is better. One transcript sha per arm per cell = full determinism.

Arms: nbxt = NeuroBrix triton; nbxc = NeuroBrix compiled;
tf = vendor transformers fp16 greedy (the traced-from stack);
fw = faster-whisper 1.2.1 / CT2 4.8.1 fp16 (R16-retained);
nemo = NeMo 2.7.3 on torch 2.5.1+cu121 (PIN DEVIATION, annotated:
the R16 pin said torch 2.7+cu126; driver 535 = CUDA 12.2 refuses
cu126+ wheels at runtime; 2.5.1+cu121 runs EMPIRICALLY — the pip
metadata constraint torch>=2.6 is unsatisfied-but-working, and the
smoke transcribed the canonical text before any cell).

| row / clip | arm | n | median RTFx | min-max | shas | gate |
| parakeet_jfk | nbxt | 5 | 0.981 | 0.954-0.996 | 1 | PASS |
| parakeet_jfk | nbxc | 5 | 2.461 | 2.335-2.535 | 1 | PASS |
| parakeet_jfk | nemo | 5 | 12.515 | 11.316-13.173 | 1 | PASS |
| parakeet_ref | nbxt | 5 | 1.101 | 1.083-1.112 | 1 | PASS |
| parakeet_ref | nbxc | 5 | 2.976 | 2.858-3.184 | 1 | PASS |
| parakeet_ref | nemo | 5 | 14.823 | 12.692-17.802 | 1 | PASS |
| whisper-large-v3-turbo_jfk | nbxt | 5 | 1.682 | 1.667-1.727 | 1 | PASS |
| whisper-large-v3-turbo_jfk | nbxc | 5 | 3.448 | 3.343-3.481 | 1 | PASS |
| whisper-large-v3-turbo_jfk | tf | 5 | 11.814 | 10.566-12.153 | 1 | PASS |
| whisper-large-v3-turbo_jfk | fw | 5 | 22.790 | 20.485-28.250 | 1 | PASS |
| whisper-large-v3-turbo_ref | nbxt | 5 | 1.575 | 1.563-1.584 | 1 | PASS |
| whisper-large-v3-turbo_ref | nbxc | 5 | 3.867 | 3.835-3.991 | 1 | PASS |
| whisper-large-v3-turbo_ref | tf | 5 | 13.559 | 11.956-14.453 | 1 | PASS |
| whisper-large-v3-turbo_ref | fw | 5 | 29.438 | 23.109-31.702 | 1 | PASS |
| whisper-large_jfk | nbxt | 5 | 0.403 | 0.398-0.411 | 1 | PASS |
| whisper-large_jfk | nbxc | 5 | 1.613 | 1.536-1.649 | 1 | PASS |
| whisper-large_jfk | tf | 5 | 8.876 | 7.691-10.222 | 1 | PASS |
| whisper-large_jfk | fw | 5 | 15.009 | 14.788-16.459 | 1 | PASS |
| whisper-large_ref | nbxt | 5 | 0.304 | 0.297-0.305 | 1 | PASS |
| whisper-large_ref | nbxc | 5 | 1.400 | 1.380-1.429 | 1 | PASS |
| whisper-large_ref | tf | 5 | 8.417 | 7.908-8.683 | 1 | PASS |
| whisper-large_ref | fw | 5 | 14.341 | 11.803-14.445 | 1 | PASS |

Time slots (UTC, from report.json campaign blocks): turbo 01:41-01:45,
large-v2 01:45-01:53 (rerun2, flightrec 20260830_012534), parakeet
01:49-01:56 (flightrec 20260830_014908).

## Honest reading

* The dedicated inference stacks LEAD, and the gap is written down:
  faster-whisper (CTranslate2, fused fp16 inference engine) posts
  14-29x realtime; vendor transformers 8-14x; NeuroBrix compiled
  1.4-3.9x; NeuroBrix triton 0.30-1.7x. The encoder-decoder STT path
  has had none of the decode-band optimization the LLM rows got —
  these numbers are the "before" of that work, recorded per the
  reference-dimension rule.
* Parakeet: the artifact whose family fix landed tonight benches
  end-to-end for the first time. NeMo vendor leads; the nbx arms
  carry the rnnt flow as closed at the audio closure.
* Accuracy: every arm of every cell transcribes the canonical texts
  correctly (gate PASS x 24 cells); every arm is bit-deterministic
  across its 5 reps (1 sha each).

