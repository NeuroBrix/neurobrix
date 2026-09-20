# Audio is the family where a certified change is visible in the artefact

## The property
**Re-certifying a kernel can change an audio artefact's bytes while leaving
every image and text artefact untouched.** This is expected, it is not a
regression, and it means audio must be judged by a third-party recogniser rather
than by digest.

## The measurement that established it
2026-09-18, after certifying 34 shapes that a key change had made missing. Every
proven cell re-run in BOTH Triton modes, replay cache cleared, same seed
(`NBX_FORCE_RAND_SEED=1234`), zero autotune misses:

    swin2SR x2 / x4 / realworld-x4      digest unchanged
    swinir x2 / x4                      digest unchanged
    real-esrgan x2 / x4 / x8            digest unchanged
    whisper-large-v3-turbo              digest unchanged
    parakeet-tdt-1.1b                   digest unchanged
    TinyLlama-1.1B-Chat[-v1.0]          digest unchanged
    Kokoro-82M                          30168a29dc3e -> e6c1e4f488b8   CHANGED

Both Triton modes agreed with each other in every case, before and after.

## Why audio and not the rest
A certification picks a tile configuration. A different tile is a different
order of floating-point accumulation, so the low bits of a result move. What
happens next differs by family:

* **Text** — a greedy argmax over a vocabulary. A perturbation in the low bits
  almost never changes which token is largest, so the artefact is identical.
* **Images** — values are quantised to 8-bit channels on the way out. A
  perturbation far below 1/255 disappears in the rounding.
* **Audio** — 16-bit PCM samples ARE the values, written out at full precision
  with no argmax and no quantisation to hide behind. Every sample carries the
  perturbation into the file.

So audio is not more fragile; it is the only family that does not have a
discretising step between the arithmetic and the artefact.

## What follows for judging
**Kokoro and chatterbox are judged by a third-party ASR, never by digest.** The
ASR reads the sentence back — "The quick brown fox jumps over the lazy dog" —
which is the R29-hardened test: an instrument OUTSIDE the engine judging the
artefact's CONTENT. A digest comparison between two arms of our own stack is
MEASURED at best, and for audio it is not even stable across a certification.

This is what the zoo's `audio_gate` was already built for: SNR first, then a
log-mel distance, then a transcript comparison, "the same words = the same
speech" (`tools/precision_zoo_campaign.py`). Two unseeded Kokoro runs score
19.36 dB SNR and 0.101 mel distance — both past their bars — and pass on
WER 0.0.

## The separate, unrelated fact about Kokoro
Kokoro is also a stochastic model: unseeded, it draws noise and no two runs
match at all (77.6% of samples differ, max 16.9% of full scale). That is a
different thing from the property above and is closed in the ledger as NOT a
defect — `NBX_FORCE_RAND_SEED` makes it reproducible on demand.

The two together are why audio gets a recogniser and not a checksum.
