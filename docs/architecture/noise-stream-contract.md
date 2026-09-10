# Noise stream contract (Triton branch)

The Triton branch draws every stochastic tensor — a diffusion run's initial
latent, ancestral-sampler noise, DDIM `eta > 0` variance — from NeuroBrix's own
kernel, not from a vendor library. This page records what that stream
guarantees, and on what evidence.

It is an **engine fact**, established once and re-established only when the
stream changes. It is not a campaign row and not a per-release gate.

## What the stream is

Three pieces compose, and each is ours:

| piece | where | what it does |
|---|---|---|
| run stream | `kernels/rng_stream.py` | one seeded stream per run, armed from `defaults.seed`; per-draw kernel seed = `splitmix64(run_seed, counter)`, truncated to 31 bits |
| kernel | `kernels/ops/rand_op.py::randn_kernel` | `tl.randn(seed, offset)`, Triton's Philox, one draw per element |
| offset mapping | same file | every element takes its global index as the Philox offset |

Only the middle row is Triton's. The seed derivation, the truncation and the
offset mapping are NeuroBrix's, and each can be wrong while `tl.randn` is
perfect.

## Why this needs its own proof

The vendor-correctness cell gates a diffusion family by **pinning the initial
latent**: our engine renders, dumps its starting noise, and the vendor denoises
that exact tensor. That is the right primary gate, because it turns PSNR back
into a real bound — but it has a blind spot no campaign row can reach. A pinned
latent is *our* latent handed to *them*, so the stream that produced it is never
itself under test. Pin the latent everywhere and the noise stream is proved by
nothing, forever.

There is also no pixel-level reference to be equal to: the compiled engine uses
`torch.Generator`, a different algorithm, and cross-engine bit-equality is
explicitly **not** the contract (`kernels/rng_stream.py`). So the proof must be
statistical.

## What is proven

`tools/noise_stream_validation.py`, 2026-09-09, V100 sm_70, n = 4 194 304
elements per draw, run seed 42. Every bound is **4 standard errors** of that
statistic's own sampling distribution under the null "n i.i.d. draws from
N(0,1)", written in the tool before any draw. Four sigma rather than three
because a run tests twenty-one statistics at once.

| property | statistic | measured | bound | verdict |
|---|---|---:|---:|---|
| no torch in the noise path (R33) | `torch in sys.modules` | False | False | PASS |
| centred | mean | +1.90e-04 | 1.95e-03 | PASS |
| unit variance | var − 1 | −6.53e-04 | 2.76e-03 | PASS |
| symmetric | skewness | +2.34e-03 | 4.78e-03 | PASS |
| normal-tailed | excess kurtosis | +1.13e-03 | 9.57e-03 | PASS |
| normal (body) | KS vs N(0,1) | 2.79e-04 | 7.96e-04 | PASS |
| normal (tails) | Anderson-Darling | 0.475 | 1.091 | PASS |
| independent within a draw | autocorrelation, lags 1–8 | max 8.05e-04 | 1.95e-03 | PASS ×8 |
| independent across draws | corr(draw i, draw i+1), i = 0..2 | max 1.21e-03 | 1.95e-03 | PASS ×3 |
| independent first vs last | corr(draw 0, draw 3) | 7.07e-05 | 1.95e-03 | PASS |
| neighbouring seeds decorrelated | corr(seed 42, seed 43) | 6.02e-04 | 1.95e-03 | PASS |

**21/21 within bound.** Report: `validation_outputs/noise_stream_2026_09_09/report.json`.

Re-run with:

```bash
PYTHONPATH=<engine src> python3 tools/noise_stream_validation.py \
    --out validation_outputs/noise_stream_<date>/report.json
```

## The one limitation, quantified

`rng_stream.next_seed()` truncates to 31 bits because `tl.randn` takes an
**int32** seed — the narrowing is forced by the kernel API, not chosen.

Measured over a run of 1 000 000 draws: **242 seed collisions**, against a
birthday expectation of 232.8 for a 31-bit space. The truncation therefore
behaves as a random map — the mixing is sound, and what remains is a property of
the space's size, not a defect in the derivation.

The consequence is bounded: two draws that collide on the seed produce
bit-identical noise **only if they also cover the same offset range**, i.e. have
the same element count — which successive latents of one render do. The
probability that a run of N stochastic draws contains such a repeat is
approximately `N² / 2³²`:

| draws in a run | P(a repeat) |
|---:|---:|
| 100 | 2.3e-06 |
| 1 000 | 2.3e-04 |
| 10 000 | 2.3e-02 |
| 46 000 | ~0.5 |

A 20-step render makes tens of draws, so today's exposure is ~1e-7 and no
current model approaches the region where this matters.

**If it ever does**, the fix costs nothing and does not need a wider seed: vary
the Philox *offset base* per draw (`offset = counter · n + i` instead of `i`)
so that colliding seeds still address disjoint counter space. That change moves
every seeded output, so it is a deliberate, gated act — recorded as
`D-RNG-31-BIT-SEED-SPACE` in the debt registry, not taken in passing.

## What this contract does not say

* It says nothing about the **compiled** branch, which draws from
  `torch.Generator` and is the vendor ecosystem's own stream.
* It does not make the two engines bit-equal. When a bit-identical cross-engine
  noise diff is needed, `kernels/rng_pin.py` (`NBX_FORCE_RAND_SEED`) remains the
  tool and takes precedence in the wrappers.
* It is a statement about the *distribution*, not about any single run's values.
