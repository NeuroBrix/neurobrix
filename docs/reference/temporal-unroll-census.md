# Which video components unroll a temporal loop? — 2026-09-12

## The question, and why it had to be answered before anything was said outside

A causal video VAE processes frames in chunks. If the tracer UNROLLS that loop,
the graph holds one copy of the chunk body per chunk and the chunk count is fixed
at the trace stimulus. Such a component is not symbolic in time however many
symbols its table declares: a graph unrolled for five chunks cannot process
eighty-one frames, and the symbolic machinery then folds the temporal factor into
a neighbouring slot rather than failing honestly. That is how three Wan
containers came to refuse `(1, 6528, 1, 112, 112)` against `(1, 384, 1, 112, 112)`.

If it is the video family, it is a limit of our support and has to be known
before anyone describes that support. If it is a few models, it is a named debt.

**It is a named debt: three models, all of them Wan, and only their VAE encoder.**

## The instrument, and the one it replaced

There is no sound one-point test. The first attempt scored each graph by whether
its module repetition counts `r` were explained by the chunk count `k`
(`r % k == 0 or r % k == k-1`) and reported seven components at a perfect 1.00 —
every one of them at `k = 2`, where that predicate accepts every integer. The one
component with a fitted slope scored 0.79 and fell below the threshold. Register
entry 38.

The signature read on two fitted points is far sharper and needs both of them.
Between k=5 and k=7 the twelve module groups of the measured case move as
`27k-40`, `19k-26`, `16k-17`, `13k`, `13k-12`, `7k`, `4k`, `3k-1`, `2k`, `2k-2`,
`k`, and one constant — **eleven of twelve affine in k**. So the instrument
traces the same component at two stimuli with the SAME tracer (a fit across two
tracer versions is not a fit) and reads the slope:

```
ops(k) = a*k + b        a >= 2 and a*k/ops >= 0.20  ->  UNROLLED
```

The threshold is not `a > 0`: Allegro's decoder gains exactly one op between k=3
and k=7, a slope of 0.25 that `a > 0` reads as a loop body.

Positive and negative control sit on the same model: `Wan2.1-VACE`'s encoder at
517 ops per chunk, its decoder flat at 384.

## Measured — 11 components, 6 models, two stimuli each

| component | ops @k=3 | ops @k=7 | ops/chunk | verdict |
|---|---:|---:|---:|---|
| `Wan2.1-VACE-1.3B-diffusers/vae_encoder` | 1237 | 3305 | **517.00** | **UNROLLED** |
| `Allegro-TI2V/vae_encoder` | 761 | 761 | 0.00 | flat |
| `CogVideoX-5b-I2V/vae_encoder` | 265 | 265 | 0.00 | flat |
| `Allegro/vae` | 1008 | 1009 | 0.25 | flat |
| `Allegro-TI2V/vae` | 1008 | 1009 | 0.25 | flat |
| `CogVideoX-2b/vae` | 929 | 929 | 0.00 | flat |
| `CogVideoX-5b-I2V/vae` | 929 | 929 | 0.00 | flat |
| `Wan2.1-T2V-1.3B-Diffusers/vae` | 384 | 384 | 0.00 | flat |
| `Wan2.1-VACE-1.3B-diffusers/vae` | 384 | 384 | 0.00 | flat |

**One of nine measured components unrolls.** Every decoder tested is flat, and so
are the two non-Wan encoders — the asymmetry that started this investigation is
real for the Wan VAE and is NOT a property of video components in general.

## Inferred — 2 components, and what the inference rests on

`Wan2.1-I2V-14B-480P-Diffusers` and `Wan2.2-I2V-A14B-Diffusers` have no local
snapshot (purged under R38), so no second trace point exists for them. Their
cached encoder graphs carry the measured anchor's chunk-loop module groups
**identically**, on both the repetition counts and the number of modules at each:

| component | chunk-loop groups at k=3 | identical to the anchor |
|---|---|---|
| `Wan2.1-VACE/vae_encoder` (**measured anchor**) | `{2:1, 3:9, 6:12, 8:22, 12:1, 21:22}` | — |
| `Wan2.1-I2V-14B-480P/vae_encoder` | `{2:1, 3:9, 6:12, 8:22, 12:1, 21:22}` | yes |
| `Wan2.2-I2V-A14B/vae_encoder` | `{2:1, 3:9, 6:12, 8:22, 12:1, 21:22}` | yes |

Those counts are `7k, 4k, 3k-1, 2k, k` and one constant at k=3 — the anchor's own
loop. **The claim is narrow on purpose**: it is not that the graphs look alike.
Their groups ABOVE the loop differ (`41/39/31/27` against `67/50/39/37`), because
these are VAEs of different sizes carrying the same chunk loop. Six groups
agreeing exactly on two numbers each is what the inference rests on.

Converting either to a measured line costs one snapshot re-download and two
traces of about fifteen seconds. Until then the line reads INFERRED, and an
inferred line and a measured line do not read the same.

## Not measurable, with the reason

| component | why |
|---|---|
| `Allegro/vae_encoder`, `CogVideoX-2b/vae_encoder`, `Wan2.1-T2V/vae_encoder` | the model has no such component |
| `SANA-Video_2B_720p/vae` | at T=25 the synthesised latent carries 64 channels where the weights want 128 (`expected input[1, 64, 27, 16, 24] to have 128 channels`). The latent WIDTH depends on the temporal stimulus, which is its own question and not this one. Reproducible; T=9 traces fine. |
| `Open-Sora-v2`, `mochi-1-preview` | snapshots purged; no second point available |

## What it means

Three models carry the limit, and the third is inferred rather than measured. The
debt is `D-TEMPORAL-UNROLL` in `DETTE.md`; the two remedies are there and neither
is a number. What this census changes is the scope sentence: **it is not the
video family**, and it is not a property of encoders in general — it is the Wan
VAE encoder, whose chunk loop the tracer unrolls.

Instruments: `tools/temporal_unroll_census.py` (static survey, and it refuses to
give a verdict from one point), `nbx/campaigns/prepared/unroll_measure.py` (the
two-stimulus fit), `tools/unroll_census_report.py` (renders this table from the
measurement file, so a number cannot be retyped wrong).

---

## Companion: the weight-extent collisions, 2026-09-12

The same catalogue carries a second blindness of the same shape. 33 axes are
traced at a value that is ALSO a parameter extent of their own component, where a
dim bound to the wrong quantity reproduces the trace exactly — the collision at 1
with a different coincidence.

**The instrument is a second trace at a value the weights do not carry**, and the
first version of it compared a tree with itself and printed a verdict. Asked for
`Flex.1-alpha/text_encoder` at 71 instead of 77, it got a graph still recording
`trace_value=77`, compared 466 identical dims with themselves, and concluded *"the
axis is BOUND to the symbol"*. It now REFUSES when the recorded value has not
moved.

**And the refusal is a better answer than the verdict it replaced.** When the
stimulus cannot move the axis, the model FIXES it — a position table, a fixed
tokenizer length, an architectural hidden dimension — so the axis cannot vary at
runtime either, and its equality with a parameter extent is structural rather
than a blind spot. Nothing to act on, and nothing claimed about a binding.

| component | axis | at | asked | result |
|---|---|---:|---:|---|
| `Flex.1-alpha/text_encoder` | `seq_len` | 77 | 71 | **fixed by the model** — value did not move |
| `Flex.1-alpha/transformer` | `seq_len` | 4096 | 4091 | **fixed by the model** — value did not move |
| `Ming-Lite-Omni-1.5/image_vae` | `seq_len` | 3 | 7 | not measured — a single-component trace does not reach this component on a multimodal container |
| `MiniCPM-o-4_5/flow_dit` | `seq_len` | 3 | 7 | not measured — same |

The instrument (`nbx/campaigns/prepared/weight_extent_differential.py`) and the
stimulus override it needed (`forge trace --trace-seq-len`) are the deliverable
here as much as the rows: before today there was no way to move a sequence axis
off a colliding value at all, so none of these 33 could be adjudicated by
anything.

**The spatial axes are now reachable, and the note that said otherwise was
wrong.** 18 of the 33 are `height`/`width` at 128, 64 or 32 — the VAE family. An
earlier version of this paragraph said the spatial stimulus is written at three
sites "rather than read from one" and that consolidating them came first. Reading
the three sites showed they are not copies: `(10,12)`, `(18,26)` and `(14,22)`
each carry their own de-collision analysis for their own component class — the
RoPE grid's token axes where T and H had collapsed into one symbol, a T-derived
product chain, a set of subset products that must miss every common channel
count. Each was paid for by a frozen graph, and one value would have deleted
three pieces of reasoning.

What was missing was not consolidation but an OVERRIDE reaching all three:
`forge trace --trace-spatial H,W`, the same shape as `--trace-time` and
`--trace-seq-len`. It refuses equal extents, because equal dims give `s_h == s_w`
at trace and the runtime freezes both axes when two symbols share a value — which
is why no square stimulus appears in any of the three sites either.
