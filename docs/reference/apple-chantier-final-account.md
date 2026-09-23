# The Apple / Metal chantier — final account

**2026-09-22 into 2026-09-23. Merged to main at `df8a4842`, both remotes.**

## What was asked, and where it stands

| | |
|---|---|
| census pass B finished, every open model named with its cause | **done** — 33/33, zero unclassified |
| all censused keys certified on Metal against the fp64 oracle | **3 178 of 3 179** — the one exception named below |
| checkpointer holding the repo, pushed to both remotes | **done**, throughout |
| served directory switched | **done** — 956 pruned, gate `rc=0`, 17 files, 0 refused |
| verification judged at zero miss, each shown with its command output | **done** — 0 misses across 30 cells, 29 artefacts |

## The census

Re-censused with request-dependent dimensions BUCKETED in the autotune key, so a prompt one
token longer no longer meets a key nobody certified. **3 179 keys of 59 models**, of which 73
were found only because verification could finally run: the census shadow reads graphs without
executing and under-predicts what a real run forms.

**33 models are open**, each named with cause and owner in `catalogue-state.md`: 11 memory
limits of an 18 GB card, 6 the rack's layer_streaming, 6 trace or op defects, 3 censused
without the image their flow requires, 4 owing a re-trace for a frozen symbol, 2 a missing
capability, 1 a Metal driver fix.

## Engine defects fixed, each red then green

| | |
|---|---|
| `a054ef6c` | the between-key pool drain resolved to `None` and never ran — on BOTH machines, for every certification either had ever run |
| `393570c6` | `depthwise_conv2d` returned wrong numbers in bf16 with padding; the stencil multiplied in the operands' own dtype for every type except fp16 |
| `17c96d16` | `__main__` discarded `main()`'s return, so no command could report failure by its exit status |
| `bc541cb9` | the census recorded `" ".join(argv)`; a multi-word `--prompt` could not be replayed, and three verification cells had never run |
| `263fbb4a` | `mm`/`addmm` band above 2^31 output elements, fixing a silent wrong answer (rows past `2**31 // N` came back zeros): deviation **1.0 -> 5.39e-07** |
| `6dabcb36` | a shape over the screening budget is screened on named ROW WINDOWS instead of seated unverified |
| `8485d9fb` | the divergence refusal divided by a tolerance that may legitimately be zero |

## The one key not certified

`addmm M_BUCKET=4194304 N=540 K=180`. **It is no longer a wrong answer** — banding fixed the
engine. What remains is that the certifier validates one band against an oracle built for the
whole output (register 504), and that the census demands a launch shape the engine no longer
makes (filed as an engine defect for the rack: the census is blind to what the wrapper does at
launch, the same class as mochi's).

**Four avenues tried, each recorded with what it proved** — three kernel spellings, banding,
keying each band on its own rows (reverted), re-censusing the model that demands it. **No model
output is affected**: `swinir-classical-x2` does not form that shape on its real path and
verifies clean with 0 misses.

## What the session was actually about

Six instruments were answering a different question than the one asked, and **every one
returned a clean-looking answer** — the direction that does not invite a second look:

| instrument | measured | read as |
|---|---|---|
| the 2^31 gate's memory probe | whether `libcudart.so` exists | "the card is busy" — it skipped on Metal while the defect it guards was live |
| `neurobrix`'s exit status | nothing; the return was discarded | "certification succeeded" — the campaign announced DONE at 3 057 of 3 106 |
| the verification miss counter | whether a key sat in a warm cache | "zero miss against the directory" — 8 misses became 0 with nothing certified |
| the census `command` field | a space-joined argv | "replayable" — 45 of 59 models were not |
| `ps` RSS on Apple | non-Metal allocations only | "healthy at 0.9 GB" — the real footprint was 21.9 GB |
| the stability witness | the regime held within 8 % | "the machine was quiet" — it cost 313 quarantined entries |

And the directory's own claim did not survive measurement: **88.1 % of Apple rankings and
33.2 % of CUDA ones were decided inside their host's own spread**, so "certified" now asserts
**correct and pinned**, never fastest.

**The same failure, committed by me, an hour after cataloguing it.** The windowed screen hid
three of its own defects behind an `except Exception: kept = None` I wrote myself — the exact
pattern removed from the pool drain that morning. Making it say *why* is what found the other
two.

## Owed by the rack, named

1. The 2^31 **Metal lowering** — ours is worked around by banding; the lowering is triton-ext's.
2. The **census is blind to what the wrapper does at launch** — census and Prism are theirs.
3. `test_the_boundary_does_not_widen.py` RED — accepted as this machine's, queued.
4. Whether the **windowed screen** should replace the budget on CUDA, where not one of 12 871
   entries records whether it was screened at all.

## Still open here, named

- **15 pre-existing kernel-test failures** on this branch, verified identical before my work.
- The **screening cost**: the same selection runs 9.98 s before the windowed screen and 313 s
  after. Bounded by the window, not the shape; the lever is `_SCREEN_WINDOWS`, not a return to
  seating large shapes unverified.
