# Catalogue repairs — containers fixed at the source and delivered

A repair is not a patch and not a diagnosis. It is one motion, and every step of
it is recorded here because a re-trace without a re-upload is a forbidden state
(CLAUDE.md, hub freshness) and a container that only exists locally has not been
delivered.

**The five steps, and a repair is closed only when all five carry a date:**

1. **Cause** named at the source — Forge, never a runtime compensation.
2. **Re-trace** producing a new graph, with its op count.
3. **Rebuild**, with the container size beside the previous one.
4. **Re-upload**, reusing the exact slug so the record, its counters and its
   creation date survive.
5. **Proof by run** — the model producing output from the installed container.
   Not a reasoning about why it should. A verification that has not looked at the
   real artefact says nothing, and this project paid for that twice on the day
   this file was created.

A repair missing step 5 is listed as **delivered, unproven**, never as closed.

---

## 1 — `rhymes-ai/Allegro-TI2V` · 2026-09-12

**The first container of the catalogue repaired end to end and delivered.**

| step | evidence |
|---|---|
| cause | `RuntimeExecutor._container_output_size` refused every container whose flow was not `iterative_process`, and never consulted the VAE when the backbone's latent was flattened. Allegro-TI2V died on `Key 'latent_height' not found in runtime/defaults.json`. Fixed at the resolution engine, not in the model: a decoder's input extents ARE latent extents and the container states them. |
| re-trace | not required — the defect was in the runtime's reading of the container, not in the graph |
| rebuild | 2026-09-12 14:06, 26.1 GB written |
| re-upload | 2026-09-12 15:13:48, `forge replace` onto the existing slug; hub shows 24.3 GB, record preserved |
| install | 2026-09-12 15:14:41, `forge local --overwrite` |
| verification | `_container_output_size` reads **(144, 208)** from the shipped container — measured inside the installed form, not inferred from the diff |
| proof by run | **2026-09-12 16:00 — PASSED.** 8 frames at 144x208 from the installed container with the local archive deleted: `rc=0`, 30,493 bytes, and the video inspected rather than trusted — 8 frames, full 0-255 range, mean absolute inter-frame difference 17.3, first and last frames differ. It is a video, not a file of the right size. |

**Two walls stand beside the proof, and neither is container integrity.** They are
recorded because a repair that closes one error and hides the next is not closed:

1. **The request's image must match the resolution the container resolves.** A
   448x448 input against the resolved 144x208 fails with *"Expected size 18 but
   got size 56"* — the traced latent against the image's. The cascade derives the
   output size from the TRACE; when a request supplies an image, which of the two
   should win is a contract question the resolution engine does not currently
   answer. Before this repair the same request died earlier, on
   `Key 'latent_height' not found`, so this wall is newly reachable, not newly
   created.
2. **The container's own default of 88 frames does not fit this rack.** Both
   modes fail at the same site: compiled says *"CUDA out of memory. Tried to
   allocate 25.27 GiB"* at `aten.convolution::1` on a 31.74 GiB card; triton
   reaches the same place as a sticky `cudaErrorIllegalAddress` in
   `NBXTensor.from_numpy`, which is the first CHECKED call after the poisoning
   site and not the fault. One 25 GiB allocation in a 5D VAE is `DETTE.md` D2
   territory, not a defect of this repair.

Three attempts preceded the passing one and none of them said anything about the
container: pinned to a 16 GB card, pinned to `"2,3"` which remapped the ordinals,
and the container's own 88-frame default. Register entry 39.

The local archive was deleted after the hub copy was verified; the installed form
is a separate tree and survives. Both live on the NAS, not on the root
filesystem — a distinction that cost a wrong wall in a report the same day.

---

## What this file is for

The catalogue is 56 local containers and the hub is the distribution
destination. When a defect is found in one of them, the interesting question six
months later is not "was it diagnosed" — the register and the verdicts hold
that. It is **which containers people are actually being served**, and whether
the fix reached them. That is one table, and it has to be written at the moment
the upload lands or it is written from memory.

---

## 2 — `THUDM/CogVideoX-5b-I2V` · 2026-09-12

| step | evidence |
|---|---|
| cause | The I2V encoder's causal temporal pad recorded `2187·s − 2184` against a truth of `s + 2`, exact at the traced `s = 1` where every rule coincides. Seven resnet blocks compounded it; Prism asked 944 GB of activations for a component whose weights are 822 MB. Fixed at the source: a genuinely constant axis carries no symbol (an I2V encoder conditions on one image), Forge `96fc046`. |
| re-trace | 2026-09-12 15:22 and again 15:52 after an inert tracer edit — **same sha `7216b9fb1478`, bit for bit** (R27/R28). 389 → 265 ops, symbols `batch/height/width`, no temporal symbol. Profiled at 49 frames: 210.26 GB before, 0.02 GB after. |
| rebuild | 16:00, 21.5 GB (23 126 413 914 bytes). Two earlier builds were refused before publication: one without its backbone (5.45 GB, snapshot purged), one with half a text encoder (17.32 GB, one shard of two) — both now refused at entry by the index-aware build door. |
| re-upload | **22:09:39 UTC**, `forge replace` through the internal entry point `http://10.0.0.39:3000` (the public name was court-blocked), 2498 s at the 10 MB/s cap with zero `SlowDownWrite` after the host reboot. Hub record: `fileSize 23126413914`, `updatedAt 2026-09-12T22:09:39.879Z`. Three earlier uploads failed: 0 % (backbone-less), 5 min (half an encoder), 81 % (504 with no backoff), then a fourth stalled on a store whose drive MinIO had marked hung. |
| install | 22:10:26 UTC, `forge local --overwrite` — staged, verified against the archive, then repointed. The installed `vae_encoder/graph.json` holds 265 ops and no temporal symbol. |
| regression gate | passed component by component before the upload: text_encoder 1.000×, transformer 1.000×, vae 1.000×, vae_encoder 0.982× (the corrected graph). |
| proof by run | **2026-09-12 22:45 — PASSED.** 9 frames at 448×448 from the container installed at 22:10:26 (refused any earlier install): `rc=0`, range 9–253, mean absolute inter-frame difference 3.34 — a video, not a file. Run unpinned on the rack while the Wan2.2 upload continued at 30–37 MB/s with zero `SlowDownWrite`. |

---

## 3 — `Wan-AI/Wan2.2-I2V-A14B` · 2026-09-12 — TWO LINES, on purpose

**Line one, the repair.** The shipped `topology.json` (container dated 2026-07-03)
carried `shapes=NONE` for `transformer`, `transformer_2` and `vae`, so the output
resolution could never be read from it whatever the runtime did. The builder has
written component shapes since (Allegro-TI2V, built the same day, carries them),
so the fix is a rebuild — which needed the snapshot back (118 GB, re-downloaded
in two passes, the second resuming 99.56 GB).

| step | evidence |
|---|---|
| cause | shipped topology without component shapes; container predates the builder that writes them |
| re-trace | not required for this line — the graphs in `.cache/graphs` carry the shapes; the build reads them |
| rebuild | **22:11:30 → 22:22:46**, 676 s, **118.07 GB**, written directly on the pool as the only writer at ~400 MB/s (119 GB cannot stage on a 53 GB root filesystem — said before the build, not discovered at 90 %) |
| regression gate | **1.000×** on every component: text_encoder 10.59 GB, transformer 53.25, transformer_2 53.25, vae 0.47, vae_encoder 0.48 |
| re-upload | **22:22:46 → rc=0 after 5144 s** through the internal entry point (`replace`, same slug `Wan-AI/Wan2.2-I2V-A14B`): 126.77 GB, ≈24.6 MB/s mean under adaptive pacing from 40 MB/s (30–37 MB/s in the first hour, 26 in the last), **zero `SlowDownWrite`** in the per-minute samples. The hub record reads `updatedAt 2026-09-12T23:48:30Z`, `PUBLIC` |
| install | **23:48:30 → rc=0 after 312 s**: 100 files, 118.07 GB extracted, five components present in the cache (`text_encoder`, `transformer`, `transformer_2`, `vae`, `vae_encoder`), manifest written 23:53:42 |
| proof by run | owed — 9 frames, refuses a container installed before 21:28 UTC; it needs the whole rig (two 53 GB transformers) and runs in the night queue's phase B, after the four bench queues drain |

**Line two, the debt this repair does not touch.** Its VAE **encoder is unrolled
over the temporal axis** (`docs/reference/temporal-unroll-census.md`, INFERRED
from the measured anchor `Wan2.1-VACE`: identical chunk-loop module groups
`{2:1, 3:9, 6:12, 8:22, 12:1, 21:22}` at k=3). A rebuild does not change a
graph. The container will resolve its output size and remain incapable of an
encoder frame count it was not unrolled for — `DETTE.md`, `D-TEMPORAL-UNROLL`.
Delivering line one without line two would be delivering the fix for the error
we found and hiding the one underneath.

---

## 4 — `hpcai-tech/Open-Sora-v2` · 2026-09-12 — DIAGNOSED, re-trace queued

| step | evidence |
|---|---|
| cause | shipped `topology.json` (container dated 2026-06-30) carries `shapes=NONE` for `transformer` and `vae` while `.cache/graphs` holds them (`vae: z [1,16,9,14,22]`); the output resolution cannot be read from the container whatever the runtime does |
| snapshot | purged under R38; re-downloaded **16:38**, 64.43 GB in 862 s, build door satisfied |
| rebuild | **refused at entry, 22:10:27, in 5 s**: *"component 'scheduler' has no cached graph.json — topology/graph-cache desync"*. Its graph cache dates from 2026-06-30 like the container; the custom (non-diffusers) pipeline's synthetic topology lists a `scheduler` the cache never held. Nothing was written. |
| re-trace | **attempted 22:56:31, refused in Phase A**: *"custom pipeline requires model_index.json at …/Open-Sora-v2"*. The hub repo ships the vendor's own layout (four weight files at the root, `google/t5-v1_1-xxl`, `openai/clip-vit-large-patch14`) and no `model_index.json`; the June trace ran against a hand-made diffusers-shaped arrangement that the R38 re-download of 16:38 replaced with the vendor layout. **Rebuilt 23:06 from declarations, not memory** — class names from the registry entry, weight file names from the worker's loaders, `transformer/config.json` from the `MMDiTConfig` the worker constructs (`cond_embed` true, `in_channels` 64 — the container's own `extracted_values` recorded 64), `vae/config.json` from its `AutoEncoder3DConfig`, text encoders and tokenizers as symlinks to the vendor's subdirectories; the reasoning is written beside the weights in `NEUROBRIX_LAYOUT.md`. The trace needs a 32 GB card to itself (a 31.6 GB VMM pool; a bench beside it is a perturbed bench), so `opensora_chain.sh` waits for the first of cards 2/3 whose bench queue ends, then traces, converts (`--rebuild --stage-root`), gates and uploads |
| re-upload · install · proof by run | owed; the proof needs the rig free of timed cells and is launched by hand after the chain's end marker |

Not listed as delivered. The line in `catalogue-state.md` stays DIAGNOSED until the
proof by run says otherwise.
