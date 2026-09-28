# Release decisions — items that shape a release, recorded where a release is decided

One entry per item, dated, with the measurement that raised it. A campaign directory holds
the evidence; this file holds the decision the release has to make. Closed items move to the
CHANGELOG entry that closed them.

## 0.5.5

### A compiler upgrade inherits a directory proven under the previous compiler (2026-09-20)

**Measured.** The certified autotune directory `nvidia/volta` shipped in 0.5.4 records
Triton 3.6.0 in every one of its 9 655 proofs. The generator gate (`ec938641`) serves a
certified setting only to the compiler that proved it, which is right. So an installation
that upgrades to Triton 3.8.0 — which `pip install torch==2.14` does by itself — gets a
first run in which **every kernel shape sweeps at execution time**: on this rack, a
regression battery that took 64 minutes served ran on a four-hour trajectory unserved
(the replay cache grew 256 → 1 115 entries in 2 h 30; upscaler cells 12 s → 102–113 s).
Nothing warns; the run is merely slow, and the settings it sweeps land in the local replay
cache, which records no compiler version at all and therefore serves them to the next
compiler as well.

**The decision 0.5.5 has to make**, one of:

1. **Ship the directory for both compilers** — the `variants` slot already composes one
   entry per memory class; a generator dimension beside it lets one file carry the 3.6.0 and
   the 3.8.0 proofs, and the gate picks by the running compiler. Re-proof is a night of four
   cards per compiler (measured 2026-09-16/17: 10 087 entries).
2. **Name the upgrade path** — `neurobrix autotune certify --reprove-generator` on the
   machine, said by `neurobrix doctor` and by the engine at the first refused entry
   ("certified under triton 3.6.0, running 3.8.0: N entries will sweep; re-prove with …"),
   so the sweep is a choice the user made rather than a slowdown nobody explained.

The second is cheap and honest and should ship regardless; the first is what makes an
upgrade free. Either way the **replay cache must record the compiler** it swept under and
serve only to that one — the directory gate without the cache gate is half a door.

### The engine under a `CUDA_VISIBLE_DEVICES` mask (2026-09-20) — MEASURED, no release item

A user in a container, on a shared server or under a scheduler runs masked as the normal
case. Three regression guards (`test_serve_warm.py`, `warm_cell_runner.py`,
`test_upscale_offtrace.py`, 2026-08-27) and debt entry D-AUTODETECT-VISIBLE-MASK say the
engine mis-places under a mask; `_apply_visible_filter` (`acd14637`, 2026-09-03) re-indexes
the visible set and the profile is keyed by it since. Whether a facet remains — the profile
index against the runtime ordinal — was measured the same day: red on the tree before the
guard for one card and for two, green on main for one card, for a 32 GB card as ordinal 0
and for two cards with a card-spanning model. No facet remains; the guards and the debt are
in the vacuous-gates register as entry 79 and the skips are off. Nothing for 0.5.5 to close.

### A runtime flag read from the build toolchain's registry is absent in every installed engine (2026-09-20)

**Measured.** Wan2.1-T2V-1.3B rendered a lattice of 16-px cells from every worktree and, by
the same mechanism, from every `pip install`; from the developer checkout it rendered the
sailboat. One judged run per arm on the same commit settled it: the checkout carries
`.nbx_registry`, a gitignored pointer to the build toolchain's `model_registry.yml`, and the
flows read `zero_pad_embeddings` through it (`registry_flags.get_component_flag`). Without
the pointer the flag defaults to false, the UMT5's non-zero pad embeddings (212 of 226
positions for a short prompt) enter cross-attention unmasked, and the video is noise. This is
0.5.4's "degenerate Wan, cause not found".

**What ships in 0.5.5.** The builder now writes `zero_pad_embeddings` into the container
(tokenizer extracted values), so the engine reads it from the `.nbx`; the runtime registry
read remains the developer's override. The four hub containers that declare the flag
(Wan2.1-T2V-1.3B, Wan2.1-I2V-14B-480P, Wan2.1-VACE-1.3B, Wan2.2-I2V-A14B) must be rebuilt and
re-uploaded — no re-trace, the graphs are unchanged.

**The class, closed in the engine and the build the same day (6fb35c3b).** Five other readers
took a flag only from the registry — `i2v_latent_conditioning`, `vace_control_conditioning`
(both engines), the two precision pins Prism reads in `solver.py` (`requires_fp32_compute`,
`fp16_conv_cascade_safe`), and the vision input processor's `pad_image_to_num_frames`. The build
now writes every one of the six into the container's extracted values under the component that
declares it; the container records them when opened (`nbx/component_flags.py`, before Prism
plans); the reader's order is env override → registry (the developer's override) → the
container → default. The rule: an engine decision that depends on a file only the build
toolchain has is a build-side value, written at build.

**What 0.5.5 owes the hub (public, the owner's act):** thirteen registry entries declare at least
one of the six flags and all thirteen are on the hub (listing of 2026-09-20, 47 models): the four
Wan containers above, Allegro-TI2V, CogVideoX-5b-I2V, SANA-Video-2B-720p, PixArt-XL-1024 and
PixArt-Sigma-XL-1024 (`fp16_conv_cascade_safe`), HAT-L-x4, HAT-S-x4, SwinIR-Classical-x2 and -x4
(`requires_fp32_compute`). Each is a rebuild (no re-trace) and a re-upload under the same slug.
Until then every one of them runs at its author's settings only in the author's checkout.

### Sana-1600M-MultiLing does not reproduce the vendor at 1024² (2026-09-21)

**Measured.** The retrace gate's vendor arm (the vendor pipeline rendered at the exact prompt,
seed, steps and guidance) against both containers at 1024×1024, sequential oracle: the hub's
container (traced 2026-06-05) at **17.66 dB** (SSIM 0.800), today's retrace at **17.25 dB**
(SSIM 0.783). Neither reproduces the vendor; the two containers agree with each other no better
(23.7 dB, 75 % of pixels off by more than 8). PixArt-Sigma under the same arm the same morning:
old 13.09 dB, retrace 30.03 dB — so the arm, the seed protocol and the request are sound; the
divergence is Sana's, predates the retrace and survives it.

**What it means.** The public container renders a different picture from the vendor's for a
user's prompt, with no warning. The gate refused the retrace's upload (an artefact that fails the
gate is a workstream, not an upload); the hub keeps the June object until a judged replacement
exists. The 09-09 vendor-correctness cell read "image agrees" from Sana at another request — the
disagreement is at this one (1024², the family protocol's prompt, seed 42), so the next step is
the drift walk on THIS request: the compiled oracle against the vendor per stage boundary (text
encoder → transformer step 0 → decoder), the first over-bound op classified by origin
(kernel / policy / discrete / carrier / scale), never localised from the layout.

**Release consequence.** 0.5.5 does not ship a Sana-MultiLing container as correct; the hub entry
is marked or unlisted until then (see the session record of 2026-09-21 for what the hub offers).

## 2026-09-28 20:06 — RED LINES: census once, certify once, run only to confirm (the owner, via the supervisor)

**The measurement that raised it.** On the Mac the same day: a census walk of chatterbox's generation recorded 4 232
keys in 21 minutes and was still going (the rack: 46 minutes, 4 131); the two Triton gates of the `adbf6e84` bump, read
as "served", took 545 of their 1 400 key uses from the runtime replay cache (sweeps made by an earlier sweeping gate,
keyed by architecture without the code generator) and 27 from runtime sweeps; the working report ran to 336 lines.

**The decision, binding both machines from 20:06.** The slowness of a run is the autotune; we autotune without running
models. (1) One census table per hardware profile, committed in this repository (model, mode, op, kernel key, shape class,
dtype, the census tool revision and the container sha), produced by one tool over every catalogue container, regenerated
only for the containers that changed or for all when a key definition changes — never by walking a model's generation
step by step; a census that takes an hour is a broken census. (2) One certified directory per profile, the only one runs
read, committed, never discarded, certified from that table and nothing else, re-certified only where a best configuration
can move or a shape class changed. (3) Confirmation runs are certified-only: a key missing from the directory is an ERROR
naming the key and the census row that should have held it, never a runtime sweep; the runner always passes the flag; a
miss is a census defect. (4) Confirmation requests are the smallest that still judge the model, chosen per family from
data; full-quality renders once, for the release notes. (5) Findings go once into canonical records; the report is a
current state under 150 lines; history in dated archives; one watcher for inbox and peer, one per running campaign.

**Release consequence.** No new full-model run on either machine until (3) exists in the engine with its test; the
release candidate's gate runs once, on both machines, after every model × dtype × mode is certified from (1) and verified
at zero miss under (3). The manifesto (`CLAUDE.md`, untracked by this repository's own rule) carries the same text at its
top on each machine.

## 2026-09-29 01:37 — old certificates are re-proven, not re-swept; the census comes from one shared derivation (the owner, via the supervisor)

**The measurement that raised it.** On the Mac, 2026-09-28/29: 1 068 keys re-swept under the served generator from
census files in four hours (one to twelve keys a minute; one key in five refused for witness drift, four in five of
those passing a retry), and the table's 1 335 keys begun the same way — because the runtime does not serve a proof made
under another compiler (the generator door) and the certifier's `--reprove-generator` re-ranked every such key by a full
sweep.

**The decision.** 'Certified' means correct and pinned, not the fastest. For every key of the table with an entry under
the retired generator, the certifier RE-PROVES that stored configuration under the served generator — the fp64 oracle and
one timing under the witness, one configuration, seconds — and serves it, recorded as re-proven; a full sweep only where
the stored configuration fails the oracle under the new generator, or where there is no entry at all. Re-ranking for
speed belongs to the kernel-optimisation chantier at the end of the roadmap, not to the release. One implementation for
both machines. The census: the same `.nbx` serves both machines, so one derivation from the graphs (no shadow execution)
produces the profile-independent part once, committed, projected through each profile; the walks of 2026-09-28 were the
last, and their tables are that derivation's test data.

**Release consequence.** A pass reports how many keys were re-proven, swept, and why; no key is certified on a drifting
witness and none is left uncertified without its retry.
