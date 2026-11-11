# Release decisions — items that shape a release, recorded where a release is decided

One entry per item, dated, with the measurement that raised it. A campaign directory holds
the evidence; this file holds the decision the release has to make. Closed items move to the
CHANGELOG entry that closed them.

## 0.5.5

### A compiler upgrade inherits a directory proven under the previous compiler (2026-09-20)

**Measured.** The certified autotune directory `nvidia/volta` shipped in 0.5.4 records
Triton 3.6.0 in every one of its 9 655 proofs. The generator gate (`418d2968`) serves a
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
engine mis-places under a mask; `_apply_visible_filter` (`7a307287`, 2026-09-03) re-indexes
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
re-uploaded — no graph rebuild, the graphs are unchanged.

**The class, closed in the engine and the build the same day (5eb49103).** Five other readers
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
(`requires_fp32_compute`). Each is a rebuild (no graph rebuild) and a re-upload under the same slug.
Until then every one of them runs at its author's settings only in the author's checkout.

### Sana-1600M-MultiLing does not reproduce the vendor at 1024² (2026-09-21)

**Measured.** The rebuild gate's vendor arm (the vendor pipeline rendered at the exact prompt,
seed, steps and guidance) against both containers at 1024×1024, sequential oracle: the hub's
container (built 2026-06-05) at **17.66 dB** (SSIM 0.800), today's rebuild at **17.25 dB**
(SSIM 0.783). Neither reproduces the vendor; the two containers agree with each other no better
(23.7 dB, 75 % of pixels off by more than 8). PixArt-Sigma under the same arm the same morning:
old 13.09 dB, rebuild 30.03 dB — so the arm, the seed protocol and the request are sound; the
divergence is Sana's, predates the rebuild and survives it.

**What it means.** The public container renders a different picture from the vendor's for a
user's prompt, with no warning. The gate refused the rebuild's upload (an artefact that fails the
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

## 2026-09-29 13:45 — the owner's method restated (the supervisor's inbox entry 69)

ZERO hacks, ZERO hardcode, data-driven. Universality is ONE code that runs every model with the values carried by
the data: the `.nbx` configs, the hardware profiles, the family profiles. The container builder embeds them from the registries;
the builder's model registry may be extended with any value a model needs to execute,
each with its source. No compute wasted, no errors, problems treated at the source. The tools that census,
compute certificates and run the anti-regression are named as the origin of many problems: the Dell writes their
contracts and tests, one implementation for both machines; the Mac reviews what arrives on the peer channel
against Metal. Raised by the day's measurements: a settle step written around Prism's host reading (removed, the
reading fixed at the source, `2db23385`), a census walk whose dtype placement was the door's (Voxtral), and a
zero-miss column read as a judged image (renamed; images judged from outside).

## 2026-10-12 15:25 — NOTHING IS LOST (the owner, via the supervisor)

Four sub-agents ran on the Dell all night, never collected; their work was found nowhere.

1. Every job launched (sub-agent, chain, script, GPU run) has, before it starts, a line in the agent's report — what, where it writes, how its end is known — and an end watcher. No work runs unreported.
2. A sub-agent writes its own deliverable to disk in the campaign, at a path fixed when it is launched, not only in its reply. It is collected and closed in the turn it ends.
3. At every wake, every context compaction and every turn end, the live sub-agents and chains are listed and compared with the report; a gap is fixed and noted.
4. Every sub-agent and chain of the last 48 h has its deliverable on disk and in the report, or is recovered from the campaign and the session logs under RECOVERED.
5. A job that ran unreported is a fault of method, not an incident.

## 2026-10-12 15:30 — CONTAINER BUILDER guidelines (the owner, via the supervisor)

The container builder is the proprietary tool that builds our models' containers.

- It is used explicitly through its entry point and documented commands (its repository README and the CLAUDE.md at its root). It has its own git branch, and the builder's main is the only line.
- A model's container is never built with a custom script.
- The container builder stays portable and builds every model we own: a lost container is rebuilt by it into an exact copy.
- Every specification stays in force:
  - NeuroTax 5.0;
  - symbolic shapes are mandatory, with no frozen dimension;
  - one algebraic-function tool for symbolic shapes, extended and fixed as needed;
  - unified VMM for every build.
- A fix never breaks another model's build. The build suite of the whole catalogue is the proof before a container-builder fix lands.
- Hard code is allowed in the container builder, a private tool, but is not desirable.

## 2026-10-12 18:20 — CONFIDENTIALITY and ECONOMY (the owner, via the supervisor)

Confidentiality: the work is confidential. Never /feedback, /bug, or any send to the model vendor beyond the
model calls themselves. No session-link trailer and no co-author trailer in any commit message (the engine
repository is public). Agent settings disable non-essential traffic, telemetry, error reports and co-author
lines on all three machines from their next session start.

Economy: sub-agents only in extreme necessity, the reason written in the report before launch (zero by default);
never a turn spent checking that a job progresses (a shell watcher wakes the agent at its end); scripts written
once into the campaign and re-run by path, never pasted; one command per step with truncated output; the report
is rewritten on disk, not repeated in the conversation; anything not visible in the owner's terminal is closed.

## 2026-10-12 19:50 — CHAIN OF COMMAND, CLI-ONLY TESTS, LAST DATED PROOF, CODE HYGIENE (the owner, URGENT, via the supervisor)

0. **Chain of command.** The supervisors steer, not the agent. An inbox entry marked URGENT is executed before any
   other action, at once, and acknowledged on the first line of the report; orders run in the order given.
1. **Tests are the engine's command line, nothing else.** A model is tested as a user does: the `neurobrix` entry point
   with the command and flags its request needs (model, prompt, resolution, steps, frames, temperature, audio or image
   input), then the output is looked at. No custom test script. A script may only chain many identical commands, or
   measure a dimension outside the engine's workflow; it lives in the campaign, never in this repository, and invents
   no execution path. The gate harness only orchestrates that exact command, written in every result row so it can be
   replayed by hand. Test scripts that do not respect this are inventoried and deleted.
2. **"Validated many times, broken today."** Before a validated model is announced broken, it is compared with its last
   dated proof under the same CLI command and flags, naming which of the three changed: the container (sha), the engine
   (commit) or the request/test. If the test changed, the defect is the test. No fix breaks something else: each fix is
   proven at the CLI on the fixed model and on a neighbouring validated model before it lands.
3. **Code hygiene.** A change is complete or does not exist: no unwired fragment, no old code path left beside the new
   one, the touched file cleaned, every change committed and pushed to both remotes (git history is the only way back).

## 2026-10-13 02:15 — THE DTYPE IS PER COMPONENT (the owner, via the supervisor)

1. Each component of a model runs in the dtype its safetensors headers give, and the container builder declares it as is, per component,
   in the container. There is no "dominant dtype" voted across components: a vote, a tie or a walk order never decides a
   dtype. The state dtype in `runtime/defaults.json` is derived per component, never global.
2. The only exception: a value beyond the hardware's precision margin (fp16 overflow on V100) forces fp32. That decision
   belongs to the DtypeEngine at run time, from the hardware profile and the tensor itself, never by model name, and it
   is never written into the container: the container stays portable (Apple and Hopper do not share a margin).
3. No GPU on a known defect: every container whose state dtype differs from its components' header dtypes is found by
   reading the manifests and headers (zero execution), corrected by the container builder (metadata rewrite, no graph rebuild when the graph is
   right), then its dated proof is replayed at the CLI. The Mac's reading of 2026-10-13 flagged 20 of 49 containers.

## 2026-11-05 22:55 — NBXTENSOR 0.0.1 IS PUBLISHED (the owner's GO 22:55, confirmed 23:51, via the supervisor)

1. **The tree published** is the final gate's, `f1292590a`, plus the five files of `nbxtensor/` changed since, each through
   the lot suites (no card and on a card) and the clean-machine test: `CHANGELOG.md`, `pyproject.toml` (the dependency on
   `nvidia-cuda-runtime-cu12` on Linux, without which a clean machine finds no GPU runtime), `backend/metal_device.py`
   (the Metal dispatch limits declared or refused), `tests/test_readme.py` and its test — then the version commit
   `4a80d4da1` (the changelog dated, the project URLs, six comments that named a person now carrying their reason).
2. **The gate's proofs**: 196 of 196 cells green, 49 models x {PyTorch, Triton} x {16 GB, 32 GB V100}, every output
   judged by eye (`registres/benchmarks/gate_times.tsv`, campaign `2026_11_04_porte_finale`); no cell over 1.5 times its
   previous gate measured alone; the torch-free proof on card 3 (library 225 passed, five reference models with the base's
   output shas, an uncertified sweep of 22 keys, zero attempts to import torch).
3. **The clean-machine test**: the sdist and wheel built from the version commit pass `twine check`; installed in a
   fresh container (torch absent) and in a fresh venv, TinyLlama-1.1B-Chat-v1.0 through `neurobrix run --triton` gives
   the gate's judged output (sha256 `6f0f9704…`).
4. **Not in 0.0.1**: the replay's extents (Metal replay, landed after as 0.0.2's first lot); the Triton environment
   switches that choose a route or a tile (3.13, 0.0.2).
5. **Where it is**: PyPI `nbxtensor` 0.0.1 (wheel sha256 `1fd17d29…`, sdist `bc7f890f…`, the artefacts of point 3),
   `pip install nbxtensor` with no version installing it on a clean machine; GitHub `NeuroBrix/nbxtensor`, public, tag
   `v0.0.1` on the export of the version commit, its release marked Latest (a stable release, not a pre-release); GitLab
   `neurobrix/nbxtensor`, public, the same export, tag and release; the engine repository's tag `nbxtensor-v0.0.1` on
   `4a80d4da1`. The tag is never moved: what the published README gets wrong (its install line, a CPU backend named in
   the package description, references to documents the library's repository does not hold) is corrected in 0.0.2.

## 2026-11-06 18:10 — NBXTENSOR 0.0.2 IS PUBLISHED (the owner's GO covers the 0.0.x series, the supervisor 2026-11-06 10:25)

The owner's decision of 10:22: nbxtensor supports what Triton supports, and its README keeps "Runs wherever Triton runs —
NVIDIA, AMD, Intel and Apple Silicon GPUs"; the AMD and Intel code ships in 0.0.2, untested on a card (no such card in the
lab), written to the level of the NVIDIA and Apple drivers. Content: the HIP launch driver, Intel's Level Zero runtime and
launch driver, the install help (`python -m nbxtensor doctor`), the AMD/Intel device files, the README showcase and
CONTRIBUTING, nine environment switches removed, the replay's launch extents, the certified directory's compiled index.
Proofs on the version commit `587325db3` (the engine's tag `nbxtensor-v0.0.2`): the card-less suite rc 0, the device
suite 0 failed, the torch-free proof (the library's tests, five Triton models at their gate request, their shas equal to
the recorded ones, no torch import attempted) PASS; the Mac's Apple suites green (2 112 passed, the one red a test that
read a data key of the Metal table as an entry point, fixed: 64/64). Published: PyPI `nbxtensor` 0.0.2 (sdist + wheel,
twine check passed, read: no machine path, no personal name, no private tool's name); GitHub `NeuroBrix/nbxtensor` and GitLab
`neurobrix/nbxtensor`, the export `081204ef`, tag `v0.0.2`, release with both files; the engine repository's GitLab
release `nbxtensor-v0.0.2`. Checked on a clean `python:3.10-slim` container: `pip install nbxtensor==0.0.2` from PyPI
(numpy, triton 3.8.0, no torch), `python -m nbxtensor doctor` ready on the card, the README's GPU example correct.

## 2026-11-06 20:35 — NBXTENSOR 0.0.3 IS PUBLISHED (the owner's GO covers the 0.0.x series)

Two defects of 0.0.2 read by the supervisor in its published metadata (18:25): nbxtensor depends on the CUDA runtime wheel
on Linux and a runtime library loads without its card, so on an AMD or Intel machine 0.0.2 named the CUDA backend and
never tried HIP or Level Zero; and the package summary still named NVIDIA and Apple only. Detection now tries only the
runtime libraries of the GPU vendors present on the PCI bus (all of them where the bus cannot be read), and the summary
says it runs wherever Triton runs — NVIDIA, AMD, Intel and Apple GPUs. Proofs on the version commit `cd3ee0367` (the
engine's tag `nbxtensor-v0.0.3`): the card-less suite rc 0, the device suite 0 failed, the torch-free proof PASS.
Published: PyPI `nbxtensor` 0.0.3 (sdist + wheel, twine check passed, content read clean); GitHub `NeuroBrix/nbxtensor`
and GitLab `neurobrix/nbxtensor`, the export `425a9508`, tag `v0.0.3`, release with both files; the engine repository's
GitLab release `nbxtensor-v0.0.3`. Checked on a clean `python:3.10-slim` container: `pip install nbxtensor==0.0.3` (no
torch), `python -m nbxtensor doctor` ready on the card, the README's GPU example correct.

## 2026-11-09 16:25 — NEUROBRIX 0.6.3 AND NBXTENSOR 0.0.8 ARE PUBLISHED (the owner's GO 16:22, via the supervisor)

Content: serving (`neurobrix serve`, `chat`, the OpenAI-compatible HTTP API), the conversation's KV cache kept between
turns, one name per model (the hub name), the clean terminal, integrated GPUs, the in-run re-plan after a device OOM.
Proofs on the tag commit `2c4e37f1` (tags `v0.6.3` and `nbxtensor-v0.0.8`): the gate 196/196 rc 0 (190 same sha, 2
card-dependent, 6 slow cells gone on back-to-back re-timing), the whole suite, the five shas and the torch-free proof,
the user trials L (Linux) and M (Mac) on the built wheels. Published: PyPI `nbxtensor` 0.0.8 and `neurobrix` 0.6.3
(wheel sha256 65a8ee26…, the tested one), GitHub and GitLab releases. Stopped by the supervisor at 16:24: the release
text claimed the macOS 14 Metal wheels, which `install-metal` 0.6.3 does not install; the text was corrected in place
(both releases, CHANGELOG known limits and README on main, `028cb917`), the publication then resumed (16:26). The PyPI
trials: M green; L red on Sana's first launch with the Triton engine (the reload probe on a zero-size constant), fixed
in 0.6.4.

## 2026-11-09 18:08 — NEUROBRIX 0.6.4 AND NBXTENSOR 0.0.9 ARE PUBLISHED (the owner's permanent GO for this corrective, 16:47)

The last corrective release (the owner, 17:13): from now on fixes accumulate on main and leave grouped in the next
version useful to users. 0.6.4 keeps two public promises 0.6.3 broke: `install-metal` installs the macOS 14 Metal
wheels for CPython 3.10-3.13 (and the CPU pool runs on macOS 14's Accelerate, which has no BLASSetThreading), and the
Triton engine's first launch no longer stops on a zero-size constant (Sana 1024 from PyPI). The GO's three conditions,
all met on the tag commit `3aa4fb93`: the Mac's trial on the final wheels (dry-run 3.12/3.11, TinyLlama certified-only
under 3.11, Sana under a simulated macOS 14 Accelerate, outputs equal by sha); the Sana fix understood and proven on the
user path (a fresh container, Sana first, the text encoder's reload probe run, rc 0, the image equal to 0.6.3's judged
one); trial L on the local 0.6.4 wheels. The new rule held: every sentence of the CHANGELOG and the release notes read
against the code and the tests by an agent that did not write it (the Mac: 15 promises, all held). Published: PyPI
`nbxtensor` 0.0.9 and `neurobrix` 0.6.4 (the tested wheels, read back by sha256), GitHub and GitLab releases. PyPI
trials L (Linux) and M (Mac, CPython 3.11, the macOS 14 wheels) green.

## 2026-11-10 13:50 — NEUROBRIX 0.6.5, NBXTENSOR 0.1.0 AND NBXGRAMMAR 0.1.0 ARE PUBLISHED (the owner's GO, 12:52)

The next useful version (the owner, 2026-11-09 18:58): the serving daemon re-plans after a device out-of-memory,
constrained decoding on the OpenAI API (`response_format: json_schema`, forced tool calls) by NeuroBrix's own grammar
engine, `/v1/embeddings` and an embedding family, the KV host tier and the mid-decode spill (a served context past the
card, stated at start, `400 context_length_exceeded` past it), hybrid laptops described; with the owner's order of
02:49 on the slowness of mixture-of-experts models on two cards (Qwen3-Coder-30B-A3B served 6.7 s → about 0.5 s a token,
run 0.33 s). Three bricks, each its own version: the grammar engine became its own package and public repository,
`nbxgrammar` 0.1.0 (Rust, std only, one wheel per platform: Linux x86_64 and aarch64, macOS arm64 and x86_64, Windows
x86_64, built and smoke-tested by its CI, the Linux library byte-identical between the rack and the CI); nbxtensor left
0.0.x for 0.1.0 (the owner, 12:57); neurobrix 0.6.5 pins both exactly, nbxgrammar behind the five platform markers
(the Mac's condition, held: every pinned platform had its wheel on PyPI before the engine). The freeze `83732b95`
(tagged at `e26bc5f1`, the CHANGELOG heading only): plans 0 of 196 moved in execution against 0.6.3, the five shas
identical in the torch-free environment, the whole suite once and the device suite (one regression found — a test
whose reference the tier restored into — fixed, and the compiled engine's park found missing and added), the serve/KV
cells of both engines tier on and off, trials L-G and M+G on the final wheels; every promise reread by the Mac (README,
docs, CHANGELOGs, notes; the owner's rule: Intel and AMD GPUs are targets, integrated Intel GPUs described and named,
never "driven"). Published, each read back: PyPI `nbxgrammar` 0.1.0 (six files), `nbxtensor` 0.1.0, `neurobrix` 0.6.5
(publish.yml); tags `v0.6.5`, `nbxtensor-v0.1.0`, nbxgrammar `v0.1.0`; GitHub and GitLab releases; the organisation
page. PyPI trials L (Linux, both legs) and M (Mac, CPython 3.11, nothing local) green: constrained decoding installed by
`pip install neurobrix` and answering a valid JSON.

## 2026-11-11 05:16 — NEUROBRIX 0.6.6, NBXTENSOR 0.1.1 AND NBXGRAMMAR 0.1.1 ARE PUBLISHED; PUBLISHED NUMBERS ARE CONSECUTIVE (the owner, 02:25 and 02:31)

A grouped publication of the work landed since 0.6.5 (the supervisor's order, 02:25): a long prompt read several times
faster by the serving daemon inside a memory plan that counts what the engine uses, no memory taken on GPUs a model
does not use, the output layer counted once, reference flags refused by name, the experimental Intel backend's Gen12
chips and its run-time refusal, the grammar engine from its own repository.

**The owner's rule of that night.** Main had taken a number at every landed lot (0.6.6 to 0.6.21, nbxtensor 0.1.1 to
0.1.9) while PyPI stood at 0.6.5 / 0.1.0. The owner refused the jump: a version number exists only when it is
published, published numbers are consecutive, main carries one number per product — the next to publish — and a
publication increments it, a fix does not (`docs/reference/RELEASING.md`, rule 0). The release therefore went out as
0.6.6 / 0.1.1 / 0.1.1: the sixteen and nine changelog entries merged into one each, organised by what a user gains;
and, by the owner's one-time word, the public history after the last published tags was re-created commit for commit
(same authors and dates, the lots kept one commit each) so that no commit message and no committed file names a number
that was never published. Engine main: `2cbf08d56` → `96fa039f1` (the tip's tree byte-identical to the renumbered
tree); the exported nbxtensor repository re-derived from it; nbxgrammar needed nothing. Published tags and every
commit at or before them kept their sha.

**Proofs.** Plans: 0 of 196 moved against the last gated listing; against 0.6.5, 27 cells of seven mixture-of-experts
models (expert cache size, session reach) — the 14 compiled ones replayed green when they moved, the 13 Triton ones
replayed on the rewritten freeze, green, every output read, none slower than 1.5x its last gate time. The five shas
identical in the torch-free environment, 0 torch import attempts. The whole suite once: two reds, both fixed at the
source (a single vendor's device prefix in the weight loader; a test file that ended the interpreter when run alone),
then the device suite, 0 failed. Two defects of the new user's path reported on 0.6.5 during the night and fixed
before the tag: a model's name (one message whatever was typed, the lists print the names `run` accepts, and 33
documentation commands corrected to them) and the "First launch ... once" line said at later launches. The freeze
`b8350802b`. Every public sentence reread by the agent who did not write it; the two pure packages rebuilt
byte-identical on the second machine. Trials L (Linux, both legs) and M (Mac, Metal) on the built wheels, then from
PyPI: green, with the reporter's own sequence (three launches in one home, the four name cases).

**Published, each read back:** PyPI `nbxgrammar` 0.1.1 (six files), `nbxtensor` 0.1.1, `neurobrix` 0.6.6
(publish.yml, pinning both exactly); tags `v0.6.6` and `nbxtensor-v0.1.1` on `b8350802b`, nbxgrammar `v0.1.1`; GitHub
and GitLab releases; the organisation page. Seen at the PyPI trials, nothing of ours: for the first minutes after an
upload the index `pip` reads does not list the new files yet, and an install then fails with "No matching
distribution" (both machines' first attempts; the second, minutes later, installed).

## 2026-11-11 11:48 — NEUROBRIX 0.6.7 AND NBXTENSOR 0.1.2 ARE PUBLISHED: A CORRECTIVE VERSION (the owner, 10:13)

Found the morning after 0.6.6 by a test order on Apple: nbxtensor's Metal driver kept its pinned buffers under their
base address alone, and a pool segment begins where its first block does — an address inside another block of the
segment could be answered from the first block's short buffer, past its end: wrong bytes, nothing raised. What it could
reach: the expert store that copies experts into an arena, on Apple only, which the engine does not build there (a
unified device's store binds its experts in place). The supervisor asked for one deciding test — can a real `serve`
path on the published 0.6.6 reach it: NOT REACHED on the two plans the mixture-of-experts model that fits a 24 GB Mac
takes (no pinned address out of two million past its buffer); the expert-store plan itself could not be run on Apple
hardware. The owner's decision all the same: a defect by which a model can return wrong values is not a small bug and
does not wait for a batch. The version carries main as it stood since v0.6.6 and nothing else (the fix, the card
residency armed only where the backend's kernels read device addresses, the expert tables naming their card, two
documentation corrections, one test pinning that the speech front ends never pad).

**Proofs, proportionate (no full gate).** The touched directories on both machines (rack, on cards: 1 554 passed, 0
failed; Mac, Metal: 1 510 passed); the five shas identical in the torch-free environment; deepseek-moe-16b-chat on a
16 GB card through the entry point, both engines, byte-identical between the freeze and the tag v0.6.6; every public
sentence reread by the agent who did not write it — the changelog says what was observed, not that no model could be
affected; the pure packages rebuilt byte-identical on the second machine; trials L and M on the built wheels, then
from PyPI.

**Published, each read back:** PyPI `nbxtensor` 0.1.2 and `neurobrix` 0.6.7 (publish.yml; `nbxgrammar` stays 0.1.1,
not republished); tags `v0.6.7` and `nbxtensor-v0.1.2` on `7772275ba`; GitHub and GitLab releases of both; the
organisation page.
Said as it is: the library's two files on PyPI are the frozen bytes; the engine's two archives, built by publish.yml,
have the freeze's CONTENT (every member's sha256 equal, 548 of 548 in the wheel, 549 of 549 in the sdist) and not its
bytes — the workflow builds without the commit's date. The same was true of 0.6.6. To correct in the next batch.
PyPI trials L (Linux, both legs; the Sana image byte-identical to the one judged for 0.6.6) and M (Mac, Metal) green,
each on its second run: for the first minutes after an upload the index `pip` reads does not serve the new files,
while a request of our own to the index already lists them — the check must be pip's own (`pip download --no-deps`).

## 2026-11-11 23:44 — NEUROBRIX 0.6.8 AND NBXTENSOR 0.1.3 ARE PUBLISHED: A CORRECTIVE VERSION (the supervisor 19:15, the owner 21:47 / 22:03)

Found while a mixture-of-experts coding model (Qwen3-Coder-30B-A3B) served NBX Code's first runs, and by the gate's two
new kinds of cell: wrong answers or stops on the paths that model takes, and on a small card. The owner's order:
wrong values at a user ship at once, and every defect met on the two-card path is fixed before the version goes out.
Seven user-facing fixes, each found by a measure and fixed at its source with a test seen red:
a served MoE model slowed after its first request (three causes: a replayed step left the dense weights on the host;
every later token re-bound all the weights; each weight bound as a view was copied onto itself); an answer could end
with a bare tool-call marker; a MoE model spread over two cards read the wrong experts and wrote past its outputs
(nbxtensor's per-card tables now keep the layer's length) and its decode cache was sized against both cards then the
whole rung (out of memory, both engines); a dense model split over two cards failed at its first hand-off (nbxtensor:
a chunk of a card's expandable pool was granted to its card alone); a model streamed layer by layer on a small card
answered garbage (a replayed piece did not count its attention call, every later layer's keys went one layer off) —
the last two in 0.6.7 too. One wording change: 77 engine messages speak of building containers.

**New cells of the gate, seen failing first.** The SERVE cell (`tools/gate_serve_cell.py`: one daemon, three requests,
the third at the first's speed within the gate's 1.5x): RED on ef8f0edee (2.33x), GREEN on the freeze (0.98x). The
MULTI-CARD cells (the cell runner on two cards, certified-only, both engines): a dense model (GLM-4.1V-9B-Thinking,
2 x 16 GB) and the MoE model (2 x 32 GB) — red before their fixes, green on the freeze, the same answer on both engines.

**Proofs on the freeze a45f4e6a0.** Torch-free proof PASS (0 torch imports; the library's 399 tests; the five shas
equal to their references); every lot's touched directories on cards (tests/unit/triton 225-281 passed, the prism
directory in 14 shards, the boundary, the library); trials L (Linux containers, both legs) and M (Mac, Metal) on the
wheels, then from PyPI, all green, 0 Traceback; the Sana image looked at, byte-identical between the wheels and PyPI.
Every public sentence reread by the Mac (one maker name struck from the release notes).

**Published, each read back:** tags `v0.6.8` and `nbxtensor-v0.1.3` on `a45f4e6a0` (origin and GitLab); PyPI
`nbxtensor` 0.1.3 (both files by sha256) and `neurobrix` 0.6.8 (publish.yml 43114215983; the archives have the freeze's
content, every member's sha256 equal: 548 of 548 in the wheel, 549 of 549 in the sdist); GitHub and GitLab releases;
`nbxgrammar` stays 0.1.1. Said in the release notes and not as a claim: for a MoE model that does not fit one card, two
cards are not faster than one today (the two-card plan runs op by op: the reading, with the measures, is in the
register), and how to run it on one card.
