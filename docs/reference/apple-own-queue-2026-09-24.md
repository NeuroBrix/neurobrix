# The twelve Apple models that were mine — final account, 2026-09-24

The 2026-09-22 census left 33 of 59 models open on this Mac. Twelve were mine to answer. This
is what each one is now, what was fixed, what was named, and the four decisions that are
Hocine's rather than mine.

Everything below is measured. Where a number is quoted, the command that produced it is in
`campagnes/2026_09_22_apple/`.

---

## The account, model by model

| # | model | was | is |
|---|---|---|---|
| 1 | PixArt-XL-1024 | `Cannot broadcast (2,1,1152) and (32,4096,1152)` | **closed**, censused + certified |
| 2 | PixArt-Sigma-XL-1024 | same | **closed**, censused + certified |
| 3 | PixArt-XL-2-1024-MS | `addmm::0 AssertionError:` (empty) | **closed**, censused + certified |
| 4 | PixArt-Sigma-XL-2-1024-MS | same | **closed**, censused + certified |
| 5 | Sana-1600M-MultiLing | `Cannot broadcast (1,32,128,128) and (1,128,128,32)` | **closed** by the rack's `2a21e41e` |
| 6 | Sana_1600M_1024px_MultiLing | `bmm shape mismatch: (140,33,16384) @ (35,16384,128)` | **named** — 3 of 4 defects fixed, closes by the rack's retrace |
| 7 | Allegro-TI2V | `None of the sources resolved: ['global.image']` | harness **closed**; memory + a new `slice::55` defect named |
| 8 | CogVideoX-5b-I2V | same | harness **closed**; memory only — APPLE (memory / rung) |
| 9 | Wan2.1-VACE-1.3B-diffusers | same | harness **closed**; memory + a new `div::9` defect named |
| 10 | Qwen3-VL-30B-A3B-Thinking | `aten::index_put ... value broadcasting unwired` | **named** — a capability to build |
| 11 | VibeVoice-1.5B | `decode branches need the KV cache path` | **named** — a capability to build |
| 12 | granite-3.1-1b-a400m-instruct | `the triton-ext driver cannot bind pointer ...` | **named**, and **nothing is owed upstream** — see the correction |

Four closed outright, one closed by the rack, three had their stated cause closed, four named
with a measured cause. Nothing is unclassified and nothing is guessed.

---

## What was fixed, and how each was proven

Three defect classes, not one. The six "shape defects" of the mandate were three mechanisms
across two model families, and separating them was most of the work.

**1. A patchified token count the spatial pass could not see** (`ae0d1908`).
`_spatial_promotion_pass` knows H, W, H·W and their *upscaled* multiples; it was written for
VAE decoders, which scale up. A patch-embedded transformer **divides**: its grid is
(H/p)·(W/p), so PixArt's 64, 4096 and 8192 matched nothing and 224 target groups stayed
literal. `metadata_ops._reshape` then invented a numel-preserving shape instead of refusing,
folding the ratio into the batch. The token expression is now harvested from the graph's own
correctly-symbolised patch-embed view, so no patch size is inferred and any tokenisation
scheme is covered.

Measured before and after on the census shadow:

| request | before | after | GEMM M after |
|---|---|---|---|
| 1024² (traced) | clean | clean | 8 192 |
| 1536² | **clean** | clean | 18 432 |
| 2048² | `(8, 4096, 1152)` | clean | 32 768 |
| 4096² | `(32, 4096, 1152)` | clean | 131 072 |

1536² was already correct **because** its ratio is 2.25 and the invention cannot scale a batch
by a fraction. The clean integer ratios were the broken ones — a gate that sampled only 1536
would have reported health.

**2. A request symbol standing in for a fixed extent** (`86fa1ef4`, `15a1fbfb`, `c76a4620`).
Three shapes of the same mistake, each needing a different discriminator:

* a **slice end** — PixArt's timestep embedding is 256 wide, split in half; at 1024 px the
  half (128) equals the latent height, so the tracer bound `emb[:, :128]` to `height`.
  Discriminated by reachability: that tensor descends from `input::timestep`.
* a **head dimension** — Sana has 70 heads of 32 and a 32×32 latent. Reachability cannot help
  (the tensor *is* spatial, the position is not), so the discriminator is structural: a group
  of view-target entries reconstructing a **weight extent** may not hold a request symbol.
* the same **inside arithmetic** — Sana writes its hidden size as `mul(70, height)` and its
  token count as `mul(height, width)`, side by side in one target. An entry naming **both**
  axes is a genuine spatial quantity; a lone axis at a weight-fixed position is the mistake.

The rule was shaped by measurement, not applied and hoped for: Sana's VAE `aten.view::0` is a
**genuine** `[SYM(s0), tokens, SYM(s1), SYM(s2)]`, which a blanket rule would have frozen and
broken the decoder at every size but the traced one. Blast radius, measured on all five image
containers: **Sana 243 entries changed, all four PixArt containers 0.**

**3. A latent hazard closed before it was met** (`a4361cf5`). `_walk_shape_list` is shared with
`aten::expand`, where `-1` means *keep this dimension*, not *infer it*. The token-collapse
rewrite had no op-type guard, so it could have changed an op's meaning rather than its extent.
No container triggered it — 0 expand targets carry a `-1` after the pass — so this is a hazard
closed, not a regression fixed.

**4. A census that silently dropped a required input** (`96cc16ea`). `_declares_image_input`
answered `except OSError: return False`, making *"I could not read the topology"*
indistinguishable from *"this model needs no image"*. On this Mac the container cache is NFS
over Wi-Fi. The evidence that this is the path, and the only one: the recorded command carries
no `--input-image`; the recorded family is `video`, so the family guard passed; all three
models declare `global.image` and `request_args` supplies one; and the block predates the
census. Re-censused with the mount healthy, the image is now in every request and
`global.image` appears in `CLI inputs`.

**5. Two bare assertions given their numbers** (in `ae0d1908`). `addmm` and `baddbmm` asserted
`K == K2` with no message, so the census received `AssertionError:` with nothing in it and two
models' causes cost a reproduction run to learn.

Every fix was seen red first. Tests: `test_a_patchified_token_count_is_not_left_literal.py`,
`test_a_slice_end_is_not_a_spatial_symbol_by_coincidence.py`,
`test_a_head_dimension_is_not_the_image_height.py`,
`test_an_unreadable_topology_is_not_a_model_without_an_image.py`.

---

## Censused and certified

The four PixArt containers, 6 rungs × 2 modes:

* **0 logs carry a shape error**, out of 24. Before, every probe died.
* keys per model **8–9 → 112–118**; 166 total against 51.
* **166 served, 0 to certify** — measured by re-running the census against the directory, not
  read off the certifier, which printed `CERTIFY COMPLETE` with rc=0 after a **four-second**
  round while two earlier rounds had exited rc=1 refusing sweeps. Those two facts could not
  both mean coverage, so the exit code was not the instrument.
* the whole Apple directory: **3 293 entries, 0 without a passing fp64 proof**.
* the regime witness opened at 4.4110 / 4.4087 / 4.4105 ms across three rounds — 0.05 % spread
  on an 8 % tolerance, so every configuration was pinned under the same quiet host.

Checking the refused sweeps found the witness working rather than a hole: three keys were
refused on a GPU still **ramping after a jetsam kill** — drift 37 %, 40 %, 52 %, always getting
*faster* — and re-swept correctly once it settled. **The 60 s settle in `recensus_closed.sh` is
not long enough for Apple's clocks to come back from idle**, and that is tunable by
measurement.

---

## What remains, with its evidence

**The four PixArt containers still refuse the three lowest rungs** (4096 / 6144 / 8192 MB) with
`text_encoder at 9630MB (W=9083)`. A 9 GB T5 does not fit a 4 GB rung. `APPLE (memory / rung)`,
closed by naming, and verified **pre-existing**: the same refusal appears in a log made before
any of these fixes.

**Two new engine defects, revealed by closing the harness gap** — both on the ordinary arm at
several rungs, so reproducible without a probe:

* `Wan2.1-VACE-1.3B-diffusers`: `Cannot broadcast (1, 384, 1, 112, 112) and (1, -1152, 1, 112,
  112)`. A **negative extent**: −1152 is −1 × 1152, a `-1` "infer this dim" sentinel evaluated
  as arithmetic instead of resolved. Same family as the work above, which is why the expand
  guard mattered.
* `Allegro-TI2V`: `aten.slice::55 IndexError: tuple index out of range`.

**A correction to the mandate's own framing.** granite-3.1 was queued as *"waiting on a Metal
driver fix — what is owed upstream to triton-ext"*. **Nothing is owed upstream.**
`_containing_allocation` looks up `DeviceAllocator._range_size` — NeuroBrix's own registry — so
a pointer absent from it is one **our** allocator never handed out. The message begins "the
triton-ext driver cannot bind…", which is what made it read as theirs. The plan is `single_gpu`
with everything on `mps:0`, so this is not a host-placement artefact either: it is a defect in
our own residency bookkeeping on the MoE expert-pointer path (`triton/moe.py`
`_build_ptr_tables`). No tracker issue was filed, because filing one would have been wrong.

**Two capabilities to build, not bugs to fix.** `Qwen3-VL-30B-A3B-Thinking` needs
`aten::index_put` value broadcasting; `VibeVoice-1.5B` needs the KV-cache path wired for its
decode branches. Both are the doctrine's "a missing capability is a chantier": they extend
NBXTensor and the house kernel family, and neither is a half-hour's work.

---

## Cross-backend verdict

**Not Apple-only, and the distinction matters per family.**

* **PixArt** — the defect fires at plain **2048 px** on shared code, and the four containers
  are clean by the rack's own `weights_are_not_symbolic.py` (0 offending parameter dims, run
  here to confirm rather than infer). Their detector is *right*: PixArt's misattribution lives
  in an activation shape arg and a slice bound, which it does not scan. The rack's PixArt green
  is **unexercised above 1024 px, not evidence of absence**.
* **Sana** — the rack already knows. `2a21e41e` names eleven containers awaiting retraces and
  Sana_1600M_1024px_MultiLing is **first, with 174 parameter dims**.

What is genuinely Apple-only is the **rung ladder** that exposed all of it.

Owed to the rack and recorded in `owed-proofs.md`: a CUDA proof at **2048 px** for the four
PixArt containers, in `--compiled` as well as the triton modes.

---

## The four decisions that are Hocine's

1. **Sana_1600M_1024px_MultiLing.** I stopped after three fixes. In that container 32 is at
   once the latent side, the head dim, the VAE channel count and the input channel count, so
   the next discriminator starts risking a genuine spatial slice — and it is first on the
   rack's retrace list with 174 offending dims. **My recommendation: leave it to the retrace.**
   The alternative is more engine-side adaptation against a container already queued for
   replacement.

2. **`metadata_ops._reshape` invents instead of refusing.** Every defect in this campaign
   surfaced about eleven ops downstream of its cause, because a mismatched target is silently
   repaired. Making it refuse is a catalogue-wide change and its docstring says the fallback is
   load-bearing for CFG batch 2→1. **Filed with its evidence rather than changed quietly**; it
   needs a measured pass over the catalogue, which is a chantier of its own.

3. **`NBX_CENSUS=1 --compiled` is a crash, not a refusal** (`9d3ab07e`). Under a shadow the
   host-availability clamp is deliberately skipped, so the plan is accepted where a real run
   refuses, and compiled mode then **loads weights** — which a shadow must never do. Those are
   mmap'd safetensors on NFS, so an undeliverable page is SIGBUS: no traceback, every buffered
   line lost. Mode 1 needs the door the triton path already has. Until then **R30's mode-1 leg
   is unreachable on this machine** for these containers.

4. **The two capabilities** (`index_put` value broadcasting, the KV-cache decode path). Both
   are real chantiers. **They need scheduling, not a patch.**

---

## What I got wrong, and how it was caught

Recorded because the pattern is the useful part, and it is the same pattern every time — an
instrument that answered a different question than the one asked.

| instrument | measured | I read it as |
|---|---|---|
| `_reshape` at 1536² | a non-integer ratio it could not repair | "the defect does not fire here" — it fires at every integer ratio |
| certifier exit code | the last round had nothing left to do | "certified" — after a 4-second rc=0 following two rc=1 refusals |
| `grep -F` on a key | the string anywhere in the file | "present in the directory" — it was nested inside another object |
| entry field `screen` | a field that does not exist | "unscreened" — the field is `proof`, and all 3 293 pass |
| `grep -l global.image` | the string in `CLI inputs` too | "11 logs still fail" — that line is the SUCCESS case |
| `pgrep \| head -1` | the `timeout` wrapper, RSS 4 MB | "the process is stuck at 0 GB" — the Python was at 10.6 GB, loading |
| `(N MB planned)` | the component **sum** | "a plan accepted above its own clamp" — a lifecycle is checked against the peak |

Two of my own errors, not instruments: I launched the compiled check with `timeout 900`
against a 25-minute weight load, so both sizes would have been killed mid-load and reported
nothing; and I ran the 2048² render beside the census shadows, which cost the first attempt and
is exactly what `recensus_closed.sh` warns about in its own header.

---

## Not obtained

A judged 2048 px artefact. The **transformer** ran to completion on the real path — 4/4 steps,
`addmm M_BUCKET=32768` = 2 × 16 384 tokens — which is where the defect lives, so the fix is
proven under real execution and not only in the shadow. The **VAE decode** at 2048 px exceeds
this 18 GB device: killed twice, the second time with the kernel jettisoning daemons en masse.
1536 px cannot substitute (ratio 2.25, the defect does not fire). The request that would both
fire it and halve the VAE is **2048×1024**, tokens 8192, ratio exactly 2 — worth one attempt
when the link is free.
