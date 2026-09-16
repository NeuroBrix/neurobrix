# The debts, sorted by what they block — 2026-09-13

`DETTE.md` names 113 headings. Read in full on 2026-09-13 22:50 UTC and sorted
into three kinds, because they are not one list:

* **A — blocks a line of the catalogue.** A model the catalogue lists cannot
  run, or its line cannot be measured, until this is closed. Closed in this
  loop, in the order of what each conversion is worth.
* **B — blocks a certification.** A shape the certified directory cannot hold,
  a proof it cannot make, a cell that cannot finish. Closed in this loop.
* **C — a declared limit, or work of another kind.** Stated in the catalogue
  document beside the line it touches, or owned by another chantier
  (serving, build toolchain, owner decisions, instruments). Not a debt of the
  catalogue's; not forgotten either — each has its trigger recorded in
  `DETTE.md`.

A debt marked **closed** in `DETTE.md` is listed once here for the count and
not sorted. A section heading that is not a debt (the É7 pass, the "DEFERRED
items" umbrella, the five sub-headings of the CogVideoX temporal-pad analysis)
is not counted.

## Count

| kind | debts |
|---|---:|
| A — blocks a catalogue line | 19 |
| B — blocks a certification | 5 |
| C — declared limit or another chantier's | 62 |
| closed (recorded in DETTE.md) | 13 |
| headings that are not debts | 8 |
| **named in DETTE.md** | **113 headings** (of which 99 open debts, one filed 09-13 after the sort) |

## A — blocks a line of the catalogue (19), in the order of what they convert

| debt | catalogue line(s) it holds | what closes it | state on 2026-09-13 |
|---|---|---|---|
| D-DEEPSTACK-ZERO-EXTENT | Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Instruct — both engines refuse at the first bind since 09-10 | the build toolchain declares min 0 for an optional input; two re-traces (57 GB each), rebuild, re-upload | filed 09-13; the 4 cells fail identically in rerun 2 and in rerun 3 (09-14 08:24, `46479ae`: `Symbol s6 (seq_len) = 0, below the minimum extent the container declares (1)`) |
| D-DECLARED-MOE-AS-EXECUTED-VIEW | Ming-Lite-Omni-1.5 triton (`moe_fused None.data_ptr`), Qwen3-Omni thinker sized 5.2 GB / executed 57 GB | one as-executed view per component computed at solve, read by every sizer; register 54's loader fix is the first half (`46479ae`) | filed 09-13; rerun 3 on `46479ae` (09-14 08:24): DeepSeek-Coder-V2-Lite native and deepseek-moe native PASS (the OOM at weight load is gone — register 54's loader fix proven by run); Ming triton passes the loader and stops at the next wall: the launcher refuses `embedding_kernel` a device address the allocator never handed out (`0x7fba35100000`) — filed 09-14 as D-MING-UNTRACKED-ADDRESS-AT-EMBEDDING |
| D-MOCHI-CUDA-700-AT-MM → D-PRISM-MOCHI-VAE-ACTIVATION-UNDERESTIMATED | Mochi-1-preview (CUDA 700 at its first matmul, then at add, then at group_norm) | the CUDA 700 is CLOSED at the kernels: three int32 offset wraps past 2^31 elements (registers 58, `90fefd4` `7ee3d7c` `aa60c5c`), named by `--triton-sequential` + launch blocking after two sanitizer runs measured nothing; the run now reaches the VAE decoder and OOMs at `silu::26` (8.75 GB asked, 26 GB live, 32 GB card) where Prism planned 3.2 GB of activations and no tiling — the estimator, filed 09-14 | kernel side closed 09-14 09:05; Prism side open; a bounded proof (9 frames) runs after the GLM wall |
| D-TEMPORAL-UNROLL | Wan2.1-VACE-1.3B, Wan2.1-I2V-14B-480P (inferred), Wan2.2-I2V-A14B (second line) — a VAE encoder unrolled over T, 517 ops per chunk | a symbolic trip count in the build side's loop capture, then three rebuilds | measured 09-13 04:44; build-side, not started |
| D-PRISM-WAN22-TRITON-ONE-CARD | Wan2.2-I2V-A14B triton at default guidance (fits only at cfg 1.0) | Prism's activation estimate for batched CFG on the triton arena (1.3 GB estimated, ~4 GB observed); the two-card plans refused "cannot fit" | filed 09-13 |
| D-WAN-T2V-OOM-AT-5D-PAD / D-WAN-T2V-VAE-ACTIVATION-12GB | Wan2.1-T2V-1.3B (FAILED rc=-9 at 2700 s; 12.4 GB in one allocation at the VAE) | D2's 5D tiling bound for the VAE, or the temporal engine's chunking at the pad | filed 09-09 / 09-11 |
| D-WAN-VACE-BROADCAST-AT-DIV / D-NEGATIVE-ALLOCATION-SIZE-WAN-VACE / D-WAN-VACE-FRAME-TOKENS-FROZEN | Wan2.1-VACE-1.3B (NAMED DEBT line) | the trace bakes the control-token count of its default frame count; fix at the build side, rebuild | filed 09-06/09-09/09-11; no stimulus corrects it |
| D2 (5D-VAE tiling) — Allegro 88 frames | Allegro-TI2V bounded at 80 frames (88 asks 25.27 GiB in one allocation) | cuDNN ≥ 9.3 (owner's install) or a per-tile conv bound | decided 09-13: declared limit on this stack until one of the two lands — the line says so |
| D-ALLEGRO-TI2V-FRAME-TOKENS-FROZEN | Allegro-TI2V at `--num-frames 9` with a reference image (triton) | the build side promotes the conditioning branch's frame extent to the frame symbol; rebuild | filed 09-06 |
| D-ALLEGRO-TRITON-31H-PER-ARM | Allegro (FAILED rc=-9 at 2701 s; 31 h per arm at its own request) | a bounded request per family for the campaign (landed for the retrace gate, `video: --steps 4`); the catalogue cell needs the same bound | filed 09-10; the bound exists, the cell does not use it |
| D-VIDEO-CAMPAIGN-STIMULUS | Allegro rendered at the CLI's default size — colour bands on both arms | the request's default size read from the container, never a CLI constant | filed 09-05 |
| D-ORPHEUS-FT-VENDOR-CODEC / D-ORPHEUS-SEED-NOT-PINNED | Orpheus-3B ("not runnable — catalogue decision") | retire the -ft container for the -snac build (codec baked in, R34), pin the sampler on the executor's RNG stream | filed 09-06 / 09-08; catalogue decision stands until then |
| D-PRISM-SANA4K-COMPILED-16GB | Sana-1600M-4Kpx-BF16 compiled on a 16 GB card (OOM without tiling) | the Prism solve lands on op-level tiling or a two-card placement on a 16 GB profile | filed 09-07; the line is measured on a 32 GB card |
| D-PRISM-2x16-PIPELINE-OVERFILL | GLM-4.1V-9B-Thinking triton on `--gpu 0,1` (22 GB planned on 2×16) | a per-card share that fits, or a refusal by name before the first malloc | filed 09-07; rerun 3 (unpinned): GLM triton TIMEOUT 180 s on a warm retry; the wall without a budget, 09-14 09:07, rig quiet, `46479ae`: run 1 79.6 s (8 shapes swept at runtime), run 2 30.0 s (0 swept, 7 184 keys served certified) — the cell is not slow, its harness budget was spent on a cold sweep (D-GATE-TIMEOUT-VS-COLD-AUTOTUNE) |
| D-QWEN3VL-MOE-RUNS-EVERY-EXPERT-ON-EVERY-TOKEN | Qwen3-VL-30B-A3B-Thinking (48 `topk` at k=8, 0 fused; 128× the expert FLOPs) | the declared-MoE fusion on the VL family (D-DECLARED-MOE-AS-EXECUTED-VIEW's view) | filed 09-09; behind D-DEEPSTACK-ZERO-EXTENT |
| D-DSMOE-XENGINE-SHA | DeepSeek-MoE-16B-Chat triton (degenerate output, drift gate FAIL) | the drift walk on the triton arm against the compiled oracle | reclassified 08-26 as a correctness bug; open |
| D-TRACE-DEEPSEEK-MOE-ILLEGAL-ACCESS | DeepSeek-MoE-16B-Chat re-trace (illegal memory access in the build side's dtype conversion) | `CUDA_LAUNCH_BLOCKING=1` names the kernel; fix in the build side's expert loop | filed 09-08 |
| D-AUDIO-LLM-GRANITE-HOST-PLACEMENT | Granite-Speech-3.3-8B on a 16 GB card (host/device mismatch under offload) | the flow moves its index tensor with the component | filed 09-05; the line is measured on 32 GB |
| D-CPU-COMPLEX-HALF-EXP / D-KOKORO-DECODER-PINNED-HOST-READ | Kokoro-82M on a 16 GB card (lazy_sequential places the vocoder on the host) | complex dtypes treated like floating ones on host placement; a staged component's buffers all move with it | filed 09-05 / 09-06; the line is measured where it fits |

## B — blocks a certification (5)

| debt | what it blocks | what closes it | state |
|---|---|---|---|
| D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE | 184 census keys no run will present again (IEEE_PRECISION / PROMOTE_B of an older engine) | retired from the census 09-13 (`census-retirements.md`, reversible JSON); the certifier counts them apart | closed in effect; the entry stays until DETTE.md says so |
| D-AUTOTUNE-SWEEP-DELIVERY | the measured sweeps reach a user only through a rebuilt container or a store the hub does not serve | the hub serves `/<org>/<name>/autotune/<arch>.json`; the installer places it | filed 09-05 — superseded in part by the certified directory (engine component, 09-06); what remains is the per-model artefact's delivery |
| D-GATE-TIMEOUT-VS-COLD-AUTOTUNE | cells cut by a budget sized before the cold sweep was measured (chatterbox 674 keys, openaudio 693, SANA-Video 25 conv keys at 720p) | a per-cell timeout proportional to the keys the cell sweeps | filed 09-03; three catalogue cells read "no cost" because of it — owner's decision named 09-13 11:27 |
| D-CONTAINER-ENGINE-COMPATIBILITY-UNDECLARED | sixteen matrix cells that no measurement could have produced: the before tree cannot load containers built after it, and nothing declares which engine a container needs | `nbx_version` bumps at a loader-contract change (Forge), the engine declares the range it loads (`info --json`), `NBXContainer.load` refuses by name, a paired harness reads both before spending an arm — R18 is touched (an existing field acquires meaning): owner's decision | filed 09-16 |
| D-SMEM-PRUNE-GAIN-UNMEASURABLE-HERE | the flash-attention tile guard's benefit cannot be measured on this rig | a card whose SMEM the compiler exceeds (A40 class) | filed 09-10; owed by whoever has the card |
| *(new, 09-13)* register 56 — memory-class coverage | every entry proven on one memory class is served to no other: the 32 GB cards sweep until covered; 1 530 rig-proven entries serve no card until re-proven | certification of the 16 GB-proven keys on a 32 GB card (cards 2/3, behind the chain), re-proof of the 1 530 on a 16 GB card | door landed `d1def45`; re-proof on cards 0/1 DONE 09-14 01:39, the 5 628 clockless proofs of 09-07 re-proven at the protocol clock by 07:09, the 32 GB coverage DONE 09-14 21:01 on cards 3 and 2 (matmul 4 644, addmm+baddbmm 3 873, conv2d+depthwise 1 128; every checkpoint pushed to both remotes, last `d639a1b`): directory 9 655 entries — 9 554 covered on 16 GB, 9 593 on 32 GB, 9 492 on both, 19 085 of 19 147 proofs at a recorded clock, gate 0 refused; the catalogue's lines read their shapes per class in the document; the 16→32 sample was ADJUDICATED 09-14 01:47: the two SKUs agree at the locked clock (32/16 median 0.998, 39/50 same config); the 1.175× is the 09-07 proofs' unrecorded clock (5 628 proofs, likely boost, not proven) — re-proven at 1290 on the idle 16 GB cards from 09-14 02:00 (register 56, adjudicated paragraph) |

## C — declared limits and other chantiers' (61)

Grouped by owner, one line each; the trigger to close is in `DETTE.md`.

**Serving (not the catalogue's request path):** D-SERVE-WARM-REFREEZE ·
D-SERVE-DECODE-HOST-BOUND-8K · D-SERVE-WARM-KV-GROWTH-ASYMMETRY (fixed in
tree, gate pending) · P-SERVE-UNLOAD-LIVE-SET · D-KVCAP-LEGACY-FALLBACK ·
D-REQUEST-BEYOND-THE-WINDOW · D-STT-TWO-PROTOCOLS. The Studio requests
(`studio-engine-requests.md`) land on the same surface.

**Performance levers, measured walls (the line runs; the number is the
debt's):** D-TP-HOST-BOUND-COLLECTIVE · D-REPLAY-EXCLUDES-MULTI-DEVICE ·
D-REPLAY-MULTIDEV-NO-TEST-VEHICLE · D-PREFILL-H2D-SMALL-TRANSFERS ·
D-AUDIO-LLM-REFORWARDS-THE-WHOLE-CONTEXT (1.15 tok/s Voxtral, measured) ·
D-WEIGHT-LAYOUT-AT-LOAD · D-ATTENTION-OUTPUT-LAYOUT · D-ROTARY-CACHE-DTYPE ·
D-MOE-DISPATCH-CASTS · D-TRACE-GQA-REPEAT-KV-CLONE · D-STEP-CACHE-VERBATIM-REUSE
· D-PRECISION-LEVER-NOOP-COST · D-PRISM-OVERCARD-MOE · D-PRISM-WEIGHTSHARD-ACT-SMALLCARD.

**Numerical agreement with the vendor (the line runs; correctness is another
instrument's):** D-VIDEO-HORIZONTAL-STREAK-VS-VENDOR (17.5 dB, the first
defect no byte gate could see) · D-ALLEGRO-VERTICAL-SEAMS ·
D-SANA-FP16-DRIFT-BOUND · D-PRECISION-DRIFT-SITE · D-FLEX-ROW-RECIPE (blob on
both arms — the vendor recipe is owed first) · D-UPSCALER-SWIN2SR-X2-CONSERVATIVE-BLACK
· D-KERNEL-REMAINDER-HALF-DISCONTINUITY · D-ZOO-AUDIO-GATE · D-VOLTA-FLASH-PREFILL-NONDET
· D-RNG-31-BIT-SEED-SPACE · D-RNG-DRAW-UNARMED-IN-A-FLOW — **closed 09-14 06:54**: the permanent guard
(`3092fdb`) read 8/8 on card 1 — the four drawing containers, native and triton, two runs each, one
hash; MiniCPM-o's triton speech leg needed its 428 shapes swept once (45 min) because no census had
held them (the catalogue's multimodal request is text mode — a catalogue finding, its line measures
one leg).

**Build toolchain — fixed at the source, rebuilds pending or landed:**
D-TRACE-SUBMODULE-PARENT-FORWARD · D-PARAKEET-SYMBOLIC-T (build-side fix
landed) · D-TI2V-DEGENERATE-BATCH · D-TRACE-DEVICE-INDEX ·
D-TRACE-SYMBOLIC-DIMS-FOREIGN-INT (fix at source de82f60) ·
D-TRACE-INPLACE-VIEW-ASSIGN · D-PROFILE-HEAD-DIM-UNSTATED ·
D-ROPE-VALSYM-BORN-AT-SOURCE · D-RETRACE-SWIN2SR-SYMBOLIC (a trace-value
collision, the line runs) · the CogVideoX-5b-I2V temporal-pad analysis (its
five sub-headings; the container was re-traced and PROVEN 09-12) ·
D-MOE-DECLARATION-ON-THE-IMAGE-GEN-ENTRY.

**Engine purity and portability (R33/R34/R23):** D-CORE-MODULE-INIT-TORCH
(fixed 09-02) · D-R33-COMPUTABLE-BUFFER-TORCH-FALLBACK · D-CROSS-BRANCH-COMPILED-ONLY
· D-PRECISION-CONTRACT-TRITON-PARITY · D-D128-DETOUR-PORTABILITY ·
D-NVIDIA-PROFILE-GAP-TURING-ADA · D-CPU-PLACED-COMPUTE-HALF-COVERAGE ·
D-COMPLEX-CONSTANT-PROMOTES-TO-COMPLEXHALF · D-GATHER-SCATTER-OOB-SILENT
(the kernels trap since 09-02 23:15).

**Instruments, tests, workshop:** D-EVICTED-SEED-LOST-ON-KEY-OF-FAILURE (filed 09-13
from the review of the memory-class door: a seed evicted at load is not put back when the
launch site cannot compute the key; and two unlocked writers of a directory file) · D-THIRTY-THREE-RED-UNIT-TESTS (21 on
09-13, classified; rerun from a frozen tree pending) · D-TEST-PINNED-ORDINAL ·
D-OBSERVABILITY-BLIND-SITES-ACROSS-FAMILIES · D-RENDER-RESUME-NOT-BIT-IDENTICAL
· D-TSEQ-ORPHEUS-STEP110 · D-AUTODETECT-VISIBLE-MASK · D-CUDA-DEVICE-ORDER-UNPINNED
· D-VACE-DEGENERATE-MERGE / D-VACE-UNROLLED-CHUNKS (enriches A's VACE entries;
the correction is the same trace) · MAINT-HUB-STORAGE-V3-FILENAME ·
D-AUDIOLLM-LONGFORM (Voxtral class delivered, gate pending).

**Owner decisions (named as such in DETTE.md):**
D-PRECISION-CONTRACT-DEPLOYMENT-SPLIT · D-PRISM-COMPONENT-FP32-FLAGS ·
D-R33-TRITON-NOISE-STREAM · D-GITLAB-RELEASE-HISTORY-PARITY · D7 (delivered,
listed under DEFERRED for the record).

## Closed, recorded in DETTE.md (13)

D-KOKORO-NONDETERMINISTIC-VOCODER · MAINT-FLIGHTREC-LOOP3-POWERLOSS ·
D-SANA-VAE-FP16-BLACK · D-REGISTRY-LOOKUP-BY-DIRNAME ·
D-TRITON-RESUME-GUARD-NAMEERROR · D-STT-KV-WHISPER-LARGE ·
D-RETRACE-VIDEO-REQUEST · D-DECODE-PROGRESS-NOT-UNIVERSAL ·
D-REPLAY-BLIND-TO-THE-ENGINE-LAUNCHER · D-SCHEDULER-EXCLUSIVITY-ONLY-AT-ENTRY
· D-CAMPAIGN-ABSOLUTE-DEVICE-INDEX-UNDER-A-PIN · D-CPU-NO-HALF-OPS-IS-A-CONSTANT
· D-IMPORT-RESUMABLE-DOWNLOAD.

## The week's list, as the owner gave it on 2026-09-13 01:04, where each stands

| named | kind | state |
|---|---|---|
| D-TEMPORAL-UNROLL | A | measured (Wan2.2 vae_encoder, 517 ops/chunk); the fix is on the build side, not started |
| D-MOCHI-CUDA-700 | A | under compute-sanitizer, rig quiet, since 21:26 |
| Allegro D2 | A (declared limit) | decided: cuDNN 9.1 refuses the large non-batch-splittable conv; 88 frames is a declared limit until cuDNN ≥ 9.3 or a per-tile bound |
| Kokoro determinism per mode | closed | measured TRUE on the trunk 09-13 01:23 (three modes, one sha each) and CLOSED 09-14 06:54 by the permanent guard (`3092fdb`): 8/8 — the four drawing containers × native/triton, two runs each, one hash (`rng_guard_0914*/VERDICT.md`) |
| the §4.2 screen cell (A40) | B | the production wiring is the live screen's oracle (register 45-47); the sm_86 cell itself is owed by whoever has the card |
| `--explain-plan` | — | delivered (`d9ba351`, `06004e2`) |
| `budget-unified` and its byte question | — | five pinned pairs byte-identical across a strategy change; the per-family matrix ran 09-14 09:11-15:12: 30 cells IDENTICAL (Kokoro re-run with the voice it requires), 2 UNADJUDICATED on the two models whose nondeterminism is on record (orpheus-ft, CogVideoX-2b), 0 adjudicated differences, 16 UNMEASURABLE on this pair (the before tree 5ca23b1 cannot load today's containers) — `budget_unified_gate_20260913_1535/RUN.md` |
| D-IMPORT-RESUMABLE-DOWNLOAD | — | closed `8da8d1b` |

## How this document is kept

A debt moves between kinds only with the measurement that moves it, dated.
A closed debt leaves its row with the commit. A new debt is filed in
`DETTE.md` first and appears here at the next sort; this document never
carries a debt `DETTE.md` does not.
