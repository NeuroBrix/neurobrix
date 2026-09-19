# Proofs one machine owes another

A commit whose claim can only be checked on hardware the author does not have is
not finished when it is written. It is finished when the machine that has the
hardware returns the proof. This file is where those debts and their answers
live, because they cross a machine boundary and therefore cannot live in either
machine's local notes.

**One entry per owed proof. Append, never renumber. A proof that came back
NEGATIVE stays here with its result — that is the most valuable kind.**

---

## 1 — `f769f2e`, the second fault channel: CUDA proof owed to the Dell

* **owed by** the Mac's agent · **returned by** the Dell, 2026-09-12
* **the commit** is on `metal-first-light`, origin and gitlab. **Not on the trunk.**
  This proof is what authorises its merge.
* **what the Mac established there** `tl.device_assert` reaches the IR only under
  `debug=True`, and the Metal backend elides it anyway — it computes the predicate
  and discards it. Measured cost on the three armed kernels (2026-09-02):
  `index_select` writes nothing and leaves the pool's residue, `embedding` reads
  four floats past the weight, `index_put` WRITES eight floats past the tensor.
  The second channel is a constexpr `FAULT_CODE`, non-zero only where the assert
  is not honoured.
* **what it could not establish there** that on CUDA, where the assert IS
  honoured, the fix is really inert. Its own commit message says so:
  *"CUDA proof owed to the Dell."*

### What came back

**The guard holds.** Cell 1 — the only cell that can veto, and it runs first —
passed `rc=0` over six tests: each kernel still refuses an out-of-range index **by
its own name**, and the in-range control still equals torch. The second channel
has not disarmed the first. (The Mac noted that case passed there for the WRONG
reason — numpy raised on the oracle before the kernel was asked. On a card it is
the kernel that answers.)

**The bytes are identical.** Cell 5: one fingerprint, `1816336e4cc3…`, across both
arms and three interleaved repetitions.

**The channel is disarmed on CUDA.** Cell 2: assert honoured `True`, `FAULT_CODE`
armed `0`.

### And one finding, which is this proof's substantive result

**"Inert" is false on the allocation axis — the axis the merge argument rests on.**

After a complete `--triton` run of `TinyLlama-1.1B-Chat-v1.0`, the census reports
`_FAULT_BUFFERS` holding one entry, key `2`: **a fault buffer was allocated on
`cuda:2`**, while cell 2 establishes the code on CUDA is `0`.

`device_fault_buffer(device_idx)` allocates on first call and is **not gated on
the fault code**. With the code at 0 the kernel's `tl.store(fault_ptr, FAULT_CODE)`
sits under a constant-false condition and is eliminated at compile time, so the
pointer is never dereferenced — **this is not a correctness defect**. But the
buffer is born, it is one `int32` per device, and the contract says it is **never
freed** (a launch records the raw pointer and a frozen replay plan may hold it).

Four bytes per device is not worth a chantier. The discrepancy between the claim
and the artefact is, because "inert" is the whole argument for merging this into a
path that never arms it. **The shape of the repair is one condition at the call
site: take the buffer only when the code is non-zero.** That is the Mac's agent's
call, on the Mac's agent's commit, and the Dell has not touched it.

### What is still owed, and by whom

* **The PTX comparison — repaired and RETURNED the same day. It is stronger than
  the commit claims.** Cell 3 first could not run: the harness typed every pointer
  `*fp32`, including `index`, which is a tensor of integers — so
  `tl.load(index + …)` produced float32, `rows_offsets * N + indices` became
  float32, and `inp + inp_off` added a pointer to a float
  (`IncompatibleTypeErrorImpl`). Both arms failed identically, so the cell could
  say nothing about the commit; it was the Dell's excerpt that did not compile.
  Types are now READ from the kernel, not guessed, and an untyped pointer is a
  refusal rather than a default.

  Compiled for `sm_70`, `FAULT_CODE=0`, both arms:

  | | parameter declarations | mem/pred instructions |
  |---|---:|---:|
  | before (`2f69ea8`, 8 args) | 6 | 985 |
  | after (`f769f2e`, 10 args) | **6** | **985** |

  **62 PTX lines differ, and every one is debug metadata**: 54 `.loc` line-number
  directives (the source moved, the commit added lines above), 6 `.b8` and 2
  `.file`. Zero `st.global`, zero `red.`, zero `atom.`, zero `bar.sync`. The one
  occurrence of the string `fault` in the generated PTX is the worktree's own path
  inside a `.file` directive — `faultproof_f769f2e` — and not a code reference.

  So the claim *"unchanged bar one unused kernel parameter"* **understates it on
  CUDA**: with the code at 0 the parameter is not even emitted. The generated code
  is identical.

  That makes the cell-4 finding sharper rather than softer: since the kernel does
  not take the pointer at all on this backend, the buffer the host allocates for
  it is consumed by nothing.
* **The timing says nothing, and could not.** Cell 6's medians are 46.689 s
  (after) against 46.789 s (before) — a difference of 0.100 s, where the spread
  within one arm alone is 2.60 s, twenty-six times larger. This cell cannot
  resolve a cost of that size. It is also weakened by a condition of the Dell's
  own making: unit suites and git operations ran on the host while cell 5 was
  measuring. The arms are interleaved rep by rep, which is the design that blunts
  host noise, but the doctrine says quiet host during a locked bench and it was
  not quiet. Recorded rather than deduced later.

### Verdict on the merge

Nothing here blocks it **on correctness**: the guard is intact and the output is
byte-identical. What is not yet true is the **claim the commit makes about
itself**, and it is now down to ONE item, which needs no card: **gate the buffer
on a non-zero fault code.** The PTX comparison that was owed has been repaired and
returned above, and it came back better than the claim.

The merge is the owner's decision, taken knowing that the second channel costs one
unfreed `int32` per device on a backend whose generated code does not reference it
at all.

Full cell-by-cell record, on the Dell:
`validation_outputs/…` → `nbx/campaigns/prepared/cuda_fault_channel_20260912_1219/VERDICT.md`.

---

## 2 — how the certifier reads device memory: answered for the Mac, 2026-09-12

* **asked by** the Mac's agent · **answered by** the Dell, same day, by reading
  the code rather than reasoning about it.
* **the question** its own oracle reads device memory by two routes — `t.numpy()`
  for most dtypes and `ctypes.string_at` on a device pointer for another, without
  asking whether that memory is host-addressable — and it suspects the second
  route is falsifying its refusals. Does the Dell's certifier do anything of the
  kind, for any dtype? Its 7 158 certified entries rest on the answer.

### The answer: no, and by construction rather than by luck

**One read path, all dtypes.** `autotune_certify` reads a produced buffer in
exactly one place — `out_tensor.numpy()` at the deviation site. No `string_at`, no
`from_address` on a device pointer, anywhere in the certifier.

**That path copies to the host unconditionally before touching a pointer.**

```python
f = self.contiguous()
if f._device != 'cpu':
    f = f.to_cpu()                      # <- the copy, not a question
buf  = (ctypes.c_uint8 * nbytes).from_address(f.data_ptr())
view = np.ctypeslib.as_array(buf).view(np.dtype(typestr)).reshape(...)
out  = view.copy()
```

The raw read exists, and it only ever addresses HOST memory. "Is this
addressable from the host" is answered by the line above it, not assumed.

**The dtype table is complete, so there is no silent misread.** Every `NBXDtype`
has an entry, `bfloat16` included as `'<V2'` — an opaque 2-byte view, not a
reinterpretation as fp32. The `'<f4'` default in `_DTYPE_TYPESTR.get` is
unreachable for a real dtype. And on this profile no bf16 buffer is certified at
all: the eight files are fp16 and fp32.

### And a second failure mode of `string_at`, which the Dell has already paid for

Worth more than the answer above, because it produces the symptom being
suspected — **false refusals** — and it has nothing to do with addressability.

`ctypes.string_at(ptr, n)` hands `n` to `PyBytes_FromStringAndSize` as a **C
int**. Any buffer of 2 GiB or more therefore comes back as

```
Negative size passed to PyBytes_FromStringAndSize
```

On 2026-09-07 that made **seven census shapes report "no config could run"** — a
4K convolution, a 1 221 120-row matmul, a 16384² baddbmm. Every configuration was
fine; the READBACK was failing, and the certifier recorded the failure against the
kernel. The comment now standing at `NBXTensor.numpy()` names those seven shapes
so the route is not taken again.

**So: check the size of the buffers your refusals concern.** If the refused ones
skew large, the suspicion is right but the mechanism may be this one rather than
host-addressability — and the two are distinguishable in one line, by reading a
1 GiB buffer and a 3 GiB buffer through the same path.

### What is NOT established here

Whether the Mac's own second route has the addressability problem it suspects.
This answers only what the Dell does, which was the question asked. The two
engines share the doctrine, not the code path.

---

## 3 — the screen oracle's coverage on this rack: answered for the Mac, 2026-09-12

* **asked in** `docs/reference/trunk-arbitration-list.md` item 1 · **answered by**
  the Dell, by reading the Mac's own table on `origin/metal-first-light` and
  counting this rack's certified directory. No card was needed.
* **the question** *"a read of the provider's coverage table (`ORACLES`, currently
  `mm` and `baddbmm`) against what the screen is asked for on this rack, and a
  decision about what `announce_no_oracle` should do when the answer is 'most
  kernels'."*

### First, the table names three kernels, not two

`kernels/screen_oracle.py` on `metal-first-light`:

```python
ORACLES = {
    "matmul_kernel":  (_mm, "c_ptr"),
    "addmm_kernel":   (_mm, "c_ptr"),
    "baddbmm_kernel": (_baddbmm, "out_ptr"),
}
```

`addmm_kernel` IS covered. That matters for the argument the Mac made for the
oracle: its measured blind spot — four `addmm` shapes where the emitted MSL
declared `alpha`/`beta` as `int`, every candidate wrong the same way, the vote
unanimous, the bare screen seating a wrong configuration every time — is a case
the provider DOES cover, which is why the same screen with the oracle refused
every time. The arbitration note understated its own evidence.

### The coverage, counted against this rack's 7 158 certified keys

| kernel | dtype | keys | oracle |
|---|---|---:|---|
| `matmul_kernel` | fp32 | 3231 | covered |
| `baddbmm_kernel` | fp32 | 2351 | covered |
| `addmm_kernel` | fp32 | 749 | covered |
| `conv2d_forward_kernel` | fp16 | 434 | **none** |
| `conv2d_forward_kernel` | fp32 | 336 | **none** |
| `depthwise_conv2d_kernel` | fp16 | 33 | **none** |
| `depthwise_conv2d_kernel` | fp32 | 19 | **none** |
| `baddbmm_kernel` | fp16 | 5 | covered |

**6 336 of 7 158 keys are oracle-covered — 88.5%. The 822 that are not (11.5%)
are exactly the convolution family**, and nothing else.

### Which changes the decision the question was asked for

`announce_no_oracle` was scoped against the possibility that the honest answer
was "most kernels". It is not. It is ONE family, it is the family the autotune
policy admits for the same reason it admits matmul (conv2d is in the sanctioned
scope precisely because it is where Triton needs tuning), and a float64 direct
convolution is a well-defined thing to write — slow, which does not matter for an
oracle that runs once per shape at certification.

So the two options are both small, and they are not equivalent:

1. **Write the conv oracle.** 822 keys move from "screened by consensus" to
   "verified against fp64", and the directory's claim becomes uniform.
2. **Make `announce_no_oracle` REFUSE to seat a configuration** rather than fall
   through to the bare consensus screen. This is the doctrinally consistent one
   while (1) does not exist: the bare screen is exactly what the Mac measured
   seating a wrong configuration unanimously, so falling back to it on the
   uncovered 11.5% is falling back to the known-failing instrument.

The Dell's recommendation is **both, in that order of value and the reverse order
of urgency**: (2) today, because it costs one branch and closes a path that is
known to seat wrong answers; (1) when someone has an afternoon, because it is what
makes the 11.5% a measurement rather than a vote.

**What is NOT established here**: whether those 822 conv keys are wrong. This
counts what the oracle would be asked and cannot answer; it does not run the
screen. The 7 158 entries were certified against this machine's own fp64 oracle at
certification time — `autotune_certify` has always used one — so this is about the
RUNTIME consensus screen, which is a different instrument with a different
coverage.

### Addendum 2026-09-13 — the convolution oracle is in the live screen, and the budget is now the boundary

The oracle written for this gap (`kernels/oracles/conv2d_fp64.py`) was never
joined to the live provider's table (register entry 46), and once joined it
never saw its constexpr arguments (they are launch kwargs, not positional
arguments — same entry); and the screen itself looked at one shape in ten
(register entry 47). All three were found by running the family live on card 0
(`real-esrgan-x4`, ten conv2d sweeps, isolated replay cache), not by the suite,
which was green throughout.

With the three repaired, what the live screen does on those ten keys is now
measured, and it is the profile's byte budget that decides: **one key (3→64
channels at 448², ≈26 MB of arguments) is adjudicated by the fp64 oracle and
recorded `screened: true`; nine (38 MB to 822 MB of arguments) are over the
profile's screening budget of 33 554 432 bytes and are announced UNSCREENED,
recorded `screened: false` with that reason.** The output image is byte-identical
across the four sweeps (pre- and post-repair), which is what a screen that
changes provenance and not choice should show.

So on this rack the convolution family's LIVE coverage is a function of the
budget, not of the oracle any more. Raising `autotune_screen_max_bytes` in the
Volta profile would extend it at the price of an fp64 numpy convolution per
candidate over hundreds of megabytes — a measurement to make before moving the
number, not a number to move. Certification has no such budget and covers the
family entirely (every conv entry in the directory carries the fp64 proof).

---

## 4 — the engine stack both machines align on, 2026-09-16: torch 2.14.0 (cu126 on the rack) + Triton 3.8.0, Python 3.10

* **owed by** the Dell (this rack) · **to** the Mac: align the engine environment on it;
  the Mac's only divergence is the Triton version triton-ext pins for Metal, dictated upstream.
* **what was measured here.** The CUDA door (`tools/stack_door.py`): `torch==2.14.0+cu126`
  from `https://download.pytorch.org/whl/cu126` embeds CUDA 12.6 with archs
  sm_50…sm_90 (sm_70 present) and sees the four V100s with their capabilities —
  ACCEPTED. The PyPI `torch==2.14.0` wheel is the cu130 build (no sm_70) and
  cu130/cu132 carry sm_75+: refused by the door. cu126 is the only cu12x of 2.14,
  and 2.14 is the last PyTorch release with any CUDA 12.x wheel and the last with
  Python 3.10 (2.15, 2026-10-28, drops both) — this rack's stack is terminal on
  both axes at 2.14; what comes after is a source build against a 12.x toolkit,
  or another rack. torch 2.14 requires `triton~=3.8.0` and installs `triton==3.8.0`.
* **Triton 3.8.0, retained as the wheel torch pins — not 3.8.0 + cherry-picks, not
  "3.8.1".** Facts: issue triton-lang#11735 names eleven correctness fixes on
  `main` absent from `release/3.8.x` (the branch is 25 ahead / 551 behind main since
  2026-06-23); the six PRs the reporter opened against the branch are all open with
  no maintainer reply; no `[v3.8.1] Release Tracker` exists and Triton's RELEASE.md
  says patch releases are optional — the 2026-10-21 date is PyTorch 2.15's GA minus
  one week, stated nowhere as a 3.8.1 commitment. The cost of the picks: a source
  build of Triton (LLVM prebuilt fetched, wall time unmeasured by any primary
  source), thirteen upstream commits carried as a local fork (R25), and a stack the
  other machine cannot reproduce from an index. The house kernels use none of the
  constructs nine of the eleven fixes touch (`tl.softmax`, block pointers,
  `tl.range(flatten=True)`, `tl.histogram`, `tl.gather`, `dot_scaled`, constexpr
  list comprehensions with `if`, `tl.full(-0.0)`, TMA); the two that could reach
  them silently — #11186 (`OptimizeThreadLocality` reordering reductions) and
  #10353's `tl.dot` out-dtype change — are exactly what the kernel suite (928 tests
  against torch), the certification screen (every setting against the fp64 oracle)
  and the full battery measure. **So the wheel is the stack; the gates are its proof;
  a source build with the picks is taken only if a gate reads a wrong result of
  that class, and then said by name.**
* **what 3.8 brings that we consume.** `knobs.autotuning.listener` (#10125): chosen
  config, per-config timings, duration, disk-cache hit for every `@triton.autotune`
  — our four autotuned kernels (mm, bmm, addmm, conv2d) get it as the observability
  seam of certification instead of a rewrite. Deterministic JIT cache keys (#10494):
  the on-disk Triton cache is invalidated once. `kernel_unload_hook` (#9444):
  harmless for the NeuroBrix launcher, which loads its own modules through
  `cuModuleLoadData`. Under 3.7+ the CUDA driver probe is native when torch is
  absent (#9578/#10935): the R33 proof (`sys.modules` without torch after a
  `--triton` run) is re-run on the new stack.
* **THE BATTERY'S VERDICT, 2026-09-17 05:21 — RED, and the switch waits.** 5 failed, 79 passed
  in 1 h 04, from the frozen worktree at `dd120774` whose directory had just been re-proven to
  100 % on both memory classes. Two of the five are `D-DEEPSTACK-ZERO-EXTENT` (Qwen3-VL, the
  two 57 GB re-traces still queued) and are not the stack. **Three are one defect and it IS the
  stack**: the warm serving path of vlm, multimodal and image refuses `aten.bmm::0` a device
  address the allocator never handed out. Same tree, same cell, only the interpreter changing,
  three repetitions per arm alternating: 2.14.0+cu126 / 3.8.0 fails 3/3, 2.5.1+cu121 / 3.6.0
  passes 3/3. Cold passes and warm fails on the same models. The allocator's segment mode is
  refuted as the cause. Filed `D-WARM-COMPILED-BMM-ADDRESS-REFUSED-UNDER-TORCH-2.14`; the
  current stack stays in force and the candidate stays beside it, which is what this order is
  for. Verdict: `nbx/campaigns/2026_09_16_converge/BATTERY_T38_VERDICT.md`.
* **the order, unchanged**: door ✓ → the stack in `venvs/nbx_t214` beside the current
  one ✓ (engine installed; first fix already needed and landed: the engine's
  `libcudart` loader opens the environment's own runtime first — the system's 12.2
  broke torch 2.14's import) → the full battery on it (kernel suite first) → the
  re-proof of the certified directory in a frozen tree (the old directory served
  until the new is complete; each proof names its generator — `autotune status`,
  the catalogue's *proven under* column) → the switch. Nothing is erased before the
  new is proven.
* **first measurements on the candidate stack**: launcher, source gates, tensor
  suites 19 passed; TinyLlama bytes IDENTICAL on old and new stack on both engines
  (sha `c70888b8e01d`, 0 sweeps — the certified directory serves at the same keys
  under 3.8.0); the triton cold run 21.7 s vs 8.1 s on the old stack (a fresh JIT
  cache, to re-measure warm).
* **returned by** the Mac when its engine environment carries torch 2.14.0 + the
  Triton triton-ext pins — say the versions here.

### Addendum 2026-09-16 15:5x — the owner's correction, and what the door measured

The correction said: CUDA 13 begins at Turing; PyTorch 2.11 removed Volta from
its cu128/cu129 binaries (cuDNN 9.15.1 no longer serves a V100); **the last torch
that sees a V100 is 2.10 in cu126**; measure the bundled ptxas before Triton 3.8.
The door was extended with the half the arch list cannot see — one cuDNN
convolution and one cuBLAS matmul RUN on every card, read against the CPU's —
and answered on this rack:

* `torch 2.14.0+cu126`: archs sm_50…sm_90 (sm_70 present), cuDNN **9.10.2**
  (`91002`), convolution and matmul run on all four V100s with results equal to
  the CPU's (max |diff| 1.3e-5 / 0.019 fp16) — ACCEPTED. The correction's facts
  hold for the cu128/cu129 builds and not for the cu126 wheel, whose metadata pins
  `nvidia-cudnn-cu12==9.10.2.21` through 2.14 (the build table on `release/2.14`
  keeps sm_70 for 12.6). So on THIS rack the last torch that sees a V100 is
  2.14.0+cu126, not 2.10 — 2.15 ships no cu126 wheel at all. The 2.10.0+cu126
  wheel is installed beside it and put through the same door, so both numbers
  stand in the record.
* Triton 3.8.0's bundled `ptxas` is CUDA **12.9** (`V12.9.86`), not 13; it compiles a
  `.target sm_70` PTX; the kernel suites and TinyLlama already ran through it on
  the V100s. Volta is not out of Triton at 3.8.
* Reformulation taken: the 3.6.0 proofs are **not invalid** under 3.8 — the oracle
  proved the source, not the compiler; only the rank as the fastest may age by a
  few percent. The re-proof is an optimisation pass on this rack, incremental
  (`--reprove-generator` skips what the running generator already ranked),
  checkpointed, invisible to a request; the document reads each rank with its
  generator. The wording in the CHANGELOG, the tool and the document was changed
  to say so.
* Measurement over directive, said: the target stays **torch 2.14.0+cu126 +
  Triton 3.8.0** unless the owner, reading this, holds 2.10 for a reason the door
  does not measure — the switch itself waits on the full battery and the
  re-proof, as ordered, so nothing is committed by this choice yet.

---

## 5 — the Metal seam fold: proof of inertness owed to the Mac by this rack, 2026-09-16

* **owed by** the Dell (this rack) · **when** the fold lands on `main` by merge.
* **what the Mac is folding.** Three trunk files name the Metal fork today —
  `kernels/autotune_refusals.py:58` imports `MetalNonRecoverableError` into the
  shared refusal module, `triton/metal_driver.py` imports `MetalBackend` by
  name, `triton/metal_backend.py` iterates two possible providers. The fold:
  the engine targets Triton, the backend is chosen by the profile, and the
  provider's exceptions are translated into our own refusal type before they
  reach `autotune_refusals.py`, which then names no vendor.
* **the proof this rack returns.** The fold is declared INERT on CUDA, so it is
  proven where it is declared: from a frozen worktree at the merge commit, on
  this rack's V100s — (1) the kernel suite (`tests/unit/kernels`,
  `tests/unit/nbx_tensor`, `tests/unit/runtime`) before and after, same count;
  (2) a byte pair on four models across the merge, cold, `--triton`, outputs
  compared byte for byte; (3) `neurobrix autotune check` on the certified
  directory, 0 refused. The three numbers go in this entry with the commit.
  A refusal path that changes shape is not inert, whatever the tests say: the
  entry will name what the refusal module answers on a CUDA compilation error
  before and after.

## 6 — real-esrgan-x2 re-traced at the stimulus its own extents clear, 2026-09-16

* **owed by** the Dell (this rack) · **when** a card frees between campaigns; not urgent and not
  a class proof either way.
* **what was measured.** The stimulus-collision guard (`cad67f7`) reads a model's architecture
  extents off the model. Over the ten cached upscalers at the family default 112x80, nine clear
  and `real-esrgan-x2` does not: its width times its own scale, **80 x 2 = 160**, is the RDB
  dense-concat width (64 + 3x32) carried by 69 of its convolution weights. The guard would trace
  it at **112x144** instead; the nine others are returned unchanged and re-trace byte-identically.
* **what it does NOT mean.** The container on disk was proven the same evening at four sizes —
  96x96, 160x112, 208x144 and the 448x448 that first exposed the white square — and the symbol
  census reads 0 breaks and 0 never-carried on it. The collision is a LATENT ambiguity, not a
  live defect: nothing in that graph is frozen today. What is wrong is the epistemic status —
  the render is correct by luck rather than by construction.
* **the proof this rack returns.** A re-trace at 112x144, the symbol census re-read on the new
  graph (expected: still 0 and 0), and the same three non-trace sizes judged again, so the line
  moves from *correct* to *correct for a reason*. Neither container is published, so nothing on
  the hub waits on it.

## 7 — WITHDRAWN: "the 1 168 entries the re-proof will never reach" was wrong, 2026-09-17

**The claim is retracted. They were reachable and they drained.** At 22:24 the unknown-card
bucket held 1 168 entries, 725 of them in `matmul_kernel.fp32.json`. At 02:55 the same file held
**4** and the directory held **223**. The pass reaches them; it had simply not reached them yet.

**How I got it wrong, since that is the part worth keeping.** I had two readings six minutes
apart showing no movement, and a control that looked decisive: of 725 unknown-card keys present
in the 32 GB side tree, **0** were re-proven there while **1 374 of 3 920** known-card keys in
that same tree were. I treated zero-against-thirty-five-per-cent as an attribution.

It was an ORDERING artefact. A certifier walks its key space in an order, and the legacy
unknown-card keys sit together in it — so at any moment before the walk reaches them they are
uniformly absent from the done set, and a snapshot of a contiguous region reads as a property of
the region. **I had named that exact confound four hours earlier**, on the question of whether
running the test suite beside a certifier moved its rankings: *"a time window is a contiguous
slice of shape space, because keys are certified in order — so the comparison is confounded by
construction"*. I applied it there and then walked into it here.

The control that would have settled it: read the SAME bucket twice with enough time between the
readings for the walk to move — which is what the morning did by itself. Six minutes was not
enough and I should have said so rather than concluding.

**And there IS a real unreachable set — it is 184, not 1 168, and it was already named.** Card 1
closed its own kernels at 02:57 with `rc=0` and this line: *"71 census key(s) are unreachable to
this engine — the debt `D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE`, not a failure."* That
debt (filed 2026-09-11, 184 of 6 277) is keys an OLDER engine's wrapper computed differently, so
no run presents them again; they were retired from the census on 09-13 and the certifier counts
them apart. So the engine already knew which keys it cannot reach, said so at the end of the run,
and the number is two orders smaller than the one I invented for it.

**What remains true and is now the only open part.** 223 unknown-card entries remain in the main
tree and 1 317 in the side tree. The 16 GB class is at 99.1 %, so those 223 sit inside the last
0.9 % of the pass. Whether the bucket reaches zero is answered by reading it when the pass ends,
not before. `tools/reproof_coverage.py` is the instrument and its `?` row is the number.

## 2026-09-18 — the Mac's three handed-over models, measured on CUDA

The Mac handed over `hat-l-x4`, `hat-s-x4` and `canary-qwen-2.5b` on 2026-09-18
(`models/_agents/for_the_dell_cuda_proofs_owed.md`), with the question, for
hat-l-x4, of whether this is *an estimate that is wrong* or *a model that machine
cannot hold*. Measured here on one 16 GB V100, Triton mode, from the
`runtime_values` worktree.

**All three run.** hat-l-x4 rc=0 in 22 s, hat-s-x4 rc=0 in 12 s,
canary-qwen-2.5b rc=0 in 33 s.

| model | Prism plan | held (`peak_driver`, pool on) | short by | artefact, judged |
|---|---|---|---|---|
| hat-l-x4 | 3383 MB | 12783 MB | **3.78x** | 1792x1792 PNG, looked at: a sharp, coherent x4 upscale — correct colour, skin speckle preserved, clean edges |
| hat-s-x4 | 3245 MB | 12907 MB | **3.98x** | same, judged the same way |
| canary-qwen-2.5b | 6447 MB | 12073 MB | **1.87x** | transcription **matches `benchmarks/assets/jfk_11s.expected.txt`** word for word — an instrument outside the engine, written before the run |

**The answer to the question: it is the ESTIMATE, and hat-l-x4 and hat-s-x4 are
the same defect, not two different ones.** The plan is short by a factor of
roughly four for both. A model that runs to completion in 22 s on a 16 GB card is
not a model that cannot be held; what cannot be held is the gap between 3383 MB
promised and what execution actually takes. On a 24 GiB machine under a 4096 MB
floor, a plan of 3383 MB is allowed to start and then meets its real need — which
is exactly the shape of `rc=42 at 1567 MB with zero autotune misses`.

**The drained figures, which are the comparable ones** (`NBX_ALLOC_POOL=0`, deferred
queue at a 64 MB floor — how the 1.41x-2.27x residue was measured). The pool-on
numbers above include up to ~8 GB of free-list cache (`pool_peak` 8050 / 7944 /
8076 MB) and are a watermark of bytes taken from the driver, not of the live set:

| model | plan | held, drained | |
|---|---|---|---|
| hat-l-x4 | 3383 MB | **7552 MB** | **2.23x short** |
| hat-s-x4 | 3245 MB | **7361 MB** | **2.27x short** |
| canary-qwen-2.5b | 6447 MB | **3814 MB** | **1.69x OVER** |

The two HATs land inside the residue band already measured on this rack, at its
top. **canary-qwen-2.5b is the first model measured here where the plan is
GENEROUS**, and that is worth as much as the shortfalls: the residue is not a
uniform scaling factor that could be corrected with a multiplier. An
over-estimate is the safe direction for a crash and the wrong direction for
placement — it can push a model to a heavier rung of the cascade than it needs.

So the answer to the Mac's question, in its comparable form: hat-l-x4 holds
**7552 MB** against a **3383 MB** plan. Under a 4096 MB floor that is exactly why
it stops, and the 16 GB card runs it in 22 s. The estimate is the defect.

Plans read with `--explain-plan`: hat-l-x4 weights 79 MB + activations 3144 MB +
overhead 161 MB, peak at `aten.add::27`, **no tiling planned**; hat-s-x4 weights
19 MB + activations 3071 MB + overhead 155 MB, same peak op, no tiling;
canary-qwen-2.5b three components on one card, peak at `aten.mm::196`.

Note that the large and the small HAT are given activation figures 73 MB apart
(3144 vs 3071) while their measured watermarks differ by 124 MB in the other
direction. The estimator is not distinguishing them.

## 2026-09-18 — the shared-cache install hazard: FIXED here, nothing owed back

Item 7 of `for_the_dell_cuda_proofs_owed.md`. `c40abf30` on `main`, both remotes.
One brick, `src/neurobrix/nbx/atomic_install.py`, used by `cli/commands/registry.py`
and by `NBXCache.extract`:

* a per-model lock taken with `os.mkdir` (atomic on NFS), which **refuses** rather
  than waiting or breaking, naming the holder's host, pid and age — and never
  judges a lock from another host stale, since we cannot see that machine's
  process table;
* a staging directory carrying host and pid, so neither machine can delete the
  other's work;
* **two renames instead of a removal** — the live tree is renamed aside, staging is
  renamed in, the aside is deleted afterwards — because POSIX `rename(2)` refuses
  to replace a non-empty directory, which is why both call sites used to delete
  first.

Reading it turned up a third race the report did not name: **`NBXCache.extract` did
not stage at all**, removing the live tree and unpacking in place at the final
name, where a `manifest.json` exists from the first member on. The injection that
restores that behaviour shows the final directory at **47 distinct partial sizes**,
each one a moment another machine would have loaded an incomplete model.

Eleven gates, each seen failing on a named injection.

**What the Mac will see**: an import of a model this machine is already installing
now fails with a named refusal instead of interleaving — that is intended.
Directories named `<model>.installing.<host>.<pid>`, `<model>.lock` and transiently
`<model>.replaced.<host>.<pid>.<stamp>` may appear beside models in the shared
cache; none is a model and none is listed as one. One left behind by a crashed
install on the Mac's own host is cleared by its next install of that model; one
left by this machine is not, by design.

## 2026-09-18 — the Mac's radix sort, proven on CUDA: correct AND faster

Item 5 of `for_the_dell_cuda_proofs_owed.md`, the one its author called "the one most
likely to bite": `radix_sort_sweep_kernel` became three kernels (tile counts, tile prefix,
scatter) because the decoupled-lookback spin does not terminate on Metal. CUDA owed a
**permutation check across the tile boundary** and a **throughput number**, since three
launches replace one on a machine where the lookback worked.

**Correctness.** At the Mac's head `831e10c8`, on one V100-SXM2:

    tests/unit/kernels/test_the_sort_crosses_its_tile_boundary.py
    tests/unit/kernels/test_sort_values_and_indices.py      31 passed in 18.49 s

The boundary cell is the permutation check, and the Mac had already written it. It does not
exist on `main`, so there is no same-cell baseline to quote — what is quotable is that it
passes on CUDA, which is what was owed.

**Throughput**, int32, five repetitions, same card, same sizes, warm:

| n | Mac `831e10c8` (three kernels) | `main` (one-kernel lookback) | |
|---|---|---|---|
| 65 536 | 2.702 ms | 2.306 ms | **17 % slower** |
| 1 048 576 | 10.832 ms | 11.930 ms | **9 % faster** |
| 4 194 304 | 42.003 ms | 47.779 ms | **12 % faster** |
| 16 777 216 | 164.960 ms | 183.775 ms | **10 % faster** |

**The rewrite does not regress CUDA — it is faster at every size above 64 K**, by 9-12 %, and
slower only at the smallest, where three launches cost more than one and there is not enough
work to amortise them. So the contingency the handover named — *"if it regresses on NVIDIA,
the right shape is a capability row choosing the lookback where it terminates, not a revert"*
— **is not needed**. One implementation serves both backends, and it is the better one here.

Scope, stated: one card, one dtype, one distribution (uniform int32), warm, five repetitions
with the spread shown. The 65 536 row is the only one where the extra launches show, and it
is the row where a launch-bound measurement is least trustworthy — its max is more than twice
its min on both trees.

## 2026-09-18 — the Mac's remaining CUDA proofs: launcher, workshop, and `core/paths.py`

Run at its head `831e10c8` against `main` as the baseline, on this rack.

| | `main` | Mac `831e10c8` |
|---|---|---|
| `test_launcher.py` + `tests/unit/workshop` | **25 passed** | **15 passed, 10 skipped** |
| `core/paths.py` | not present (ImportError) | present |

**The ten skips are not a coverage loss.** They are `test_workshop_layout.py` looking for the
workshop root relative to its own tree: from a worktree at
`/home/mlops/nbx/worktrees/mac_ff288827` it computes `/home/mlops/nbx/worktrees/nbx`, which
does not exist, and says so — *"no workshop root at ... — discipline not installed here"*. The
same class as a frozen worktree missing its ignored pointers, and a property of where the tree
sits rather than of what it contains. Everything that runs, passes.

**Item 1 — `_current_backend()` on CUDA.** Covered by `test_launcher.py` passing at the Mac's
head: the launcher cells exercise the backend detection, and nothing reports "declares no
measurement protocol".

**Item 6 — `core/paths.py`.** It resolves the rig's real locations and creates nothing:

    cache_dir()      -> /home/mlops/.neurobrix/cache    exists=True
    store_dir()      -> /home/mlops/.neurobrix/store    exists=True
    neurobrix_home() -> /home/mlops/.neurobrix          exists=True
    describe()       -> says which source each came from ('said_by': 'default')

`describe()` naming the source of each path is what makes this checkable rather than
plausible: a path that came from an environment variable and one that came from the default
are different facts, and it says which.

**Still unmeasured here**, and they are the three the handover itself expected to be
invisible on CUDA: `host_values()` (the bf16 branch — V100 is sm_70 with no native bf16, so it
may simply not arise, in which case the claim stays Apple-only and this says so), the witness
re-entrancy guard (a clock-lock rig never calls `_witness_ms()`), and the MoE capability row
(`{"cuda": True}`, so the refusal cannot fire).

---

## 2026-09-19 — `0c824682`, the reshape rung reaches an upscaler: Apple's half owed

* **owed by** the Dell (this machine) · **for** the Mac to confirm on Apple
* **the commit** is on `main`, origin and gitlab.

### What the Dell established

The request-reshape rung was never missing. `_spatial_component_tiling` sizes a
spatial cut, `plan.component_tiling` carries it and
`core/runtime/executor.py:578` builds a `TilingEngine` from it, so residency is
bounded by the PIECE — the property the op-level rung cannot give, because that one
bounds the transient while the full output stays allocated for downstream
consumers. `real-esrgan-x8` at 1024 px proved that twice on CUDA: it died at
`aten.leaky_relu::278` and then at `aten.convolution::350` for 8 589 934 592 bytes
**with `tile aten.convolution::350 in 64 bands` already in its plan.**

It refused the ENTIRE upscaler family, for two reasons, each demanding a number the
model does not have while the container already held the answer:

1. **the scale factor.** `config.get("upscale")`, then a VAE block list, then
   `if not scale_factor: return None`. Every upscaler in the Dell's cache ships an
   EMPTY `config` — real-esrgan x2/x4/x8, swin2SR-classical-sr-x4-64, hat-l-x4 —
   and each states its factor exactly in its own shapes. The function had already
   read both shapes for its downsampler guard; it now derives the ratio from them
   when the config is silent, requiring both axes to agree and the ratio to be
   exact.
2. **the latent grid.** `if not (vae_scale and _h and _w): raise
   MissingRuntimeValue(...)` told the operator to declare a VAE scale for a model
   with **no VAE**. `InputConfig`'s own docstring already said that absence is
   legitimate for "a dimension the model does not have: no VAE, no vae_scale". An
   upscaler reads pixels and writes pixels; its tiles cover the request's own grid.

Measured after: `tile_size 565, overlap 70, scale_factor 8, tiled_activation
5 230 MB` against **16 384 MB** whole, and the plan now says in its own words
`dropped full-extent op-level tiling (component-level tiling active)`.

### What the Dell could NOT establish, and is owed from Apple

**That the same two guards were what stopped it there.** Both fixes are
vendor-neutral by construction — they read the graph's own input/output ratio and
the request's own extents, and name no backend — but "vendor-neutral by
construction" is an argument, not a measurement.

**What to run, and what each answer means:**

```
neurobrix run --model real-esrgan-x8 --input-image <1024px> --explain-plan
```

* **`component tiling model: {...}` appears** → the rung now reaches the upscaler on
  Apple with no further change, and the remaining question is only whether the
  artefact is clean.
* **it does not appear** → the thing to report back is WHICH guard still bites.
  `NBX_PRISM_TILE_DIAG=1` prints the sizing decision (`full_act`, `budget`, the
  bound `InputConfig`, and TILE/native). A `MissingRuntimeValue` naming
  `vae_scale` means fix 2 did not reach that path; a silent `None` before the diag
  line means the scale factor was still not derivable, and then the useful datum is
  that model's `profile.json` `config` and its graph's input/output shapes.

**And the artefact, judged the way the Dell judged its own** — whole, and then a
crop at FULL resolution centred exactly on an internal boundary. The Dell's
harness stitch of the same request measured a seam of **+0.22 sigma vertical and
+1.27 sigma horizontal** against its own neighbourhood, below the Mac's
hand-proved **+2.07 sigma**, and showed no discontinuity to the eye. Note the two
differ by design: the Dell's harness HARD-TRIMS the halo, while
`tiling_engine.py` blends by ACCUMULATE-AND-DIVIDE, so the engine's own artefact
should be at least as clean and a step at a boundary would be a ramp rather than
an edge.

**Not yet returned by the Dell**: the engine's own artefact for that request was
still rendering when this entry was written. The harness's is judged and clean.
