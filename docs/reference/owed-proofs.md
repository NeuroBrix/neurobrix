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

---

## OWED TO THE DELL — the certifier's stability contract now has two regimes (2026-09-16)

The common certifier changed, minimally, and your suite must see it before a red
does. What changed:

1. **Protocol file scoped by backend.** `tools/rig_protocol.json` →
   `tools/rig_protocol.cuda.json` (your V100 clock lock, unchanged in content).
   The engine (`autotune_certify._protocol_file`) and the workshop
   (`tools/rig_clock.py`) now read `rig_protocol.<backend>.json`. This was a bug
   regardless: the un-suffixed file leaked the V100 protocol to a Mac, which
   refused certification on clocks it cannot read. **Your cuda path is
   unchanged** — same clock, same `nvidia-smi -ac`, same door.
2. **The regime field is clock-OR-witness.** `rig_protocol_refusal` dispatches
   on `regime`: `clock_lock` (yours, the exact prior check) or `witness` (Apple,
   `rig_protocol.metal.json`). Under `clock_lock` the new witness code is INERT
   (`certify_key`'s witness bracket runs only when `_regime()` is `witness`), so
   your timings and proofs are byte-identical to before.
3. **`proof_records_clock` → `proof_records_regime`** (alias kept). It now
   returns true for a recorded clock OR a recorded `stability_witness`. Your
   proofs record `machine.clocks_mhz`, so `entry_covers(..., need_clock=True)`
   answers exactly as before.

Tests: `test_autotune_certify_clock_door.py` now forces the cuda protocol in its
fixture (the default on a Mac is metal), `test_witness_regime.py` is new, and
`test_the_protocol_authority_...` asserts the backend-scoped name. Run your
suite on the cuda path and confirm the clock door is unmoved. See
`docs/reference/what-certified-means.md` for the contract.

Not verifiable on the Mac yet: matmul does not compile on Metal (a codegen bug,
`r_55` used out of its declared scope, exposed once the `llvm.intr.assume` refusal
was lifted), so the witness — a matmul — cannot yet run there and the Apple
certification stays blocked on that fork codegen chantier. The contract is in
place for when it compiles.

---

## OWED TO THE DELL — the Metal backend seam de-vendors the shared refusal module (2026-09-16)

The engine no longer names a backend vendor. This is a trunk architecture
correction (R33/doctrine: the engine targets Triton; the backend is a
selection, not a branch), and its first increment touches a module the Dell
runs — so you must see it before a red does.

WHAT CHANGED:
- `kernels/autotune_refusals.py::_is_backend_refusal` no longer imports
  `triton_msl.errors.MetalNonRecoverableError`. It asks the Metal seam,
  `triton.metal_backend.is_backend_refusal(exc)`, which collects the refusal
  types of whichever Metal backend is present (the bledden fork, and triton-ext
  when it names one) — so the shared module names no vendor.

WHY IT IS INERT ON CUDA:
- `backend_refusal_types()` imports each backend's type in a try/except; on the
  Dell no Metal backend is installed, so it returns `()`, and
  `is_backend_refusal` returns False — EXACTLY what the old code returned when
  the `triton_msl` import failed. Verified here: fork type -> True, ValueError
  -> False; on a machine with no Metal backend, always False.

WHAT YOU MUST PROVE:
- Inert on CUDA: `_is_backend_refusal` on a cuda run behaves byte-identically —
  no config that was excluded is now kept, none that was kept is now excluded.
  Your autotune/certify kernel suite must be byte-identical (no proof diff).
- The import path adds no cycle on your tree (autotune_refusals ->
  triton.metal_backend is one-way; metal_backend does not import
  autotune_refusals).

STILL TO COME (same seam, later increments): backend chosen by PROFILE (fork vs
triton-ext), `metal_driver.py` selecting the compiler through the seam rather
than importing `MetalBackend` by name. Each lands with its own owed-proof.

---

## OWED TO THE DELL (Forge / graph capture) — real-esrgan-x2 declares spatial symbols and never uses them

The defect is in the CAPTURE, so it is Forge's, so it is yours. Here is the
discriminant and the three measurements that settle it, so you do not have to
rediscover them.

WHAT THE GRAPH SAYS. `components/model/graph.json` → `symbolic_context.symbols`:
    s0 = batch  (trace 1)
    s1 = height (trace_value 64, source input::pixel_values::dim_2)
    s2 = width  (trace_value 64, source dim_3)
and NO op references s1 or s2. Every spatial dim is frozen at its trace value.
The first op (the pixel-unshuffle decomposition) carries a literal target shape:

    aten.view::0   input_shapes [[1,3,64,64]]
                   attributes.shape [ {symbol s0}, 3, 32, 2, 32, 2 ]

Only the batch is symbolic; `32, 2, 32, 2` are trace literals, so the graph
demands exactly 64×64 (32×2) while declaring it accepts any size.

THE THREE MEASUREMENTS (run here, M4 Pro, compiled/MPS arm — no Metal backend
involved, so this is not a backend matter):
1. **64×64 → 128×128, std 105.3** — a correct upscale. It matches the trace.
2. **224×224 → RuntimeError at `aten.convolution::0`**: "weight of size
   [64,12,3,3], expected input[1,147,32,32] to have 12 channels, but got 147".
   147 = 3×7²: the resolver held the literal 32 spatial and floated the
   unshuffle factor to 224/32 = 7, where the model's factor is 2 (12 = 3×2²).
3. **448×448 → a blank white 128×128, SILENTLY** (432-byte PNG, std 0.061, 7
   distinct values, 122/128 constant rows). No exception. This is the
   silent-wrong; it was classed "tiling or request" for four days on a statistic.

THE DISCRIMINANT (mechanical, and it proves it is the capture):
for each model, {declared symbols} vs {symbols referenced in op shapes} —

| model | declared | USED in ops |
|---|---|---|
| **real-esrgan-x2** | batch, height, width | **batch only** |
| swin2SR-classical-sr-x2-64 | batch, height, width | height, width |
| hat-s-x4 | batch, height, width | height, width |
| swinir-classical-x2 | batch, height, width | height, width |

real-esrgan is the ONLY one whose capture failed to substitute the spatial
symbols, and the ONLY one that fails. The other three are PROVEN at 448×448
(artefacts judged by eye: correct apple, correct geometry).

WHAT WE DID NOT DO: no runtime patch. Papering over this in the resolver would
hide a graph that lies about its contract. The fix belongs where the lie is
written — the capture must substitute s1/s2 into op shapes as it already does
for the other three.

GENERAL FORM (register, 2026-09-16): a declared symbol that is never consumed is
worse than an absent one, because it makes the check PASS — every gate reads the
declaration. The emptiness of {declared} − {used} is mechanically checkable and
is worth a capture-time assertion.

---

## OWED TO THE DELL — the Metal seam, increment 2: the backend is chosen by the PROFILE (2026-09-16)

Completes the de-vendoring told in increment 1. Same rule: you must see it
before a red does.

WHAT CHANGED:
- `triton/metal_backend.py` gains `METAL_BACKENDS` (the ONLY place either
  implementation is named: the bledden fork `triton_msl`, and triton-ext's
  `triton_apple_backend`), `selected_metal_backend()` and
  `backend_compiler_class()`.
- The hardware profile may declare `metal_backend: triton_msl | triton_ext`.
  `apple/apple_m4_pro.yml` now declares `triton_msl`. **No CUDA profile declares
  anything**, and nothing reads the key off Apple.
- `triton/metal_driver.py` no longer imports `MetalBackend` from the fork by
  name; it asks the seam for the compiler class.
- `kernels/autotune_refusals.py` (SHARED, you run it) no longer names a vendor
  ANYWHERE — including its docstring, which used to name the fork's refusal type
  in prose. It asks the seam.

REFUSALS, so a measurement can always name what produced it:
- a declared backend that is NOT installed is refused BY NAME (never silently
  swapped for the other one);
- an unknown backend name is refused;
- both installed and none declared is refused ("must say which");
- none installed is refused.

WHY IT IS INERT ON CUDA:
- `metal_driver.py` is the Metal driver; it is not imported on a CUDA run.
- `selected_metal_backend()` is only reached from the Metal path.
- The shared module's `_is_backend_refusal` asks the seam, which returns `()`
  refusal types when no Metal backend is installed → False, exactly the old
  import-failure answer.

WHAT YOU MUST PROVE: byte-identical autotune/certify behaviour on CUDA (no
config newly excluded or newly kept), and that no CUDA profile needs the new
key. Tests here: `tests/unit/triton/test_metal_backend_is_a_selection_not_a_branch.py`
(9 cases, no GPU, probes monkeypatched) — including one that greps the shared
refusal module for vendor names and fails if any returns.
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

### Returned by the Dell, 2026-09-19 01:31

**The engine's own artefact is in and it is clean.** `GREEN_x8_1024_by_the_rung.png`,
8192 x 8192, produced by the component-tiling rung at tile 565 / overlap 70, blended
by accumulate-and-divide. Looked at whole and at full resolution across the join: a
coherent apple at 8x, no grid, no visible seam. Against the harness's independently
stitched artefact of the same request, `mean|d| 0.036` with 0.26% of 67 million
pixels differing by more than two levels -- which is what blending versus trimming
the overlap costs, and nothing more.

**A warning about HOW to measure the seam, learned the hard way here.** Measure at
the boundary THE ENGINE USES, not at the geometric midpoint. The engine's tile
stride is 565 input px = 4520 output px; my harness cut at the midpoint. At the
midpoint the engine reads +0.23 and +0.57 sigma and looks perfect; at 4520 it reads
**+10.35 and +12.18 sigma** -- and there is still no line, because that region is
smooth red and the local noise floor (0.247) collapses, so a small step reads as
many sigma.

**Use the ABSOLUTE step, which is what an eye sees:**

| boundary | step, grey levels of 255 | typical elsewhere |
|---|---|---|
| vertical @ 4520 (the engine's own) | 1.047 = **0.41%** | 0.166 |
| horizontal @ 4520 | 0.635 | 0.100 |
| vertical @ 4096 (not a boundary here) | 0.377 | 0.166 |

So the tile boundary is measurably elevated, about 6x the local step, and
sub-visible. **A sigma is a ratio and does not travel between pictures** -- the
+2.07 sigma figure from Apple was measured on its own image, so compare the absolute
step, and say which boundary it was taken at.

**Two independent harness stitches on two different cards came out BIT-IDENTICAL**
(same sha256, 0 differing pixels of 8192x8192x3), so the method is deterministic
across cards and a single judged artefact is not a single lucky run.
## 2026-09-20 — the Mac's adaptive-memory proposal, reviewed on CUDA

`docs/reference/adaptive-memory-a-runtime-controller.md` (metal-first-light) names four
additions. Measured against what main holds today:

| addition | on main | by |
|---|---|---|
| 1. a truthful estimate | the denominator: an op is budgeted against what is LEFT of the card (`live_before_op`, `resident_bytes`), and the plan states the margin it keeps (`max(3072 MB, 12 % of capacity)`) | `79558447`, `77848eee` |
| 2. a plan that can say "not whole", entering the tiling rung deliberately | the request-reshape rung `_spatial_component_tiling` reaches the upscaler family (scale derived from the graph's in/out ratio; no `vae_scale` demanded); judged green at x2/x4/x8 on 16 GB and 32 GB cards, engine artefact 8192² with a sub-visible seam of 1.047/255 at the engine's own boundary | `0c824682`, item 1 closed by the owner 09-20 |
| 3. a runtime controller at the allocation failure — ONE re-entry into the tiling rung with the allocator's real figures | **not landed anywhere**; the document itself says the re-entry is "the next step, named rather than assumed small" | — |
| 4. the refusal names what would have fit | `kernels/oom_advice.py` | `66ed17af` |

**Verdict.** The request-reshape rung the proposal asks for is in main and proven (item 1).
Additions 1, 2 and 4 hold on this side; nothing to land for them beyond what landed. Addition 3
is design: it is the right seam (a wrapper cannot change its own output contract — the Mac's
measurement that band-streaming at 4 GiB and 16 GiB dies identically is reproduced by our
four OOMs at 8 589 934 592 bytes, the second WITH a 64-band plan). It stays a named follow-up,
not a rung to land today: with 2 the plan already refuses or reshapes before execution for the
family that motivated it, and a re-entry after the plan was chosen is a change to the
executor's contract that needs its own red-then-green case (an estimate wrong by enough to OOM
under a plan that was already tiled).

## 2026-09-20 — three things the CUDA side established for the Apple side

### 1. Kokoro moved through the merge, twice, both times through `aten::pow`

The zoo byte pair (A = main with the 3.8.0 directory, B = the merge head) left Kokoro-82M
UNADJUDICATED: a seedless tts request cannot hold still (the family's `seed: 42` sits under
`calibration:` only, so the Triton stream runs unseeded and the vocoder's `aten.rand::0` [1, 9]
is the first op to differ between two runs of one tree). With `--seed 42` each tree holds
still (A1 == A2, B1 == B2) and the two differ: SNR 24.8 dB between waveforms of identical
length, whisper-large-v3-turbo transcribing both as "Hello.".

`git bisect` on a 9-second row, autotune off, the floor measured before each lever:

| lever | from → to | first bad commit |
|---|---|---|
| main → the Mac's branch | 8c4b4e1d… → 650f2eec… | **85c6426a** — kernels: stop calling NVIDIA's device library from portable kernels |
| the branch → the merge head | 650f2eec… → 7dfe314b… | **9ddebd18** — pow takes the exact route for a small integer exponent |

Kokoro calls `aten::pow` 49 times (e = 2 and 3); nothing else in the branch reached its
bytes. The portable route moved it first (up to 15 ulps from the fp64 oracle on a square,
recorded in `pow.py`), the exact route moved it back toward the oracle. Both are deliberate and
in the CHANGELOG; the row is ADJUDICATED: moved, attributed, judged clean. On Apple the same two
commits run the same kernel, so the same movement is expected there — a byte pair across the
merge on Apple will show it and should not be read as a Metal regression.

One instrument note: `NBX_OP_FINGERPRINT` hashes the first 8 192 bytes of a tensor by default,
so its "first differing op" is the first op whose PREFIX differs (it named a layer-norm mean
while the layer-norm's input already differed). Full hash: `NBX_OP_FINGERPRINT_MAX=0`.

### 2. The ladder law, applied to the CUDA cards (4bd6a5e7)

The law (2eeff74a): the budget rounds DOWN onto whole-GB rungs before anything downstream is
computed; nothing downstream is rounded; never tuned so a supervising machine's ambient stops
straddling a boundary. Applied here it had nothing to act on: on this rack the "reading" that
reached the rung was capacity − margin, a constant per card, and the rung was reached only
when a request overflowed the whole card. The plan never read the card's LIVE free memory.
Measured on card 3 (32 GB) under a neighbour's 18–27 GB hold: real-esrgan-x8 at 1024 planned
whole and died on its first allocation, 5 of 5.

Now the free reading (one door, `DeviceAllocator.free_memory_mb`) enters the plan first, rounds
down onto the ladder, and the tile budget follows from that rung. The ladder's rungs are the
Mac's and are not tuned to this rack's ambient. First rows under the patch (autotune off, so the
rows measure the plan and the fit, not a runtime sweep):

| card | class | hold | reading | rung | budget | rc |
|---|---|---|---|---|---|---|
| 1 | 16 GB | 2 GB | 13.17 GB | 12288 MB | 4.80 GB | 0 |
| 3 | 32 GB | 18 GB | 13.14 GB | 12288 MB | 4.80 GB | 0 |
| 2 | 32 GB, 1536² | 0 / 2 GB | 30.40 / 29.14 GB | 24576 MB | 9.60 GB | 1 — CUDA 700 at `aten.convolution::351` |

The 1536² rows were not the plan's failure: the rung and budget are right by the law. Pinned
with `--mode triton-sequential` and `CUDA_LAUNCH_BLOCKING=1`: the x8 network's LAST conv
(64 → 3 channels at a 6344² tile) has a small output — under the 4 GiB band-streaming
threshold, so it ran whole — and an input of 2 576 000 000 elements, and its loop-derived
int32 channel offset wrapped past channel 53. Widened to int64 in conv2d and conv1d
(d95be536, register 80, a 2.3-billion-element cell seen red then green). The five-row series on
both classes (5/5 rc=0 each, identical rungs row by row) and the 1536² re-run are in
`nbx/campaigns/2026_09_20_ladder/LADDER.md`. On Apple the same kernel runs the same offset
form, so a tile whose input crosses 2^31 elements would have faulted there too.

### 3. A runtime flag read only from the build toolchain's registry — the Apple installs ran without it too

`zero_pad_embeddings` (and five other per-component flags) were read at runtime through
`.nbx_registry`, a gitignored pointer only a developer checkout carries. Every worktree and
every `pip install` ran Wan2.1-T2V-1.3B with the flag at its default: a lattice of 16-px
cells where the checkout rendered the sailboat (one judged run per arm, same commit; period-16
column signature 0.1 with the pointer, 18.0 without; the rebuilt container without the pointer:
0.1). If an Apple Wan measurement was taken from a worktree or an install, it was taken
without the flag. Fixed in main 6fb35c3b: the build writes the six flags into the container's
extracted values, the container records them when opened, the reader's order is env override →
registry → container → default (`docs/reference/release-decisions.md` lists the thirteen hub
containers that need a rebuild and a re-upload).

## 2026-09-21 — the Apple x8 retention object, the CUDA arm

**The object as named (the Mac, 2026-09-20):** `TritonSequence.run()` retains 410 MB live and
parks 332 MB in the pool per invocation at constant tile shape, `skip_kills` false and kills
firing — the class its own docstring warns about. Real-esrgan-x8 at 1024 px is owed on it.

**The instrument, landed for both backends:** `NBX_RUN_LIVE_DIAG=1` prints, at the entry and the
exit of every `run()`, the device's driver-held bytes (`_cuda_live_bytes`, which counts
pool-parked blocks until they are returned to the driver) and the pool-parked bytes; live
proper is their difference. A series, not an inference from an OOM.

**CUDA, the same request (x8 at 1024², rung 8192 by a 20 GB neighbour, tile 448, nine
invocations at one shape, static kernel configs):**

| arm | run#1 entry → exit | run#2 entry → exit | runs #3–#9 (each) | peak driver |
|---|---|---|---|---|
| pool on | 45 → 9 968 MB (9 849 parked) | 10 097 → 10 489 (9 777 parked) | 10 489 → 10 489, parked 9 777 → 9 703 | 10 489 MB |
| pool off | 45 → 119 MB | 631 → 704 MB | 704 → 778 MB, back to 704 at the next entry | 6 976 MB |

So on CUDA nothing is retained per invocation: with the pool off, every invocation from the
third on enters at 704 MB and exits at 778 MB, and the 74 MB it takes are released before the
next entry. The one step that exists — +512 MB between the exit of the first invocation and
the entry of the second — is taken OUTSIDE `run()`, by the caller between tiles (the tiling
engine's output canvas: 8192² × 3 channels in fp16 is 402 MB), once. With the pool on the same
series reads as a constant 9.7–9.8 GB parked (peak 10 053 MB, three flushes, 8.2 GB of
smallest-fit slack over 12 544 exact and 420 fit hits) and no growth either.

**What that says for Apple.** The growth the Mac measured is not in `run()`'s CUDA path at
this request; the same instrument on Apple, at the same request, will show whether the step
is per invocation there (then it is the Metal driver's allocator or the pool's Metal path —
`kernels/metal_device.py`, the pool's free-list on that backend) or between invocations (then
the caller, as here, and the number should be one canvas, not one per tile). The series is
the handover; the 410/332 figures need their entry/exit pairs before a kernel is named.

## 2026-09-21 — the ladder's other half, sent as the ladder rule was received: the tile lattice

**The law, in the same words as the ladder's:** the tiled extent handed to the kernels is
snapped DOWN onto the vendor profile's lattice (`tiling.extent_lattice`) after the budget
chose it and before the overlap and stride are derived from it; nothing else downstream is
rounded; the lattice is a property of the backend's kernels, measured on that backend, never
tuned to a picture or a card's ambient. The ladder bought determinism by rounding the memory
reading down; the lattice buys back the performance determinism alone does not.

**Measured on CUDA (V100, real-esrgan-x8 at 1024², rung 8192, budget 3.20 GB, one card):**

| tile | lattice | unused area vs 457 | static configs | production, cold | production, warm |
|---|---|---|---|---|---|
| 457 | none | 0 | 717.9 s | 2 065.8 s (11 shapes swept) | 31.9 s |
| 456 | 8 | 0.4 % | 544.7 s | 1 286.6 s (11 swept) | 28.4 s |
| 448 | 16 (and 32, 64) | 3.9 % | 40.9 s | 20.4 s (certified) | 19.5 s |
| 432 | 16, not 32 | 10.6 % | 37.4 s | — | — |
| 416 | 32, not 64 | 17.1 % | 35.0 s | — | — |
| 384 | 128 | 29.4 % | 31.5 s | 905.6 s (9 swept) | 15.4 s |

The unit is 16 on Volta: every multiple of 16 sits on one plateau (31.5–40.9 s static) whether
or not it is a multiple of 32 or 64, a multiple of 8 does not, and a coarser unit only spends
tile area (384 loses 29 % of the tile for 23 % of the time). Under production autotune the
warm cliff is 1.6× (31.9 against 19.5 s) but the cold one is a hundredfold, because an aligned
extent lands on shapes the directory already certifies while an odd one sweeps eleven
unscreened shapes for 34 minutes — alignment also shrinks the set of shapes a backend has to
certify. Landed: `volta.yml` `tiling.extent_lattice: 16` with the table beside it, read through
one door in `PrismSolver._tile_extent_lattice` (env `NBX_PRISM_TILE_ALIGN` overrides for a
measurement), a four-cell file, and the model-level green (no override: 39.7 s at the 8192
rung). **Owed from Apple:** the same table on M4 Pro — the lattice there is the Metal conv
kernel's, not 16 by inheritance; measure 457 / 456 / 448 / 432 / 416 / 384 at one rung and
write the unit into `apple_m4_pro.yml` beside its numbers.



---

## 2026-09-21 — RESOLVED: the single-tile upscaler was a STALE LOCAL CONTAINER, not the engine

**Superseded.** The Dell retraced real-esrgan-x2 on 09-20 (graph `74a2d7ea`, view::0 now
`floordiv(s1,2)` — symbolic) and it is on the shared cache; my local copy was `626f2e07`,
the old frozen graph, identical to the hub because the store stopped accepting writes.
Refreshed from the shared cache → x2@448 → **896×896**. No engine bug, no Dell datum owed.
The rule that stands: verification copies come from the shared cache (canonical), never the
stale hub; and the Apple census now reads graphs from the shared cache, tagging each model's
keys with its graph_sha so a later retrace invalidates exactly its own keys.

### (historical, now moot) 2026-09-21 — CORRECTED: the single-tile upscaler output is MINE (Apple), not the rack side

My earlier entry here handed the reshape-rung fold breakage to the rack side. **That was
wrong, and the correction is the owner's, checked:** the Dell ran `real-esrgan-x2` at 448 in
BOTH modes on main HEAD and on BOTH parents of the merge `78784abe` — **896×896 every time,
the 49 tiles accumulated.** So the engine's fold is correct on CUDA; the single-tile output
is Apple-specific. The merge's engine-side files are the launcher, the Metal backend and
driver, the certifier, the tensor library, and the Triton sequence — the cause is in one of
those, on my side. A bisect landing on a merge assigns no parent side without testing each;
the Dell tested them and they are green, which is the datum I owed and did not produce.

**ROOT CAUSE, precisely characterized on Apple (2026-09-21):** real-esrgan-x2 is a
PIXEL-UNSHUFFLE upscaler. `aten.view::0` targets `[s0, 3, 32, 2, 32, 2]` (32 = trace_H/2)
and `aten._unsafe_view::0` targets `[s0, 12, 32, 32]` — the spatial dims enter as H/2, W/2
at the pixel-unshuffle positions, NOT as H, W. The census confirms s1/s2 (height/width) are
`never_carried`, first broken here. The spatial-promotion pass
(`triton/promotion._spatial_promotion_pass`) matches H/W at shape positions [-1]/[-2]
against the TRACE H/W (64); the pixel-unshuffle `32` (=64/2) matches neither, so it is left
frozen (MEASURED: view::0 args byte-identical before and after the pass). With the spatial
frozen at 32, the batch symbol `s0` (trace 1) inflates to 49 to absorb the runtime
448×448 → the graph runs `[49,3,64,64] → [49,3,128,128]`, and `output_dispatch.final_as_array`
takes `np.take(arr, 0, axis=batch_axis)` — batch index 0 — yielding the single 128px tile.

**The unresolved contradiction (needs the Dell):** the hub x2 and my cached x2 graphs are
BYTE-IDENTICAL (same frozen view::0), the promotion pass is shared across all modes and
platforms, and it PROVABLY leaves the 32 frozen on Apple. Yet the Dell reports 896×896 on
CUDA in both modes "with the 49 tiles accumulated." So on CUDA the identical frozen graph
either (a) has `s0` resolved to 1 with the spatial promoted to 224 — which the shared pass
does NOT do here — or (b) reassembles the 49-batch `[49,3,128,128] → [1,3,896,896]` in an
engine step I could not locate in the flow, output_dispatch, or the graph. I need the Dell's
answer to ONE question: on CUDA, what is `GraphExecutor.run`'s OUTPUT shape for
`real-esrgan-x2 @448` — `[1,3,896,896]` or `[49,3,128,128]`? That single datum says whether
the divergence is at symbol resolution (before the graph) or at reassembly (after it).

**The clean resolution regardless: RETRACE.** The census (now working on Apple) marks x2 —
and every pixel-unshuffle upscaler — for retrace: a symbolic-spatial retrace removes the
frozen 32 and the batch inflation entirely, and the family verifies natively at any size on
both platforms. That is the doctrine's own prescription for a frozen container, and it does
not wait on the engine-side contradiction above.

**What stands as the symptom:** `real-esrgan-x2 --input-image apple_448.png` (traced
`[1,3,64,64] -> [1,3,128,128]`) emits 128×128 on Apple — one tile, upscaled (pixel-matched:
`mean|Δ|=1.0` vs `crop(0,0,64,64).resize(128)`, `74.3` vs whole-downscaled), BOTH modes; the
run forms a batch of 49 (`batch_dim=49` in the autotune keys) and only tile 0 survives to the
output. The whole upscaler family's verification is blocked on it. **Owner: me. Finding the
Apple-side accumulate/fold defect is the current work; the family is BLOCKED, not delivered.**
The `real-esrgan-x8 @448` `leaky_relu _device_idx` face was already fixed (the interceptor
dispatches by kind before size, 329ab4a9-line).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HgxLbUbkxogC87ppc4tQ5H

---

## 2026-09-21 — zero3 selection on unified memory (cafaf799): the CUDA inertness arm

**The change.** Strategy 3 no longer selects `zero3:` where `_device_is_unified` says the
offload frees nothing; zero3's torch path refuses by name on a non-CUDA device.

**Inert on this rack, three ways.** (1) The door: `_device_is_unified("cuda:0" / "cuda:1",
<this rack's own profile>)` answers False, so the new condition cannot fire on a discrete card.
(2) The cells: from cafaf799 frozen in a worktree, every prism cell main carries passes here
(232); the discrete arm of the Mac's own cell passes. (3) The catalogue: a Prism plan census
of the 59 installed containers on this rack's four-card profile, main against cafaf799 — 56
planned, 3 refused identically ("cannot run on this machine"), 0 plans differ; strategies
single_gpu 35, lazy_sequential 9, block_scatter 6, pipeline_parallel 2, weight_sharding 2; no
zero3 placement on a four-card rack at all, so the branch it guards is not even reached.
(The census recorded strategies; the per-component device list came back empty from my
reader and is not claimed.)

**What the branch owes its own cells.** Three of them called `load_profile("default")` and
read the machine that wrote them: on the Dell the file is `default-<hash>.yml` and they raised
FileNotFoundError. Fixed on branch `zero3-cells-any-machine` (57f1a739, on both remotes, from
cafaf799): the live cell asks the autodetect door and skips on a host without a device, the two
that declare a unified device do so on a fixture the tree carries (`a10-24g`). Measured here:
5 passed with a card, 2 passed 1 skipped masked. Merge it with the commit.

**The granite MoE cell** (`test_the_granite_moe_block_is_fused_not_replayed`) fails on the
Dell for a different reason: it asserts the branch's rewrite (no `split_with_sizes` survives)
against a container that main's fusion (9f5e0f5b: the stacked-expert views, the traced splits
kept) runs — and on the merged tree main's walk matches granite first, so the branch's matcher
is never reached on CUDA. The A/B is below.

## 2026-09-21 — the granite fusion, measured on CUDA: main's walk against the branch's matcher

On the merged tree the branch's granite matcher (`_fuse_one_granite_layer`, be4bd421) runs only
when main's general walk returns None, and main's walk (9f5e0f5b, stacked-expert views) matches
granite first — so on CUDA the branch's matcher is unreachable and the two arms of a plain A/B
are the same code (fact and code outputs byte-identical, peak 2 707–2 711 MB). Branch
`granite-fusion-ab` (from cafaf799) carries a lever, `NBX_MOE_FUSION_MATCHER=granite`, that skips
the walk so the branch's matcher is the one that runs. Triton mode, one 16 GB card each, two
requests, each twice (both arms hold still):

| arm | fact ("capital of France", 16 tokens) | code (is_palindrome, 160 tokens) | peak driver | wall, code, cold |
|---|---|---|---|---|
| main's walk (card 0) | "Paris." — sha bdff8c41… | sha 5a52cc92…, 431 chars, **8/8 cases pass** | 2 707–2 711 MB | 23.1–23.8 s |
| branch's matcher (card 1) | "Paris." — sha bdff8c41… (identical) | sha e7725da9…, 556 chars, **8/8 cases pass** | 2 733–2 738 MB | 21.3–21.5 s |

**Judged outputs:** identical on the short request; DIFFERENT text on the long one and both
correct on every case the code never saw. The two fusions are therefore two valid numerics of
one block, not one right and one wrong. **Figures:** peak favours main by 27–31 MB (about 1 %);
wall favours the branch's matcher by 1.6–2.5 s (7–10 %) on the cold code request and by 2–4 s on
the warm fact request (7.0–7.4 against 9.4–11.2 s — different cards, so the warm figure is
indicative). The two measurements point in different directions and both go to the owner, as
asked. The Apple half — the same table on M4 Pro — is the branch's to add; the lever is on the
branch for it.

### The arbiter (the owner, 14:24): which fusion reproduces the UNFUSED ATen arm on the code request

The unfused ATen arm cannot run from the container: the traced MoE routing carries
`split_with_sizes` sizes frozen at the trace — on the code request the op refuses (`split_sizes
must sum exactly to 760`, 95 prompt tokens × top-k 8), twice `rc=1` in `--sequential` with
`NBX_DISABLE_MOE_FUSION=1` on card 0. The frozen sizes are the reason the fusion exists (the
branch's own comment says so); they are also a frozen-dimension defect of the container, listed
as such for the retrace queue. The vendor forward IS the unfused computation, so the arbiter
ran it: transformers 5.2.0, greedy, the container's own embedded tokenizer and chat template
(95 prompt tokens = 760/8), fp16, card 0, twice —

| arm | code request, 160 tokens greedy | prompt tokens | wall |
|---|---|---|---|
| vendor forward (unfused, transformers) ×2 | sha **e7725da9d641**, both runs | 95 | 6.9–7.5 s warm |
| branch's matcher (`_fuse_one_granite_layer`) ×2 | sha **e7725da9d641** — byte-identical to the vendor | 95 | 21.3–21.5 s cold |
| main's walk (9f5e0f5b stacked-expert views) ×2 | sha 5a52cc921133 — differs from the vendor | 95 | 23.1–23.8 s cold |

**Verdict:** only the branch's matcher reproduces the unfused arm; it wins regardless of speed
(and happens to be the faster one). Main's stacked-expert handling in the general walk leaves
the tree — a targeted port of the branch's matcher and its dispatcher resolution
(`expert_weight_lists`) onto main, gated on granite's three modes, since the branch carries the
Mac's whole 0.5.5 delivery and the merge is the convergence's. Script and outputs:
`nbx/campaigns/2026_09_21_granite_fusion/` (`vendor_oracle.py`, `RESULTS.md`).

**The port, judged (15:22, card 2, branch `granite-matcher-on-main`):** the branch's matcher and
its dispatcher resolution on main's tree, main's walk handling reverted — the code request in
the three served modes, twice each:

| mode | run 1 | run 2 | wall, cold |
|---|---|---|---|
| sequential | e7725da9d641 | e7725da9d641 | 32.2–32.8 s |
| compiled | e7725da9d641 | e7725da9d641 | 20.2–21.5 s |
| triton | e7725da9d641 | e7725da9d641 | 22.1–22.9 s, peak 2 738 MB |

Every mode reproduces the vendor's unfused forward byte for byte. Landed on main as c514b87b (both remotes) after the doctrine review's two HIGHs were closed: the triton fused op now carries the two stacked slots (R30), and the rewrite owns its renormalisation flag. `triton/moe.py` was not
taken: the branch's diff there is the Metal pinned-address tables and a Metal block size, no
granite content. The Mac's matcher is now the only granite fusion in the tree.

## 2026-09-21 — OWED TO THE DELL (vacuous-gates register): a new entry to number, and a stale count to fix

Two register bookkeeping items the Dell owns, because numbers are assigned on main only.

**1. A new vacuous-gate entry to number and append** (text below, ready for `### <n> — …`).
The fix is committed on `metal-first-light` as `793ac348`; it belongs in the register as the
sibling of the `__version__` and `BACKEND_NAME` entries.

> ### <n> — a certified setting served to an out-of-tree backend that had moved, its version in neither half of the identity
>
> **Where.** `src/neurobrix/kernels/autotune_certified.py::generator_identity`, 2026-09-21,
> this Mac, `metal-first-light`.
>
> **What it did.** The generator identity stamped `{triton distribution version, backend name}`.
> On Metal the code that actually runs is generated by triton-ext's OUT-OF-TREE backend (target
> `mps`), whose version is in neither half: the distribution version tracks the in-tree compiler,
> and the out-of-tree backend's own distribution metadata is a static `0.1.0`. So moving
> triton-ext (5439436 → b9d5c06) would change the compiled kernels while the identity stayed put,
> and the gate would serve every certified entry to a generator that had moved. Measured: the
> backend hash moves `msl-v0.1-99803d274e80` → `msl-v0.1-a76b1bff3a60` across that build; the old
> identity did not. Just below it, `name` defaulted to `"cuda"` behind a bare `except` — the exact
> shape of the BACKEND_NAME defect that once refused all 945 Apple entries.
>
> **What would it have done if the code were wrong?** Served — the whole certified directory to a
> compiler that had moved, silently, exactly as the `__version__` half did before its fix.
>
> **The fix.** Add the out-of-tree backend's source hash to the identity through the one door for
> writer and gate — triton's own `backend.hash()` (its kernel-cache-key hash), resolved by triton's
> own `make_backend(target)` so nothing hardcodes the mps/apple mapping. In-tree (cuda/amd) carry
> no hash, so the rack's labels are unchanged. Remove the silent `"cuda"` default: an unreadable
> target RAISES (ZERO FALLBACK); `running_generator` turns that into None (serve, don't refuse on
> an unanswerable question), the writer fails rather than mis-certify. Red then green:
> `tests/unit/kernels/test_a_proof_names_its_code_generator.py` (+4 cells),
> `test_the_generator_label_names_the_real_backend.py` (2 cells updated to the three-part contract).
>
> **The lesson, in one line.** A generator's identity must name the code that actually runs; an
> out-of-tree backend whose version rides in neither the distribution nor a static wheel number is
> a generator the gate cannot see move.

**2. A stale count to fix.** `docs/reference/vacuous-gates-register.md` states "77 entries" but
holds 81 (`_entry_numbers` counts the ranges); `test_the_stated_count_matches_the_entries_present`
fails on main today. The count was not updated when entries 78–81 landed. Fixing the count (and
numbering the entry above) are main-side, so the Dell's — flagged here rather than edited from a
branch 36 commits behind main, to honour the register's own rule that a number never moves.
## 2026-09-21 — a plan is budgeted at the request the flow executes (Wan2.1-T2V-1.3B, CUDA)

The estimator-against-ATen gap the owner named (11.6 GiB asked, 24.1 GiB held, 19.7 GB planned)
was not an estimator error at the shapes it was given: it was given the wrong shapes. The
executor renders a video whose request names no resolution at the CONTAINER's own output size
(the backbone's traced latent [1, 16, 2, 60, 104] times the VAE scale 8 = 480x832), while the
plan's `InputConfig` carried height=None, width=None — so the profiler bound the VAE's spatial
symbols to nothing, fell back to the trace extent (112x176), and estimated 1.74 GiB for a
decode whose first conv input alone is 81 x 192 x 480 x 832 x 4 B = 24.84 GB (the exact
allocation torch refused). Measured with the new default-off `NBX_PRISM_ESTIMATE_DIAG=1`:

| tree | plan request | VAE symbols | VAE first-pass peak | overflow ops | strategy, planned |
|---|---|---|---|---|---|
| main | height=None width=None vae_scale=None | s2=None s3=None (trace) | 1.74 GiB | 0 | single_gpu, 19 683 MB — OOMed |
| fixed | height=480 width=832 vae_scale=8 | s2=60 s3=104 | 35.07 GiB | 61 | lazy_sequential + component-level VAE tiling |

**The law:** one authority for the container's own spatial answers — `core/runtime/resolution/
container_size.py` (`vae_scale_factor`, `container_output_size`), the executor's own walk
extracted; the executor, the CLI's plan request and the serving engine's plan request read it.
A plan is budgeted under the request the flow executes; a request-side fact still outranks
the container. For the Mac: the same two functions apply on Metal unchanged (they read the
topology and the manifest, never a device); the Apple plan for any video container whose
request names no resolution was budgeted at the VAE's trace extent until this lands.
Branch `prism-plans-at-the-containers-resolution`; the judged run follows.

## 2026-09-21 — bucketed request-dependent key dimensions: the ladder chosen by measurement (CUDA, 16 GB class)

The owner's decision (15:40): request-dependent dimensions are bucketed in the launcher's
key; the bucket selects the configuration, the kernel runs the true size with its masks.
`tools/bucket_loss.py`: one full autotune sweep per size (the engine's own bench with the
consensus screen, directory off, private replay cache, the hardware profile bound so the
keys are the live ones), then each ladder evaluated offline — every size served the
configuration certified for its bucket's TOP, loss = time(bucket config) / time(per-size
optimum) - 1. Card 0 (Tesla V100 16 GB), 1290/877 MHz.

| kernel, fixed shape | sizes swept | ladder | buckets | median loss | max loss |
|---|---|---|---|---|---|
| matmul M, N=K=2048 fp16→fp32 (TinyLlama's prefill key form) | 88 (M 5..4096) | exact | 88 | 0.0 % | 0.0 % |
| | | **L16**: 16-step to 256, 32 to 1024, 128 to 8192 | 31 | 0.0 % | **0.0 %** |
| | | powers of two (64-step 64..512) | 14 | 0.0 % | 26.9 % |
| bmm M (=N), B=32, K=64 fp32 (the SDPA math scores) | 88 | exact | 88 | 0.0 % | 0.0 % |
| | | L16 | 31 | 0.0 % | 10.5 % (all of it in the buckets 16, 32, 48, 64) |
| | | **Lmix**: exact below 64, L16 above | 87 | 0.0 % | **0.0 %** |
| | | powers of two | 14 | 0.0 % | 15.0 % |

**What decides it:** above 64 the 16/32/128-step ladder costs nothing measurable on either
kernel; below 64 the batched GEMM's optimum moves with every size (5.6–10.5 % lost inside a
16-wide bucket), the plain GEMM's does not. The ladder retained for the branch is Lmix:
exact under 64 (where the GEMV path already serves M ≤ 4 unkeyed), then L16. Owed: the
convolutions' H/W at the tile lattice, and the 32 GB class. Raw sweeps:
`nbx/campaigns/2026_09_21_bucketed_keys/*.json`.

## 2026-09-21 — the upscaler "single tile" regression, checked on CUDA: it does not reproduce

The Mac's report (16:43): the whole upscaler family emits one tile — real-esrgan-x2 at 448
returns one 64 px tile upscaled to 128 instead of 896², in both modes; bisect names the merge
78784abe. The owner's instruction: test the merge's two parents separately.

| tree | mode | rc | output |
|---|---|---|---|
| main HEAD e3823004 | compiled / triton | 0 / 0 | 896x896 / 896x896 |
| 9000b7aa (main's parent of the merge) | compiled / triton | 0 / 0 | 896x896 / 896x896 |
| 6f2cd5b7 (the Mac's parent of the merge) | compiled / triton | 0 / 0 | 896x896 / 896x896 |

Card 1 (V100 16 GB), the 448² asset (`benchmarks/assets/apple_448.png`), single_gpu plan, the
component-level tiling engine at the 64 px trace tile (49 tiles, accumulated). Not
reproducible on CUDA on either side of the merge. The merge's 119 files touch none of
`core/module/tiling_engine.py`, `core/runtime/executor.py`, `resolution/output_extractor.py`
or `core/prism/solver.py`; the word "fold" names nothing in the engine. What the CUDA side
needs from the Mac to go further: the exact SHA it measured, its plan's strategy and
`[OpTiling]` lines for that run, and whether a tree without the merge's Metal-side files
(`metal_backend.py`, `triton_ext_driver.py`, `launcher.py`) still shows it — the tiled path
those touch is Metal's. Script and logs: `nbx/campaigns/2026_09_21_upscaler_regression/`.

## 2026-09-21 — Sana-1600M-MultiLing at 1024, the drift item, R29 first

The retrace gate read 17.66 dB (old container) and 17.25 dB (new) against the vendor at the
calibration request. Both artefacts, looked at: the vendor's render (vendored diffusers
0.36.0, fp16, seed 42, 20 steps) and NeuroBrix's triton render (seed 42, 20 steps, 57.7 s on a
16 GB card) are each a coherent, well-lit red apple on a wooden table — different
compositions (the vendor's on a plain dark-red backdrop, NeuroBrix's against dark planks with
a greener stem). Under R29 neither is degenerate; 17 dB between two different valid
compositions is the signature of the same-seed-is-not-the-same-noise class (the vendor draws
its initial noise in fp16 through its own generator, NeuroBrix in its own dtype and stream),
not of a corrupted stage. The measurement that decides it is queued: the vendor started from
NeuroBrix's own initial latent (the pipeline accepts `latents=`), PSNR against NeuroBrix's
render from that latent — high means the engine reproduces the vendor and the item is the
noise's; low means a stage diverges and the per-stage boundary walk follows. Sana's two ATen
arms could not be judged on main until d47c5e0d: the in-place-unary interceptor handed a
torch tensor to the NBX wrapper at aten.relu::0 (the Mac's fix f154dc39, cherry-picked and
proven by its four cells on CUDA).

## 2026-09-21 — the Mac's interceptor fix (f154dc39) proven on CUDA: Sana's two ATen arms, red then green

On main before it, Sana-1600M 1024 in `--sequential` and in compiled mode both died at
`aten.relu::0` — `'Tensor' object has no attribute '_device_idx'` — the in-place-unary
interceptor handing a torch tensor to the NBX wrapper below its size threshold (the same
class Wan T2V's ATen arm met under component tiling). Cherry-picked as d47c5e0d; its four
cells pass on CUDA; the two arms then render: sequential rc 0 in 18.9 s, compiled rc 0 in
20.3 s (card 2, 16 GB). Per the two-way merge rule: beneficial on CUDA, measured.

The census re-run after the refusal merge (main c7ef4fa4) holds both proofs: TinyLlama 6 = 6,
Sana 1024 58 = 58, under the one-card profiles — no key was harvested through a symbol that
fell back to its trace value on those two.

### Sana 1024, step by step from one initial latent (17:51, card 2)

| step | t | NeuroBrix triton mean / std | vendor mean / std |
|---|---|---|---|
| 0 | 999 | 0.0044 / 0.9756 | 0.0043 / 0.9757 |
| 5 | 899 | 0.0100 / 0.8804 | 0.0088 / 0.8807 |
| 10 | 749 | 0.0171 / 0.7711 | 0.0145 / 0.7706 |
| 15 | 499 | 0.0293 / 0.7365 | 0.0248 / 0.7365 |
| 19 | 136 | 0.0499 / 1.0384 | 0.0415 / 1.0420 |

PSNR vendor(seed 42) vs NeuroBrix(seed 42): 14.60 dB; vendor FROM NeuroBrix's initial latent
vs NeuroBrix: 24.58 dB; NeuroBrix twice: 94.60 dB. The loop tracks the vendor at every step
within 1e-3 on the state's std and a mean that drifts by 8e-3 over twenty steps — the
signature of two numeric paths (fp16 kernels, different accumulation orders) walking the same
schedule, not of a stage that diverges. Ten of the seventeen dB were the noise class (a seed
is not a noise); the remaining gap accumulates smoothly. Under the drift-origin rule the
origin class is KERNEL/SCALE, not policy or discrete: no single op to name. Both artefacts
are coherent apples. What would close the item as a judged replacement: this table and the
two artefacts beside the deprecated container's on the hub — the publish decision is Hocine's.

### The convolution's width, swept (card 0, 16 GB): the ladder costs 37 % in two buckets of thirty

conv2d_forward_kernel, [1, 128, 256, W] x [128, 128, 3, 3], fp16, 37 widths 64..1024. The
16-step ladder (Lmix = L16 here, no width under 64) loses 0.0 % in 28 buckets and 37.4 % /
35.9 % in the buckets whose top is 128 and 144: at those tops the optimum flips to
BLOCK_SIZE_OUTF=32 while every interior width prefers OUTF=64, so the width 120 served the
top's setting pays 37 %. Powers of two lose the same 37.4 %. Two more facts from the same
sweep: (1) widths that are multiples of 16 are FAST and widths ≡ 8 mod 16 are slow by
40–60 % under their own optimum (72: 1.47 ms against 80: 0.99 ms; 88: 1.82 against 96:
1.21) — the tile lattice law read from the kernel's side; (2) above 160 the optimum is stable
(OUTF=64 to 512, 32 at 1024) and the ladder costs nothing. What the measurement says: for a
convolution's spatial extent the bucket's TOP is not always the bucket's best representative;
the two losing buckets sit exactly where the optimum flips. Options, for the owner: keep the
ladder and certify a convolution bucket at the setting that minimises the loss over the
bucket's ends (two synthetic sweeps per bucket instead of one), or keep the convolution's
spatial keys exact below 160 (tiled extents are 16-multiples by the lattice law and rarely
land there; the untiled latents of the video VAEs do). The branch applies the profile's
default ladder to every dimension until that is decided; a profile may declare `height` and
`width` ladders of their own.

## 2026-09-21 — the Mac's branch merged on a worktree and its engine changes proven on CUDA (the two-way merge rule)

`merge-metal-first-light` = main 688b86be + origin/metal-first-light f3828311, merged clean
(the interceptor fix was already cherry-picked). Its five new cells: 7 passed, 4 skipped on
CUDA (conv bias in place, the interceptor, the MoE table) and 3 passed, 1 skipped without a
card (zero3 selection on unified memory, the Apple lattice). Judged, card 0 (16 GB):

| engine change | proof on CUDA | result |
|---|---|---|
| triton/moe.py Metal pinned-address tables and block size | granite code request, triton, bytes | sha e7725da9d641 = the vendor's unfused forward — inert |
| kernels/ops/conv2d.py + wrappers: the conv bias rides in place (one output per biased conv) | Sana 1024 triton, bytes and wall | sha 9a1fc0589057 = main's bytes; 47.0 s against 57.7 s — inert on bytes, beneficial on wall |
| core/flow/audio_llm.py: context embeds join the context device | Voxtral, jfk 11 s, compiled | rc 0, the transcript exact ("And so, my fellow Americans, ask not what your country can do for you…") — inert |
| core/prism/solver.py + strategies/zero3.py: zero3 never selected where its offload frees nothing | the branch's cell + the catalogue census of 09-21 (59/59 plans identical) | inert on this rack (no zero3 placement on four cards) |
| apple_m4_pro.yml lattice unit 16 + Apple certified directories | Apple data; the CUDA solver reads volta.yml's own lattice | no CUDA path reads them |

## 2026-09-21 — the doctrine end to end on the bucketed keys, and the DiT at step 0

**Census → certification → served, on branch `bucketed-autotune-keys`, card 2 (32 GB):** the
census of TinyLlama at the family's calibration request (33 tokens; no card) — 6 keys;
`neurobrix autotune certify --profile volta --census …` into a private directory pinned to the
card: 6 shapes certified, 0 excluded, 0 failed, 0 unreachable, every file re-reads; the served
run at a DIFFERENT request (19 tokens) swept all six — correctly: under 64 the key is exact and
33 does not serve 19. Verification serves the request the census covered, by construction;
the served run at the census's own request follows.

**Wan T2V, the DiT at step 0 (card 3):** the vendor's `WanTransformer3DModel` run on NeuroBrix's
exact step-0 inputs (the latent, t = 999, the positive text states) against NeuroBrix's own
prediction without guidance:

| | mean | std | absmax | channel means [:4] |
|---|---|---|---|---|
| vendor cond prediction | -0.0029 | 1.0813 | 4.594 | -0.078, -0.521, -0.448, -0.126 |
| NeuroBrix prediction, guidance 1 | -0.0035 | 1.0802 | 4.582 | -0.055, -0.525, -0.444, -0.134 |

max |diff| 0.128, relative L2 0.0135, cosine 0.99991 — the DiT forward is faithful at batch 1
(fp16 numerics). NeuroBrix's guided prediction at batch 2 is not (std 1.32, channel means of
±1, the field), so the batched [uncond, cond] forward is the suspect on a graph traced at
batch 1 (the 08-29 class); `NBX_CFG_SEQUENTIAL=1` runs the two halves as batch-1 forwards and
decides it on the next free card. The conditioning length (226, the graph's) stands as the
vendor's contract.

## 2026-09-21 — the shared cache against the hub, container by container (the owner's 18:19 rule; for the Mac)

**The instrument.** `tools/hub_cache_diff.py` (main be835369, both remotes): every container of
the shared cache (`~/.neurobrix/cache`, 59 with a manifest) against the object the hub serves
under its slug — sha256 of each `components/*/graph.json` and of `topology.json`, and the build
time in `manifest.json` on each side. The hub side is read without downloading a container: the
store honours HTTP Range (206), the `.nbx` is a STORED zip64, so the central directory and the
JSON members are fetched by byte range (self-test on the 21.9 GB PixArt build: 34 members, 67.7 MB
read, every hash equal to `zipfile`'s; the store dropped one 1.5 GB member stream at 0 B/s, so
reads are 8 MB chunks, retried). The page: `docs/reference/hub-cache-diff.md` (every hash, both
sides). Re-run the tool rather than trusting the page's date.

**Measured 19:20 UTC.** 36 IDENTICAL · 12 CACHE_NEWER (CogVideoX-2b, Flex.1-alpha, MiniCPM-o-4_5,
PixArt-Sigma-XL-2-1024-MS, PixArt-XL-2-1024-MS, Qwen3-Omni, Sana_1600M_1024px_MultiLing,
Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B, hat-l-x4, orpheus-3b-0.1-ft-snac, swinir-classical-x4) ·
2 HUB_NEWER (PixArt-XL-1024: the cache's old 05-20 copy beside the 09-21 build; Wan2.1-I2V-14B:
the hub's 06-26 build, the cache's 06-15 — the download is queued after the TRIM) · 7 NOT_ON_HUB
(local variants: TinyLlama-v1.0 ×3 incl. int4, Qwen3-Coder int4g128, orpheus-3b-0.1-ft,
real-esrgan-x2, real-esrgan-x8) · **2 HUB_OBJECT_CORRUPT**: `XPixelGroup/HAT-S-x4` serves
55 653 412 bytes of ZEROS against a 54 654 466-byte record, and
`ibm-granite/granite-3.1-1b-a400m-instruct` serves 2 523 531 200 bytes against a 2 791 966 656-byte
record whose first bytes are not a zip header. Neither installs. On real-esrgan-x2: the internal
and the public hub list the same 48 models and NEITHER has an x2 entry under any slug — the stale
copy the Mac diagnosed did not come from the hub's catalogue as it stands; `74a2d7ea` is the
sha256 of the cache's `components/model/graph.json` (the 20 September retrace).

**Verified and therefore published at the first window** (`--verified`, by hand from the judged
records; the tool never decides): PixArt-Sigma-XL-2-1024-MS (30.03 dB vs vendor), PixArt-XL-2-1024-MS
(gate PASS; 41.35 dB on 09-07), hat-l-x4, hat-s-x4, swinir-classical-x4 (gates PASS 2026-09-21),
granite-3.1-1b-a400m-instruct (battery goldens native + triton on the served stack). **Not
published, and why**: CogVideoX-2b (09-07 gate NEEDS_EXPLANATION: both arms within the vendor
gate, the old arm closer), Flex.1-alpha and MiniCPM-o-4_5 (09-07 gates FAIL — the cache holds
those failed builds; the hub's older objects are the last judged ones), Qwen3-Omni (one mode
judged so far), Sana_1600M_1024px_MultiLing (the drift item, the owner's decision), Wan T2V
and VACE (no gate on record; Wan T2V's tiled VAE seams are a stage-three item), orpheus-snac
(no gate on record), real-esrgan-x2/x8 (new entries written, no fidelity gate on record).
The publication runs from `after_trim_v2.sh` at 01:05 UTC: a 5-byte then a 50 MB write probe
through the internal entry point, then `hub_cache_diff.py --publish --verified` (a corrupt hub
object is replaced like a stale one; a container without a staged `.nbx` is re-packed from the
cache and its members re-hashed against the cache before the upload).

**Until then the shared cache is canonical for both machines**; the Mac reads its graphs from
there. The doctrine that governs everything from 19:07 today is in this rack's CLAUDE.md
(certified autotune, three stages) and `docs/internal/_session_current.md`.

## 2026-09-21 19:45 — the bucketed autotune key is on main (787796d3, both remotes): for the Mac

**What landed.** `matmul_kernel` and `addmm_kernel` key on `M_BUCKET`; `baddbmm_kernel` on
`M_BUCKET, N_BUCKET`; the kernel still runs the true size. The ladder is the vendor profile's
`autotune.buckets.default` (`config/vendors/nvidia/volta.yml`): exact ≤ 64, step 16 to 256, 32
to 1 024, 128 to 8 192, 512 beyond — measured 0.0 % median and maximum loss on both V100
classes against the per-size optimum (`tools/bucket_loss.py`, tables above). A profile that
declares no ladder keeps the exact key: the Apple profile must declare its own, measured, before
its census is taken in bucketed form. **The convolutions keep exact spatial and batch keys**
(`conv2d_forward_kernel`, `depthwise_conv2d_kernel` unchanged): the same ladder on a
convolution's width lost 37.4 % / 40.2 % in the two buckets where the kernel's optimum flips
(OUTF 64 → 32 at tops 128 and 144, C=128, H=256) and 0.0 % at C=256; a convolution's extents
are bounded by resolutions and tile edges, not by prompts. Brick: `kernels/autotune_bucket.py`
(`bucket`, `parse_ladder`, `ladder_for`, `bucket_of`); cells:
`tests/unit/kernels/test_a_request_dimension_buckets_on_the_profiles_ladder.py`.

**A census defect fixed with it, which the Mac's census may share.** Behind the census door no
driver names the vendor profile; `active_vendor_profile()` resolved EMPTY and every key the
shadow recorded was exact (matmul M 226 where the launcher keys 240), the SMEM budget and the
config spaces unread, the shadows ten times slower. `census.install(hardware=…)` now binds the
launcher target from the hardware profile's first device (brand + compute capability) and
clears the cached vendor profile — an NVIDIA/AMD profile only; a device whose capability is
not a number (Apple) keeps its driver's answer, so the Mac should check that its shadow sees
`apple_silicon.yml` (one recorded key of a request-dependent kernel read against the
launcher's form is the test). Cell: `tests/unit/kernels/test_the_census_shadow_carries_the_profiles_target.py`.

**What the fixed census measured here** (from the shared cache, both classes, ~4 min each):
16 GB — 34 ok, 8 failed shadows, 17 retrace, 675 entries, 268 to certify; 32 GB — 37 ok, 5
failed, 17 retrace, 722 entries, 313 to certify. The retrace queue (symbolic graph only, stage
one): the Mac's eight plus Flex.1-alpha, Ming-Lite-Omni-1.5, MiniCPM-o-4_5, PixArt-Sigma-XL-1024
(the old container), Qwen3-Omni, Qwen3-VL, Sana-1600M-MultiLing (the old container),
Wan2.1-I2V-14B, granite-3.1-1b-a400m. Certification of both classes runs now, pinned per card
(`nbx/campaigns/2026_09_21_census/certify_class.sh`).

## 2026-09-21 20:25 — the batched GEMM's contraction bucketed: the measurement (both classes)

openaudio's census (the fixed shadow) recorded 2 135 `baddbmm` keys in one request: 85 distinct
M buckets, 85 N buckets and **2 049 distinct K** — the contraction of the attention's second
product is the key length, walked one value at a time by the decode. An exact K cannot be
certified for a decode. Swept `bucket_loss.py --kernel bmm --dim K --fixed B=32,M=1,N=64`,
135 sizes 1..4 096, `NBX_AUTOTUNE_CERTIFIED=off`, one card per class, alone on the card:

| class | ladder | buckets | median loss | max loss | where |
|---|---|---|---|---|---|
| 16 GB (card 0) | Lmix (exact ≤ 64, 16/32/128/512) | 123 | 0.0 % | 20.0 % | tops 128 (K=120: 5.6 %), 176 (168: 20.0 %), 192 (184: 14.3 %) |
| 16 GB | L16 | 64 | 0.0 % | 20.0 % | the same buckets |
| 16 GB | powers of two | 16 | 0.0 % | 23.5 % | |
| 32 GB (card 2) | Lmix | 123 | 0.0 % | 10.5 % | top 112 (K=104) |
| 32 GB | powers of two | 16 | 0.0 % | 21.7 % | |

Decision by measurement: K enters the `baddbmm_kernel` key as `K_BUCKET` on the profile's
default ladder (main, with this commit). The loss is confined to a few 16-step buckets where
BLOCK_K's optimum flips; every other bucket costs nothing, and the alternative is a key nobody
can certify. Records: `nbx/campaigns/2026_09_21_bucketed_keys/bmm_K_M1_N64_{16g,32g}.json`.
The Mac's profile needs the same measurement before its census in bucketed form.

## 2026-09-21 21:20 — the memory law and the tiling standard on main (f5a88bff): for the Mac

**The law** is one brick, `core/prism/memory_budget.py`: a SHARED pool (unified memory, a
device driving a display, a card another process holds memory on, a reading that could not be
taken, host RAM) is budgeted at its free reading rounded DOWN onto the commercial ladder —
configuration, `PRISM_DEFAULTS["memory_ladder_gb"]`, 4 GB to 512 GB — never a value off the
ladder; a DEDICATED card nothing else uses is used whole, less only the runtime's own context.
The decision is read from the device (`autodetect.device_sharing`: display activity, other
processes' memory, own context; the profile's `has_unified_memory` for the pool kind). The tile a
plan cuts derives from the RUNG by a fixed rule — a dedicated card's nominal rung, a shared pool's
free rung — and is a pure function of (component, rung): the same 2 048-pixel upscale cuts 320 px
tiles at 4 GB and 640 px at 16 GB, edge tiles padded to the canonical size. Cells:
`tests/unit/prism/test_the_memory_budget_is_one_law_for_every_device_kind.py` (14 red on main
before the brick: idle 16 GB and 32 GB, partly held, a display, unified 18 186 / 10 099 MB,
unreadable, host RAM; the door). **The Mac proves the unified side**: on the M4 Pro
`is_shared` must answer from `has_unified_memory` and the budget must read the host's available
figure rounded down (the `_device_reading` branch for a unified device). Measured here before the
brick: an idle V100-16GB was budgeted at 12 288 MB and an idle V100-32GB at 24 576 MB.

**The census enumerates rungs**: `tools/certified_census.py --rungs ladder` (default) runs every
model's shadow at every rung up to the profile's card capacity through the
`NBX_PRISM_BUDGET_MB` door; a spatial family's tiling probe (its family YAML, `census.tiling_probe`)
is the request large enough to tile. The Mac's census in bucketed form should run with the same
flag once the Apple profile declares its ladder.

**Where the shadow's value-reading fixes live** (the Mac aligns on them, never rewrites):
`src/neurobrix/kernels/census.py::install` — an integer read answers 1 and a host read ones; a
value born on the host (`NBXTensor.from_numpy`) is kept and read back as written; the lazy
strategy's execution device resolves to the shadow's card; `device_utils` rebound to no-ops in
every loaded module; the dual-AR sampler draws token 0; a key with a negative extent is refused.
Cell: `tests/unit/kernels/test_the_census_shadow_carries_the_profiles_target.py`.

**b23105fe** (never offload to the host for memory on a unified device): the CUDA inertness proof
is owed once the certification rounds free a card; the unified outcome changing from a refusal to
a streaming strategy is the Mac's to prove.

## 2026-09-21 21:55 — b23105fe on main: the CUDA inertness proof

Merged from a worktree carrying the machine's ignored pointers (`.nbx_registry`,
`config/hardware/default*.yml`, `forge/`). Three ways: (1) the door — `_device_is_unified`
answers False for every card of every profile here, so the guard cannot fire; (2) the Mac's
cells pass on this tree (2 passed, the discrete arm keeps the host offload); (3) a Prism plan
census of the 59 installed containers on the four-card profile and both one-card profiles, main
against main + b23105fe, each run twice: **0 plans differ** (48 planned, 11 video containers
refused identically by the census script's own request naming no frame count). A first pass
read TWO moved plans (the old PixArt containers, `single_gpu` → `single_gpu_lifecycle`) — the
worktree had no `.nbx_registry`, the registry-flag lookup resolved nothing, PixArt's VAE lost
its `fp16_conv_cascade_safe` and planned fp32. The difference was the harness, run one side
twice before naming a culprit, and the pointer is now part of every worktree here. The unified
outcome (a refusal becoming a streaming strategy, since the engine never refuses for memory)
is the Mac's to prove. Records: `nbx/campaigns/2026_09_21_mac_proofs/`.

## 2026-09-22 00:50 — the certifier that held 155 GB, and the shadow that walked every token: for the Mac

Four measurements the owner took on this rack at 23:32 named two host burns that made the
cards idle while certification work existed. Both are fixed at the source and both have the
same shape on Metal, so the Mac should read this before its own rounds.

**A fp64 oracle computed WHOLE on the host.** `autotune_certify` proved a matmul key by
building the reference product in float64 with numpy, at the key's full shape. The 32 GB
matrix round reached `44 544 x 3 072 x 8 192` and the process held **155.5 GB of resident
host memory** (read at 23:32:41 on pid 4023027) while its card sat at 0 %; the rack's 251 GB
and its 7 GB of swap were both full, load 74.3 on 80 cores. The oracle is now WINDOWED by
rows above a cap of 2e9 multiply-accumulates (`ORACLE_MAX_MACS`): the first, middle and last
row windows of the product are computed and compared, the batched bias windowed with them
(`RowWindowedOracle`, `_row_windows`, `_matmul_oracle_fn`, commit `ec86f575`). A windowed
oracle proves the same thing a whole one does for a GEMM — every output row is the same
inner product over K, so a wrong BLOCK_K or a wrong accumulation order shows in any row —
and the deviation it reports is measured on real rows, not sampled values.

**A BLAS pool spinning behind a shadow.** One chatterbox census shadow ran at **4 235 % CPU**
(64 threads at ~24 % each) and another at 2 843 % with 16.3 GB resident: the shadow's own
host arithmetic is small, but OpenBLAS opened a pool per process and the pool spun. With
`OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1` the same shadow runs at 93–99 % CPU on one thread
and records the same keys; every shadow the census launches is single-threaded now
(`tools/certified_census.py::shadow`, commit `ec86f575`). The census must never starve
certification, and 24 shadows at 64 threads is how it did.

**A shadow that walked every token.** The census walked a decode one position at a time:
chatterbox's 2 048 speech tokens cost **541.7 s** for the keys of about 110 distinct lengths.
Under the bucketed keys a decode's shapes change only at the bucket tops of the context or
cache length, so the shadow now skips to each top and the sampled token stands for the
positions between (`census.pace`, the three triton flows, `kv_cache.skip_positions`, commit
`2e0852d8`): **37.9 s, the same decode keys**. TinyLlama: 20 s to 10 s, its 6 keys unchanged
(all under 64, where the ladder is exact). A live run never skips — `pace` answers 0 outside
the shadow, and a cell pins that.

**An extent a shadow cannot learn.** What the vocoder receives is the number of speech tokens
that survived a filter on the tokens SAMPLED, and a shadow has no values: two runs of
chatterbox gave 1 716 and 1 856 tokens and therefore two complete sets of convolution keys.
A census taken from one run certifies one speech length. `census.walk_extent(lo, hi, run)`
runs such a stage at every KEY CLASS of the extent — the ends, then the midpoint of any pair
whose recorded key sets differ, until the pair is adjacent; every kernel's key is a monotone
step function of the extent, so a pair with one key set brackets a range with that key set.
Synthetic proof on this profile's own ladders: every class met, none missed, in under four
runs per class. The door is `NBX_CENSUS_EXTENTS=1`; a walk refused at every extent RAISES
rather than reading as censused (commit `11045c4d`).

**A key whose extent is zero.** Walking that extent to its bottom recorded convolution keys
with `in_width` 0 — a four-token speech through a vocoder. No tensor has a side of zero, no
certifier can synthesise one, and no request forms the launch; the served directory already
carries one (`conv2d_forward_kernel.fp32`, a key whose `batch_dim` is 0), so a census recorded
them before anyone read one. Refused now where a negative extent already was, on the key
positions that are extents or divisors and not on a padding, which may legitimately be zero
(`autotune_certified.EXTENT_POSITIONS`, `degenerate_extent`, commit `69df7418`; seen failing
on two injections). **The Mac should check its own directory for the same entries.**

**Open, and NOT landed: the width of a one-row convolution.** A convolution whose spatial
extent is one row is a 1-D convolution over a sequence — a vocoder's samples, a mel
spectrogram's frames — and that extent is the request's, bounded by nothing: the served
directory holds **483 one-row entries at 201 distinct widths up to 210 998** (conv2d fp16,
of 798 entries) and one chatterbox run reaches 444 721. So the doctrine's second reason for
exact convolution keys — that their extents are bounded by resolutions and tile edges — is
false for this class, while its first reason (a ladder lost 37–40 % where the conv optimum
flips) was measured in the 2-D regime and says nothing about H = 1 with W huge, where the
grid is `cdiv(N*W, BLOCK)` and the optimum is EXPECTED to be flat in W. An expectation is not
a measurement, so the change sits on the branch `one-row-conv-width-bucket` (17f58975, both
remotes) until `tools/bucket_loss.py --kernel conv2d --dim W` has run on a free card of each
memory class. The decision matters beyond this rack: it sets the certification budget for the
whole audio family. One chatterbox vocoder walked over its speech length demands **5 058
keys** — 2 889 convolution, 1 818 baddbmm, 329 matmul — of which 125 of 145 distinct one-row
widths fall below 8 192, where the ladder is fine. The Mac's profile needs the same
measurement before it certifies any audio model.

## 2026-09-22 02:40 — the one-row convolution width: the measurement, on both classes; and where a shadow's coverage ends (an answer for the Mac)

**The measurement the branch waited for.** `tools/bucket_loss.py --kernel conv2d --dim W`
(the tool now takes `kh`/`kw`, so a 1-D convolution can be swept at all), directory OFF,
private replay cache, alone on its card, fp16, kernel (1,3), padding (0,1), batch 1:

| class | C_in=C_out | sizes | ladder | buckets | median loss | max loss |
|---|---|---|---|---|---|---|
| 16 GB (card 1) | 128 | 26, 1 024..524 288 | Lmix / Wq / Woct | 25 / 22 / 14 | 0.0 % | 0.0 % |
| 32 GB (card 2) | 128 | the same 26 | Lmix / Wq / Woct | 25 / 22 / 14 | 0.0 % | 0.0 % |
| 16 GB (card 1) | 512 | 19, 8 193..524 288 | Woct | 6 | 0.0 % | 0.0 % |
| 32 GB (card 2) | 512 | the same 19 | Woct | 6 | 0.0 % | 0.0 % |

All four sweeps agree. The 512-channel pair is the strongest of them: its six octave buckets
each hold three or four measured widths, every one of them served by the configuration proven
at the bucket's top, and none loses anything.

The sizes include the widths a chatterbox run actually makes (3 206, 27 408, 137 040,
210 998, 411 121, 444 721). The buckets that carry the evidence are those holding two to
four measured widths, each represented by its own TOP — the configuration proven at the top
is optimal at every width the bucket serves, including one just above the bucket's floor
(411 121 served by 524 288's). The stress case is the SMALL widths, not the large: at
W = 1 024 with 128 channels the grid is sixteen blocks on eighty multiprocessors, where
occupancy is most sensitive to the width, and it costs nothing there either.

**Landed on main (803ab6cc)**: `conv2d_forward_kernel` and `depthwise_conv2d_kernel` key
`in_width_key`/`out_width_key`; for a convolution whose spatial extent is ONE ROW these are
the input width's bucket top on the profile's `autotune.buckets.W` rows and the output width
the convolution's own arithmetic gives from that top, so a certifier synthesising at the top
forms the very key the census recorded. Two-dimensional convolutions keep their exact
extents and every 2-D entry keeps serving, because the key positions did not move. The
directory's 483 one-row entries (of 798, conv2d fp16, at 201 distinct widths) are unserved
and re-certified at their tops. Effect on the census: chatterbox censuses **333 keys** where
walking its speech length unbucketed demanded **5 058**. **The Mac needs this measurement on
its own profile before it certifies any audio model**, and the ladder is data
(`config/vendors/nvidia/volta.yml`, `autotune.buckets.W`), so an Apple profile writes its own
rows rather than inheriting these.

**The certifier's real cost, measured and fixed.** Not the oracle and not the bench: three
py-spy samples of a conv round all landed in `host_values → numpy → to_cpu → memcpy`. A
windowed oracle refuses to compute the whole reference but was handed the whole RESULT,
because the windows were cut AFTER the crossing. Now cut on the device
(`WindowedOracle.device_slices`, `RowWindowedOracle.device_slices`, `deviation_against`,
12b4d271). Measured on this rack's own proofs, same card, same census:

| round | oracle windows cut | n proofs | `runs` median | `bench` median | `oracle` median |
|---|---|---|---|---|---|
| four | after the crossing | 452 | 6.4 s | 2.2 s | 2.4 s |
| five | on the device | 84 | **1.2 s** | 1.0 s | 2.3 s |

`runs` was the dominant phase and it fell 5.3×. The key populations differ between the two
rounds, so this is the phase cost, not a controlled end-to-end A/B.

**Where a shadow's coverage ends — the `aten::embedding` gap the Mac characterised.** The
census shadow replaces two surfaces and nothing else: the `DeviceAllocator` (allocation, the
whole driver surface — syncs, streams, events, peer access, pinned host memory, device
queries) and the kernel launcher, plus, since 2026-09-21, the device utilities rebound in
every loaded module, a pointer-keyed table for host-born values, and the samplers. A path
that resolves a **runtime** rather than allocating or launching is NOT covered, and that is
the class both machines have hit: orpheus asked the driver directly here ("No CUDA GPUs are
available") and Metal's embedding asks for a GPU runtime there. On CUDA `aten::embedding`
goes through the covered seams, which is why it censuses here and not there — the difference
is the backend's dispatch, not the model. The structural answer is to shadow the RUNTIME
RESOLUTION itself rather than each caller, so a backend that resolves a runtime gets a shadow
one; I cannot write or test that on Metal, and it is the Mac's own seam to place. What this
rack can promise is that the door stays `CUDA_VISIBLE_DEVICES=`, so any path we have not
covered fails LOUDLY rather than reaching a card.

## 2026-09-22 04:05 — the ladder's open tail is too fine for a request-scale dimension (owed: the measurement)

Walking chatterbox's vocoder across its speech length with the one-row convolution width
already bucketed still demands **5 748 keys** — 3 072 convolution, 2 110 baddbmm, 566 matmul —
and the reason is no longer the convolutions. The vocoder's batched GEMM keys on the WAVEFORM:
`baddbmm M_BUCKET=1 104 384 N_BUCKET=1 K_BUCKET=9`, then `1 105 920`, then `1 107 456` — the
profile's default ladder is open above 8 192 with a step of **512**, so a dimension that
reaches **1 963 520** is cut into roughly four thousand buckets. Measured over the walk:
**1 522 distinct M values, 1 400 of them above 8 192**, for ONE model.

The doctrine's ladder was decided by measurement — exact under 64, 16 to 256, 32 to 1 024,
128 to 8 192, 512 beyond, 0.0 % loss on both V100 classes — but that sweep ran to about
4 096. Nothing has ever measured the 512-step tail at 10^5 or 10^6, where a GEMM's grid is
saturated many times over and the configuration is expected to stop depending on M, exactly
as the one-row convolution's did in W. Expected is not measured, so this is written as OWED,
not decided:

* sweep `bucket_loss.py --kernel bmm --dim M --fixed B=1,N=1,K=9` (the vocoder's own shape)
  and a second, fuller shape, over 8 192..2 000 000, on a free card of each memory class;
* evaluate the existing `Wq` and `Woct` tails against the 512-step one;
* if the tail is free, the ladder's open row becomes a coarse tail and every request-scale
  dimension — a waveform, a mel, a long context — collapses with it.

Until then the catalogue census is taken WITHOUT `--walk-extents`, so it carries the M values
of one speech length rather than four thousand, and the certification budget stays bounded.
**The Mac will meet the same tail**: its audio models key the same waveform dimension, and the
ladder is data (`autotune.buckets`), so its profile can carry its own tail once measured.

## 2026-09-22 04:12 — `metal-first-light` is on main (cc71b3d2, both remotes): the CUDA proofs

All 25 commits merged; `main...origin/metal-first-light` now counts 0 on the branch side.
The proof is a comparison, not an assertion: the SAME worktree was run at main and at the
merge, both behind the census door, and the difference accounted for line by line.

| | failing cells |
|---|---|
| main (db437f1e) | 28 |
| the merge | 46 |

Of the 18 new, **17 are cells that reach `generator_identity` directly and now need a driver**
— with a device visible, 28 of the 29 affected cells pass. That is the deliberate half of your
change: `generator_identity` RAISES where main silently defaulted to `"cuda"`, the default
that once labelled every Apple run as this rack's generator and refused all 945 Apple entries.
It is kept exactly as you wrote it.

**One was a real defect, and it is fixed**: `test_no_other_module_reaches_the_device_runtime_directly`
greps for `libcudart` with comments stripped, and the new docstring in `metal_device.py`
describing this CUDA rack tripped it. The gate now skips docstrings — ONLY docstrings, never
every string literal, because the name it hunts appears as a literal in the load it hunts
(`CDLL("libcudart.so")`) and skipping all strings would blind it. Seen failing on an injected
real load, green with it removed.

**One more, at the seam you added**: `_out_of_tree_backend_hash` let the launcher's driver
error escape, where the gate's own rule is that an unanswerable question refuses nothing. On a
machine with no device the question "which out-of-tree backend will generate the code" has no
answer; it now answers None and SAYS so once rather than swallowing it.

**The conflict, and how it was resolved**: `census.py`'s shadow value-read. Your extracted
`_shadow_item_value` had already taken this rack's integer-answers-ONE rationale, so the two
sides agreed on semantics and differed only in shape — your extracted form is kept, reading
the dtype through the single `_dtype_name`. main had already taken your dtype-by-name fix on
its own (db437f1e), after measuring that on this rack's Python 3.10 the old `str()` form
matched correctly and on 3.11 it would not: the defect is real and is yours, it simply cannot
be reproduced here.

**Engine proof**: a one-model census on main after the merge records the same six TinyLlama
keys and the same six served entries as before it.

## 2026-09-22 04:45 — the ladder's tail, measured three times: what the first two sweeps could not see

The tail is settled and the road to it is worth writing down, because two of the three sweeps
read 0.0 % on a ladder that in fact cost 5.2 %.

**Why the first sweeps could not see it.** `bucket_loss --evaluate` serves each size the
configuration proven at its bucket's representative — and the representative is the largest
MEASURED size in that bucket. If the sizes are spread so that each lands in a bucket of its
own, every size is its own representative and the loss is 0.0 % by construction, whatever the
ladder. The first two sweeps were spread that way. **A ladder is only evaluated by sizes that
SHARE its buckets, with the bucket's top among them**, and the arrangement is part of the
measurement, not its decor.

**The three sweeps** (matmul/bmm, `NBX_AUTOTUNE_CERTIFIED=off`, private replay cache, alone
on card 1, 16 GB class):

| shape | arrangement | ladder | median | max |
|---|---|---|---|---|
| bmm B=1 N=1 K=9, 26 sizes 8 192..2 097 152 | one size a bucket | octave | 0.0 % | 0.0 % |
| matmul N=K=512, 19 sizes to 524 288 | one size a bucket | octave | 0.0 % | 0.2 % |
| matmul N=360 K=180, 19 sizes to 4 194 304 | 2 a bucket | octave | 0.0 % | 2.9 % |
| matmul N=360 K=180, 27 sizes | **3 a bucket, top included** | quarter-octave AS SHIPPED | 0.0 % | **5.2 %** |
| matmul N=360 K=180, 24 sizes | **3 a bucket, top included** | the REFINED rows | 0.0 % | **1.6 %** |

The 5.2 % sat at the 10 240 bucket and the reason is arithmetic, not hardware: a quarter of an
OCTAVE at 10 240 is a quarter of the VALUE, so a request of 8 500 was handed a configuration
proven at 10 240. Beyond 65 536 the same sweep reads 0.0 % median and at most 0.6 %.

**What ships** (`config/vendors/nvidia/volta.yml`, `autotune.buckets.default`): 128 to 8 192
unchanged, then 512 to 16 384, 1 024 to 32 768, 2 048 to 65 536 — at most 3.1 % of each
bucket's top — then quarter-octave to 2 097 152 and a 524 288 open row, which is 0.7 % wide at
the catalogue's largest M (mochi's 77 414 400). A request of 8 500 now meets 8 704. Against
the 512-step tail it replaced: chatterbox's 1 400 buckets above 8 192 become of the order of
a hundred.

**The convolution-width ladder (`autotune.buckets.W`) is untouched** — its own sweeps measured
0.0 % median AND max in the same region, on both memory classes and two channel counts, with
up to four measured widths a bucket.

**Owed**: the same three-a-bucket arrangement on the 32 GB class (both its cards are
certifying and the sweep wants one alone). **For the Mac**: the ladder is data, so an Apple
profile writes its own rows — but the ARRANGEMENT lesson is not hardware-specific, and any
sweep of yours that reads 0.0 % with one size a bucket has measured nothing.

## 2026-09-22 04:55 — the tail's measurement is complete: both classes, both bucketed dimensions

The 32 GB half that the earlier entry owed, taken the same way (three sizes inside each
shipped bucket with its TOP among them, `NBX_AUTOTUNE_CERTIFIED=off`, private replay cache,
alone on card 3):

| class | dimension | shape | buckets | median loss | max loss |
|---|---|---|---|---|---|
| 16 GB | matmul M | N=360, K=180 | 8 | 0.0 % | 1.6 % |
| 16 GB | baddbmm K | B=1, M=1, N=64 | 6 | 0.0 % | 0.0 % |
| 32 GB | matmul M | N=360, K=180 | 8 | 0.0 % | **0.7 %** |
| 32 GB | baddbmm K | B=1, M=1, N=64 | 6 | 0.0 % | **0.0 %** |

The contraction costs nothing at any bucket on either class. The 16 GB maximum of 1.6 % sits
at the open row's 4 194 304 bucket and the 32 GB one at 1 048 576, both below the 5.2 % the
quarter-octave had at the knee and far below the 20 % the project already accepted when it
bucketed this same contraction in the small regime (2026-09-21, `bmm_K` on the default
ladder). The ladder is settled: `config/vendors/nvidia/volta.yml`, `autotune.buckets.default`.

Files: `nbx/campaigns/2026_09_21_bucketed_keys/{matmul_M_refined,bmm_K_tail}_{16g,32g}.json`,
`matmul_M_quarter_16g.json` (the 5.2 % that forced the refinement), and the three earlier
one-size-a-bucket sweeps kept as the record of what an arrangement can hide.

## 2026-09-22 04:57 — the convolution-width ladder at its knee: 0.0 % on both classes, and why the two ladders differ

The GEMM ladder needed narrowing just above 8 192 (5.2 % at the 10 240 bucket). The
convolution-width ladder keeps a quarter-octave there, and the earlier conv sweeps carried
only TWO sizes in that bucket — below the standard register 84 set — so it was re-measured to
the same arrangement: one-row convolution, C_in = C_out = 128, kernel (1,3), three widths
inside each bucket with its TOP among them, six buckets from the knee to 458 752.

| class | buckets | sizes | median loss | max loss |
|---|---|---|---|---|
| 16 GB (card 1) | 6 | 18 | 0.0 % | **0.0 %** |
| 32 GB (card 3) | 6 | 18 | 0.0 % | **0.0 %** |

Every bucket, both classes, including 10 240 — the one that cost 5.2 % on the GEMM. The
asymmetry is not an accident of measurement: a one-row convolution's grid is
`cdiv(N*W, BLOCK)` by `cdiv(C_out, BLOCK)`, so W only scales the first dimension and the
configuration stops depending on it as soon as the grid saturates; a GEMM near the knee is
still choosing its tile against M, and a bucket a quarter of an octave wide there is a quarter
of the value. The two ladders differ because the kernels differ, and each now says so with its
own numbers.

## 2026-09-22 06:45 — the 16 GB class is certified except seven keys that cannot fit it, and that is a census defect

A whole-census sweep of the 16 GB catalogue (`catalogue_16g_v8`, 2 888 entries, every kernel,
`--only-missing`) certifies everything and stops on exactly two kinds of key:

* **1 unreachable** — the known debt D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE, a key
  recorded under an older rule that no run will present again.
* **7 too large for the class** — six matmuls and one convolution, all from the video family,
  asking between 11.0 and 37.0 GiB of a 15.8 GiB card:

| kernel | shape | model |
|---|---|---|
| conv2d | batch 81, 96 ch, 722x1282 -> 3 ch, 720x1280 | Wan2.1-T2V-1.3B |
| matmul | M 77 594 624, N 3, K 128 | mochi-1-preview |
| matmul | M 34 603 008, N 3, K 128 | mochi-1-preview |
| matmul | M 19 398 656 / 8 912 896 / 5 767 168, N 512, K 256 | mochi-1-preview |
| matmul | M 2 621 440, N 2 048, K 512 | mochi-1-preview |

No certified entry is owed for any of them, because a 16 GB run never forms them: Prism tiles
the video decode long before the op is reached. They are in the census because **the shadow
plans an op at its GRAPH shape rather than at the shape the rung's plan would give it** — the
rung door (`NBX_PRISM_BUDGET_MB`) sizes the PLAN, but the op-level and component tiling that
the plan implies is not reflected in the recorded key. That is the next census-tool defect and
it is named here rather than guessed at: a census that records keys its own memory class
cannot reach is not yet a census of that class.

The certifier now says so with the arithmetic — bytes asked, card size, where the fix belongs
— and counts them apart from failures, so a round's `failed` count means what it says
(80d4ed18, three cells including the contrasting case of an allocation that is merely tight).

**For the Mac**: the same defect will appear wherever a model is tiled for memory, and its
symptom is a certification asking for more than the device holds. The ARITHMETIC is the tell —
if the key needs more than the card exists with, no run of that class formed it.

## 2026-09-22 07:25 — a correction to the entry above: the rung door works; op-level tiling is what misses these ops

The 06:45 entry said the shadow "plans an op at its GRAPH shape rather than at the shape the
rung's plan would give it". Half of that is wrong and the measurement says so.

The rung enumeration DOES change what a shadow records, exactly where it should. Comparing
each model's recorded keys at rung 4 096 against rung 16 384 on the 16 GB census:

| model | request | keys | differing |
|---|---|---|---|
| swinir-classical-x2 | ordinary (fits any rung) | 11 | **0** |
| real-esrgan-x4 | ordinary | 10 | **0** |
| Sana_1600M_1024px_MultiLing | ordinary | 58 | **0** |
| swinir-classical-x2 | the 4 096-pixel probe (tiles) | 11 | **18** |
| real-esrgan-x4 | the probe (tiles) | 10 | **20** |
| mochi-1-preview | ordinary | 30 | **18** |

A request that fits every rung records the same keys at every rung — which is correct, not a
defect — and a request that must be tiled records different ones. The door reaches the plan.

What is left is narrower and still real: at the 16 GB rung, mochi's plan does vary, and it
still forms a matmul of M = 77 594 624 whose operands need 37 GiB. So the gap is not the rung
door but **op-level tiling failing to cover these flattened projections** (`aten::mm` over
every pixel of a video, N = 3, K = 128) — Prism's `_try_op_level_tiling` is in the cascade and
does not catch them. The plan also prints `56 554 MB planned` on a card the door set to 16 384,
which is worth reading before anything else: if that figure is a SUM over a lazy_sequential
stream it is harmless, and if it is a peak the plan is over budget by three times. It is a SUM: `solver.py`
starts `total_mb` at zero and adds every component's allocation, and this plan's loading mode
is lazy — one component resident at a time, which is the whole point of that rung. So the
figure is harmless and says nothing about the budget. Checked rather than left hanging.

What remains, then, is exactly one thing: **op-level tiling does not cover these flattened
projections**, and that is the lead.

## 2026-09-22 10:45 — stage two is complete on both memory classes (for the Mac)

A whole-census sweep of each class, run TWICE on the 32 GB side from two different cards so
the second could only find what the first left:

| class | census | certified | failed | unreachable | too large for the class |
|---|---|---|---|---|---|
| 32 GB | catalogue_32g_v7 (3 211 entries) | **427** | **0** | 1 | 2 |
| 16 GB | catalogue_16g_v8 (2 888 entries) | all | **0** | 1 | 7 |

The 32 GB convolution family alone closed at **494 certified, 0 excluded, 0 failed, 0
unreachable**. What is left is not work:

* **the over-large keys** — mochi-1-preview's video projections (M = 19 398 656 and
  77 594 624, both asking 37.0 GiB of a 31.7 GiB card) and on 16 GB those plus Wan2.1-T2V's
  VAE projection. No run of that class forms them because Prism tiles the decode first; they
  are in the census because op-level tiling does not cover those flattened projections, which
  is the named lead, not a certification debt.
* **one unreachable key per class** — the standing D-CENSUS-HOLDS-KEYS debt.

So the directory now serves every key the catalogue's censuses demand on both V100 classes,
under torch 2.14 / Triton 3.8, with each entry proven on the class it serves. Stage three
(verification at zero miss with artefacts judged) is what this unblocks — and it waits on
nothing else here.

**Also worth your reader**: the checksum pass's `ERROR 37` was never thirty-seven bad
containers. Every one was a 503 or a Range read that failed three times — the store refusing a
RATE while answering single probes fine, the exact offset that failed on Allegro reading in
2.6 s an hour later. Three attempts over fifteen seconds is not patience against a limiter,
and the distinction it destroyed (UNREAD against MISMATCHED) is the whole point of the pass.
The reader now waits 5/15/45/120/120 s and puts half a second between chunks of an object once
refused (f46759e5). If your own hub reads ever report errors in bulk, read the reason before
the count.

## 2026-09-22 11:05 — why the store refuses writes: it is not a rate, it is seconds per write (for Hocine)

The refusal has been called "the store's 503 SlowDownWrite" and left there. Measured with the
tool's own probe path (a registry upload slot, then the presigned PUT, slot deleted after):

| operation | result |
|---|---|
| PUT 5 bytes | **200 in 7.69 s**, then 9.93 s, then 2.26 s, and once the registry itself timed out |
| PUT 1 024 bytes | **200 in 18.69 s** |
| PUT 65 536 bytes | **400 IncompleteBody after 30.02 s** — the server did not receive the bytes it was promised |
| PUT 1 048 576 bytes | **503 SlowDownWrite in 0.01 s**, `Retry-After: 60` |
| GET 65 536 bytes (twice, two offsets) | **206 in 0.009 s** |

The store READS sixty-four kilobytes in nine milliseconds and takes two to ten seconds to
WRITE five bytes. That is not bandwidth and it is not a request rate: five bytes have no rate.
It is a per-write-operation stall, and the instant 503 on anything larger is MinIO shedding
load it already cannot carry — which is why the health endpoints all answer 200 (they report
liveness, not the write path) and why the checksum pass could read at all.

Read against it: MinIO returns `SlowDownWrite` when an erasure set cannot reach write quorum,
commonly from stale or sick drive state, and the project's own tool already recorded exactly
that on 2026-09-07 — "the cluster health answered 200 with a write quorum of 1 through every
refusal". The asymmetry fits: an erasure WRITE must reach quorum across the set, so one drive
stalling every write costs seconds; a READ is served from whichever drives answer first, so it
stays at nine milliseconds.

**What I could not do**: there is no MinIO credential on this rack (`.env` holds none), the
metrics endpoints answer 403, and `ssh 10.0.0.36` is refused (publickey, password). So the
drive state itself cannot be read from here.

**The one action, for Hocine, on 10.0.0.36** — read the drive state and say which drive is
stalling:

```
mc admin info <alias>                         # per-drive online/offline and latency
journalctl -u minio --since -24h | grep -iE "drive|disk|quorum|heal|timeout|slow"
dmesg -T | grep -iE "I/O error|ata[0-9]|nvme|reset|timeout"
smartctl -a /dev/<the backing disk>
```

If one drive is stalling or offline, the write path is waiting on it and publication stays
blocked until it is replaced or dropped from the set; a service restart clears the stale-disk
form of this and is what MinIO's maintainers suggest first, with an upgrade as the permanent
fix (minio/minio#17875, fixed by PR #17085). **Nothing here should be restarted without you**:
the store holds the hub.

Meanwhile the checksum pass reads on, and publication is deferred by the tool itself rather
than retried — a 1 MB write is refused in one hundredth of a second, so retrying is not
patience, it is noise.
