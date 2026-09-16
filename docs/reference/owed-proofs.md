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

## 7 — the 1 168 entries the Triton 3.8 re-proof will never reach, 2026-09-16

* **owed by** the Dell (this rack) · **when** after the 3.8 pass completes, because it is the
  pass's completion that makes this visible rather than theoretical.
* **what was measured, with a control.** 1 168 entries of the volta directory carry a PRIMARY
  proof that names this rack's hostname and **no device memory size** — the legacy unknown-card
  class, which serves no card until re-proven (register 56). They are spread over six files, 725
  of them in `matmul_kernel.fp32.json`, the file the pass is 72 % through.

  They are not draining. Two readings six minutes apart: the 16 GB class went 6 159 → 6 231 done
  while the unknown bucket stayed at exactly 1 168. And the control that makes it an attribution
  rather than a coincidence: all 725 of the matmul ones are PRESENT in the 32 GB side tree, and
  **0** are re-proven there, while **1 374 of 3 920** known-card keys in that same tree are. At
  that rate roughly 254 of them would be done if they were reachable.
* **what it means, said plainly.** When the pass finishes, "the directory is re-proven under
  3.8.0" will be true of every key the pass can reach and false of 1 168 entries that will still
  carry a 3.6.0 proof serving no card. A completion notice that does not say so reads as more
  than it is.
* **the proof this rack returns.** Either the shapes are re-entered into the pass (they came from
  requests the zoo no longer makes, which is why `certify --reprove-generator` never visits them)
  and the 1 168 drain to 0; or they are removed as records of a machine-state the directory can
  no longer attribute, with the removal counted and said. Not both, and not silence.
  `tools/reproof_coverage.py` is the instrument: its `?` row is this number.
