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
