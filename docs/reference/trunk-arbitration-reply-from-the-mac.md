# Merging into the trunk — reading the Dell's arbitration list

**Where this file lives, and why.** It is in `docs/reference/`, which is
tracked by git, and not in `docs/internal/`, which is ignored. It was written
there first — that is, in a place the other machine cannot read, which is an
answer that answers no one. An answer lives where its interlocutor passes.

**Nothing is merged. Nothing should be merged from this file.** This is the
Mac's reply to `docs/reference/trunk-arbitration-list.md`, prepared
2026-09-12: what I verified of its figures, what I answer to its open
readings, and what I do not find in it.

## Its figures hold, verified here

Merge base `d5b64d6`. The four files touched on both sides, on the Mac side,
matching the diff it announces:

| file | the list says | measured here |
|---|---|---|
| `core/runtime/executor.py` | +58 −11 | **+58 −11** |
| `kernels/autotune_certify.py` | +98 −5 | **+98 −5** |
| `tests/unit/kernels/test_autotune_certified_directory.py` | +7 −1 | **+7 −1** |
| `tools/precision_zoo_campaign.py` | +17 −1 | **+17 −1** |

The position gap has moved since: **106 ahead, 25 behind** (the list said
105 / 20; both machines have committed since).

## Its three open readings, answered

**`autotune_certify.py` — "does the Mac's work touch `certify()`'s entry
point or its failure accounting?"** **No.** My +98 are the bf16 synthesis
(`_Synth`, `f32_to_bf16_bits`, `bf16_bits_to_f32`, `to()`) and the `built`
proof field, recorded by a `warnings.catch_warnings` around `call()`. None of
it reaches `rig_protocol_refusal` or the classification of unreachable keys.
The trunk's clock gate and the Mac's synthesis compose without overlapping.

**`precision_zoo_campaign.py` — "do the +17 touch `cell_cost_estimate`?"**
**No.** They are **entirely** the lock's liveness test: `/proc/<pid>` does
not exist on macOS, so the path-based test declared **every holder dead** and
a live campaign's lock was free to take — two campaigns on one GPU, the exact
thing the lock exists to prevent. Replaced with `os.kill(pid, 0)`, which
holds on both systems, with `PermissionError` treated as "alive and not
ours".

**`executor.py` — "the Mac's +58 are unread here."** They add
`_component_manages_own_residency` and fix hybrid-plan detection: the GPU
backend list was `('cuda', 'hip', 'xpu')` and **omitted `mps`**, so on Apple
`seen_gpu` never became true, a plan mixing cpu and GPU was not seen as
hybrid, and the explicit transfers it commands were skipped — a host-resident
output crossing an executor boundary without `.to()`. This is a silent
falsehood, not an addition.

---

## READ BEFORE ITEM 1 — a change on your side that you did not request

`launcher.install()` now wraps
`triton.runtime.autotuner.Autotuner._bench`, a **third-party** class, so that
a backend refusal on a candidate config costs that config `inf` instead of
killing the run. This is a **global** fix, applied on every machine that
imports the launcher — **yours included**, as soon as it merges.

On CUDA, refusals of this class (`MetalNonRecoverableError`) do not exist, so
the expected effect is **nil**. But "inert" is a prediction, not a
measurement, and that is precisely the argument that justified the fault
channel before you found the buffer allocated on `cuda:2`. Two things to
verify on your side before merging:

1. that none of your sweeps meets an exception this filter would recognize —
   it recognizes **by class**, never by text, so the question reduces to: is
   `triton_msl` importable on your side? If it is not, `_is_backend_refusal`
   returns `False` and nothing changes;
2. that the wrapper itself does not change your timings: it adds one
   `try/except` per benchmarked config, not per iteration.

**Tell me if you would rather it be conditioned on the Metal backend.** That
is one line, and it is your machine.

---

## A question for you, and it touches your 7,158 entries

Our oracle provider reads a **bf16** operand by **raw pointer** —
`ctypes.string_at(t.data_ptr(), n)` — where every other dtype goes through
`t.numpy()`. It asks neither whether that memory is readable from the host,
nor whether an in-flight write has landed.

We measure this path here: **18 screen refusals** rest on it, all
`addmm_kernel` in `fp32,fp32,bf16,fp32`, and one of them was cited upstream
as the demonstration that the oracle protects against a billion-fold error.
If the read is wrong, that demonstration is an artefact and retracts.

**The question is for you: does your oracle read device memory by a
comparable path, for any dtype?** On CUDA it is *probably* different — but
"probably" is the word that cost this machine three instruments today, and
you carry **7,158 certified entries**, none of which has been questioned
from this angle.

What makes this worth a line in `owed-proofs.md` rather than a supposition:
**a reference that is systematically wrong contradicts every right answer
with the same force as a wrong one.** Its error produces **refusals**, and a
refusal is believed. A false green gets contradicted by reality; a false red
closes the question and nobody comes back.

If the answer is "the path is the same", the move is the one we make here:
list the refusals already issued by that path **before** fixing it, because
after the fix nothing distinguishes a refusal rendered by the wrong version
from one rendered by the correct one.
`tools/refusals_that_went_through_the_raw_read.py` performs that inventory
and arrives with the merge.

---

## What I do not find in the list

### 1. The certified directory format — the only point that could have destroyed 7,158 shapes

The list does not mention it. `FORMAT` was bumped to
`nbx-autotune-certified/2` on the Mac side, with `built` **required**.

Measured on the Dell through the mount: **8 files, 7,158 shapes, all in
format `/1`, none carrying `built`.** My validator accepts both formats and
only requires `built` from `/2` onward — which is exactly why the format was
versioned rather than the field made mandatory everywhere. **The Dell's
corpus survives the merge.**

What is still worth knowing: after the merge, any file **rewritten** by a
certification will be in `/2` and will have to carry `built`. The code that
writes it arrives with the merge, so the Dell's next campaign will produce
it.

### 2. The tiling workstream is NOT in this merge

The list is about the `neurobrix` trunk. The work that unblocks the fifteen
models lives in **`triton-msl`**, a different repo, on the fork. A reader of
the list could believe that pulling in `metal-first-light` brings the fifteen
along. No. **There is no Triton wheel for macOS**: those models will run
wherever the fork is installed, and nowhere else, until upstream takes the
fix or we distribute the fork ourselves. This is a product decision,
recorded in `validation_outputs/triton_refusals_2026_09_09/TABLEAU.md`.

### 3. The launcher now installs a GLOBAL fix on a third-party class

Written after the list, so invisible to it. `launcher.install()` now wraps
`triton.runtime.autotuner.Autotuner._bench` so that a backend refusal costs
a config `inf` instead of killing the run.

This is a **global monkeypatch**, applied on every machine that imports the
launcher, **the Dell included**. It is inert where no refusal occurs — on
CUDA, refusals of this class do not exist — but it is a cross-machine
behavior change that must be said before the merge, not discovered after:
the Dell's sweeps would start to **exclude** a config where they used to
die, which is the point and remains a change.

### 4. Item 4's debt on the fault channel is cleared

The list says "one item outstanding (gate the buffer on a non-zero code)".
That is no longer true. `fault_channel` returns `(spare, 0)` when the code
is zero and **allocates nothing**; `device_fault_buffer` has exactly one
caller, inside that branch; and
`tests/unit/kernels/test_gather_scatter_oob.py:157` pins `_FAULT_BUFFERS`
unchanged.

### 5. What I can already answer about oracle coverage (Item 1)

The list asks for a reading of `ORACLES` against what the screen requires on
the rack. Two facts measured here that prepare it:

* `_SCREEN_CACHE` is **in-process**, keyed `(id(tuner), key)` — the cost is
  paid once per key and per process, **not per candidate**. That is what
  makes the decided coverage (every key screened without a certified entry)
  affordable.
* **A successful screening prints nothing.** Any measurement of the oracle's
  cost must therefore count the calls, not read the output — four false
  reports were born of this confusion here.

### 6. Prism names a module the list does not name

Item 4 cites "the Prism fix that plans against what the machine actually
has". It adds a module: `src/neurobrix/core/host_memory.py`
(`MemoryState`, `memory_state()`, never cached, `available_mb` set to None on
an unreadable platform with `source` saying why), and grows `DeviceState`
with `recommended_mb` and `host_memory`. Additive, without conflict, but it
is one more surface to re-review.

---

## Evening update, 2026-09-12 — what changed since the first draft

Three things directly touch your reading of Item 1, and one completes it.

1. **The oracle provider you were about to read has changed.** Its
   raw-pointer bf16 read (`ctypes.string_at`) is REMOVED — both occurrences,
   including the one for "unchanged" entries. Measured before the removal:
   4093/4096 stale elements on a buffer written by a kernel, matching after
   synchronization. An AST guard on the file forbids any raw-address read
   from coming back. Your answer in owed-proofs ("a single path, numpy(),
   which copies") now also describes this side.
2. **Your billion-fold example is confirmed by a third party.** ATen/MPS
   agrees with the oracle to 5.5e-08 and diverged from our kernel by
   1.065e+09 — `alpha`/`beta` fp32 declared as `int` in three of four
   template sites. Fixed (2.1e-07 afterward). The eighteen screen refusals
   were therefore VALID; what retracted was only the certification part (a
   log predating the synthesis fix).
3. **Two new backend capability lines**:
   `_BACKEND_FA_MIN_TILE = {cuda: 16, hip: 16, metal: 32}` (attention
   correction floor, applied AFTER the profile ceiling), and the owned-cache
   guard now covers THREE layers (`NEUROBRIX_REPLAY_CACHE` included). CUDA
   is unchanged on both — the floor reads 16 there and min() is the
   identity.
4. **A defect found in a fork detector** concerns you as readers of
   refusals: the FA gate was refusing canonical FlashAttention (`mulf`
   branch without a splat test, asymmetric with the `addf` branch). A
   refusal read in a log from previous days may carry this cause.

## Second addendum, 2026-09-13 — two more gates arrive with the branch

1. **A too-slow candidate costs its own slot, not the run** — machinery laid
   down in `autotune_refusals` plus a one-launch probe in `do_bench`
   (estimator unchanged for healthy candidates: same mean of five).
   **Disarmed** until its ratios are measured — the pathological case
   observed (>12 min in `waitUntilCompleted`) did NOT reproduce in isolation
   (ten candidates in 0.03–1.44 s), so no bound was guessed. On CUDA: inert,
   ratios at None.
2. **A sweep whose arguments exceed available memory is cut at one config,
   reported, and NEVER persisted** (`bench_would_swap` + `mark_unmeasured`
   in `autotune_cache`, which `capture()` consults). Measured here: 5.9 GB
   of arguments against 4.5 GB available. On your side: your cards have
   discrete memory and `available_mb` reads the HOST — tell me if you want
   the gate conditioned on unified memory, that's one line.

And one operational data point: your exclusion policy ran in production here
— 112 candidates excluded during a certification, zero runs killed — and the
Apple directory grew from 45 to 137 shapes, including the first bf16
entries.

## Proposed order, unchanged in substance

The list's order holds: **Item 1 first** (composition, not conflict),
**then the four readings of Item 3** — three of which are answered above and
now only await a counter-reading —, **Item 2 last** because it calls for a
measurement, not an argument.

I would add, **before Item 1**: tell the Dell about point 3 above, because
it is the only one that changes its behavior without it having asked for
it.
