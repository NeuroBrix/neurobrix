# The Triton engine's static memory plan (S4)

Status: design, 2026-10-27 — the Dell writes the engine side and proves it on CUDA, the Mac proves it on Metal.

## Why

The Triton engine holds more than the activation profile Prism prices: `PRISM_DEFAULTS["triton_arena_activation_factor"]`
(1.3, measured 1.35 on mochi-1-preview's VAE, 1.14-1.18 on Metal) multiplies every whole component's activations under the
Triton modes. A coefficient is a defect (engine rule 7), and the gap is ours, not the hardware's:

* a dead tensor waits in the hot loop's deferred queue until the queue reaches its cliff
  (`memory.deferred_drain_cliff_bytes`), then a device sync frees the batch — the cliff is priced beside every op
  (`PrismSolver._deferred_drain_mb`);
* an output slot overwritten by the next run of the same op parks its previous tensor in the same queue;
* the allocator's pool keeps freed blocks by EXACT size (`DeviceAllocator._pool_free[dev][nbytes]`): a freed 3 GB block
  never serves a 2 GB request; blocks are neither split nor merged;
* the measured result: the 8 GB rung on mochi-1-preview — the native run holds 8.4 GB, the Triton run 22.6 GB
  (2026-10-27, `proof8g/`); transformer segments at 10.1-10.7 GB live + 4.1-4.9 GB parked against 8.2 GB.

We know the whole graph before running it. The liveness of every tensor is computed at compile time
(`TritonSequence._compute_liveness`, `kill_slots`); the sizes are known once the symbols bind. A static plan assigns
every tensor an offset in one workspace so that two tensors share bytes only when their lifetimes do not overlap —
the "greedy by size" offset assignment (Pisarchyk & Lee, *Efficient Memory Management for Deep Neural Net Inference*,
2020) lands at or near the lower bound (the largest sum of simultaneously live sizes) on most networks.

## What it is

One plan per (sequence, shape bucket) — the bucket the replay already keys by (`triton/replay.py`):

1. **Record** (the first run in a bucket): the run executes as today, and `DeviceAllocator` records, per op index, the
   sequence of allocations the op makes (size, order) and which of them is still referenced by an arena slot after the
   op (an OUTPUT) or not (a TEMPORARY, its lifetime the op itself). An output's lifetime ends at the op whose
   `kill_slots` names its slot (or the run's end for the declared outputs and protected slots).
2. **Plan**: intervals [alloc op, last op] with sizes; offsets by greedy-by-size (largest first, each placed at the
   lowest offset that overlaps no live interval already placed); the workspace size is the plan's peak.
3. **Run under the plan** (every later run in the bucket): the sequence takes ONE workspace from the allocator; before
   op `i` runs, the allocator is told op `i`'s planned allocations; the op's k-th allocation of the planned size is
   answered with `workspace + offset`, and its free is a no-op (the bytes are reused by a later tensor, stream-ordered on
   the one stream: no sync, no deferred queue). An allocation the plan does not hold (another size, another count: the
   bucket's run diverged) is refused BY NAME with the op and both sizes, and the bucket is re-recorded — never served
   silently from the pool beside the plan.
4. **Price**: Prism prices a Triton component at the plan's peak — computed by the same greedy assignment over the
   activation profile's intervals (`ActivationProfiler` already walks the same liveness, `core/runtime/liveness.py`),
   the temporaries per op from the wrappers' declared workspaces (`core/prism/op_workspace.py`) — and the arena factor
   and the deferred-drain price are removed for planned runs.

## What it changes

* `kernels/nbx_tensor.py` `DeviceAllocator`: a recording mode and a planned mode (per device), backend-agnostic: the
  same code on CUDA, HIP and Metal (the Mac proves Metal).
* `triton/sequence.py`: the plan per bucket, the workspace's lifetime (released with the sequence's run tensors), the
  hot loops tell the allocator the op index; the deferred queue is bypassed for planned pointers.
* `triton/replay.py`: a recorded launch plan bakes pointers — under a static plan they are the workspace's, fixed
  across runs, so the replay needs no re-record when the pool would have moved them.
* `core/prism`: the greedy assignment shared by the plan and the price (one function), the factor removed.

## Proof

* Unit: the assignment's peak equals the brute-force lower bound on small interval sets; two intervals that overlap
  never share a byte (seen red with an off-by-one lifetime).
* Engine (entry point, both Triton modes): on the cells whose peaks are in the registers (mochi 16g/8 GB rung, Sana 4K,
  Wan2.2, the LLMs), Triton peak <= native peak on the same cell; outputs byte-identical to the judged ones.
* Prism: the arena factor gone, the plans of the gate's cells either unchanged or moved to a faster plan that now fits.

## Estimate

1-2 days for the engine side and its proof on CUDA, the Mac's Metal proof in parallel.
