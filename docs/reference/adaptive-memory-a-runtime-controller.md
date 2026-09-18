# Adaptive memory: why a plan chosen once is not enough

**Status**: design. Implementation follows the Prism residue work and the branch merges.

## The case that motivates it, with a 15-second reproducer

```
neurobrix run --model real-esrgan-x8 --input-image apple_1024.png --triton
  RuntimeError: Failed at aten.convolution::349 (aten::convolution):
  GPU malloc failed (error 2) for 8589934592 bytes
  [device cuda:0 live_tracked=8242MB pool_cached=0MB driver_free=7598MB]
```

A **single 8.59 GB allocation** on a 16 GB card, with **8.24 GB already live** and **7.60 GB free at
the driver**. The engine raises. Nothing re-plans, nothing reshapes the work, and the run is over
15 seconds after it started.

This is not an exotic configuration: it is one upscaler, one 1024x1024 image, on the rack's ordinary
card. It reproduces every time, which makes it the right case to design against.

## What the existing cascade already does

`single_gpu -> component_placement -> pipeline_parallel -> block_scatter -> weight_sharding ->
lazy_sequential -> zero3 -> op-level tiling (R31)`

The last rung is exactly the remedy this case needs — split the convolution so no single allocation
is 8.59 GB. **The cascade is not missing a rung. It is missing a moment.** Every rung is selected
once, before the first byte is allocated, from an estimate; and when the estimate turns out wrong
the cascade is never re-entered.

**CORRECTION, 2026-09-18 — read from `solver.py` rather than from the paragraph above.**
That last arrow is not in the cascade. The strategy list the solver actually tries ends
`... -> zero3 -> layer_streaming -> cpu_execution -> cpu_streaming`, and **op-level tiling is
not an entry in it at all**: `plan.runtime_op_tiling = self._detect_op_level_tiling_pairs(...)`
runs at `solver.py:1082`, AFTER `chosen_strategy` is already settled, as a decoration of
whatever the cascade picked.

So *"the cascade is not missing a rung"* is **wrong in its first half**. It is missing the
rung. And that single fact explains both machines at once, which is how it was found:

* **On Apple** the cascade exhausts and `_fail_error` refuses. Tiling is never consulted as an
  alternative to refusing, because it is not one of the things tried.
* **On CUDA** the cascade does not refuse: `cpu_streaming` accepts the model and wins by score,
  so the component is placed on the host. The tiling decoration then runs against a CPU
  placement and finds nothing to do.

Measured the same day, on both machines: `real-esrgan-x8` at 1024x1024 plans **17237 MB** —
the same figure to the megabyte — and both print **`tiling none planned`** for 17 GB of
activations against 32 MB of weights.

The missing MOMENT is real too, and addition 3 still addresses it. But addition 2's first act
is structural: put the rung IN the cascade, between `zero3` and `layer_streaming`, so it is
tried before the host rungs and before the refusal. A controller that only re-enters after a
failed allocation is dead code on Apple, where nothing is allocated, and dead code on CUDA,
where the host rung has already accepted. **Both doors are needed, and neither is the one this
document originally named.**

## Which of the three answers is right here

**A plan that never promises what the card cannot hold.** Necessary, and insufficient alone.
Necessary because the plan is what the tiling decision consumes: measured on this rack, the
activation estimate runs **1.41x to 2.27x short** against the drained watermark, so op-level tiling
is asked to fire on a number that does not describe the run. Insufficient because even a perfectly
truthful plan only tells you the op needs 8.59 GB — it does not make the card larger. A truthful
plan turns a crash into a refusal, which is better, and still leaves the user without their image.

**A re-plan on the failed allocation.** Necessary, and insufficient alone — and weaker here than it
looks. The allocator already drains its deferred queues and retries once before raising (the Sana
4Kpx work). It cannot help in this case: the 8.24 GB that is live is genuinely live, not cached, so
there is nothing to reclaim, and a retry of the same 8.59 GB request fails identically. A controller
that only retries is a controller that fails twice.

**Both, plus the action neither names.** The answer this case actually demands is a third verb:
**reshape the work**. Re-entering the cascade at the op-level-tiling rung, with the ACTUAL free
figure the allocator just reported, turns one 8.59 GB allocation into bands that fit 7.60 GB. That
is the rung that already exists; what is missing is the ability to reach it after the plan has been
chosen.

So: **both, and the re-plan must be allowed to change the shape of the work and not merely its
placement.**

## The additions, each mapped onto the cascade

1. **A truthful estimate** — the plan is never below what execution holds. Prism work, in flight;
   the residue is 1.41x-2.27x and its source is named (the fp32 islands the DtypeEngine forces on
   Volta are sized at 2 bytes and executed at 4).
2. **A plan that can say "not whole"** — an op whose single allocation exceeds the device's free
   figure is marked at plan time, and the op-level tiling rung is entered for it deliberately rather
   than as a fallback. This is a new *reason* to enter an existing rung, not a new rung.
3. **A runtime controller at the allocation failure** — the allocator's refusal carries what it
   already prints: requested bytes, live, cached, driver-free. That is enough to re-enter the
   cascade at the tiling rung for THAT op, with real numbers rather than an estimate. One re-entry,
   not a loop; if the reshaped work still does not fit, the engine refuses by name.
4. **The refusal names what would have made it fit** — the smallest input, the tile count, or the
   card size. A refusal that only says "out of memory" costs the reader the whole diagnosis.

## What this is not

It is not a caching-allocator change: the pool is off in the reproducer and the failure is
identical. It is not zero3 or offload: the 8.24 GB live set is what the op needs to read. And it is
not a threshold to tune — every number above is measured from the run that failed, and the design
stands or falls on whether the controller can reach the tiling rung, not on where a constant sits.

## Where the controller can and cannot be put (measured 2026-09-18)

Addition 3 says "re-enter the cascade at the op-level tiling rung". It is worth writing
down why the cheaper seam beside it does **not** work, because it looks like it should.

`conv2d_wrapper` already carries a kernel-level band-streaming lever
(`_NBX_CONV2D_BAND_BYTES`, 4 GiB, P-SANA-4KPX Étape 1). The reproducer's failing allocation
is **8 589 934 592 bytes** — the conv's own output, `(1, 64, 8192, 8192)` in fp16 — which is
**above** that threshold, so the band path is entered. And it still dies, for the reason its
own docstring gives:

> The full output tensor is allocated up front (downstream consumers expect it whole); per
> band we slice the input H, recurse into conv2d_wrapper for the band ... and write the band
> slice back into the full output.

**Band streaming reduces the transient working set. It does not reduce the output.** The
allocation that fails is the output, before any band runs, so no threshold inside the wrapper
can help: at 4 GiB it bands and dies, at 16 GiB it does not band and dies identically.

That is what "reshape the work" means in addition 3, stated exactly: **the output extent
itself has to shrink**, and that can only be decided where the DOWNSTREAM CONSUMERS are
known — because the reason the wrapper hands back a whole tensor is that its callers expect
one. The op-level rung is the only place that holds both sides.

So the controller's shape is fixed by this:

* it may not live in a wrapper, because a wrapper cannot change its own output contract;
* it enters at the op-failure seam, where the op_uid and the allocator's real figures are
  both in hand (`DeviceOOMError` now carries requested / live / pool_cached / driver_free /
  shortfall, `bd96c7b1`);
* it re-enters at the op-level tiling rung for THAT op_uid, which already owns the
  downstream-aware machinery (`OpLevelTilingPlan`, `register_op_uid_interceptors`, mirrored
  into both sequences);
* one re-entry, then a refusal that names what would have fit — which is already there
  (`kernels/oom_advice.py`, `66ed17af`) and already says, on this exact failure:
  `what would have fit: short by 594 MB; it would fit in 2 bands of about 4,096 MB`.

The remaining work is the re-entry itself: building a one-op `OpLevelTilingPlan` from the
failure and registering its interceptor mid-run, which the machinery currently expects to be
done at plan time. That is the next step, and it is named rather than assumed to be small.
