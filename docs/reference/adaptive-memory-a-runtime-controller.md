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
