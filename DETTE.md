# DETTE — deferred-debt registry (video family validation)

This file is the single registry of **deferred debt**: work that is flagged
during the video-family validation but is **never fixed in the current pass**.
It is touched ONLY at the very end, once all video models are closed 4/4 at the
single-GPU **13f-class** config.

Rule: when the maintainer or the supervisor says *"that's debt, for the end"*,
the item is recorded here (ID, short title, root cause / what to do, scope,
status `DEFERRED`) and left untouched until the final pass. Closing a model
4/4 at 13f-class is *correctness for now*; long clips / native resolution /
multi-GPU capacity are *capacity for the end*.

Entry format: `ID · title · root cause / fix · scope · status`.

---

## DEFERRED items

### D1 · P-PRISM-MULTIGPU (task #23) — multi-device op-input co-location

- **Root cause / fix.** The multi-device placement strategies do not guarantee
  that **all tensor inputs of an op live on that op's compute device** before
  dispatch → cross-device crash (reproduced: `aten.cat::6` with inputs split
  across `cuda:0` AND `cuda:1`, via `--hardware v100-16g-x2-01`). The fix is
  **general, never a patch**: the executor must co-locate every tensor input on
  the op's compute device before dispatch, **data-driven from the placement
  plan**, for **each** strategy — `lazy_sequential`, `block_scatter`,
  `pipeline_parallel`, `component_placement`. Then re-validate across **all**
  `config/hardware/*.yml` profiles: single-GPU, homogeneous multi-GPU,
  heterogeneous multi-GPU, variable VRAM.
- **Scope.** All multi-GPU placement; every model that does not fit on one GPU.
- **Status.** `DEFERRED` — final pass, as the general Prism capability (with D2).

### D2 · VAE-5D long-clip / native-resolution OOM single-GPU (task #5)

- **Root cause / fix.** A 5D video VAE at long clip length / native resolution
  exceeds one GPU. CogVideoX-5b at native **49 frames** needs ~35 GB > 32 GB
  (the input + accumulator of a single conv3d at full extent) even with the
  `_conv3d_via_conv2d` eager-free fix that closed the 13f case (commit
  `302a70e`). Requires **temporal 5D VAE tiling** OR **multi-GPU placement**.
  Models close 4/4 at the single-GPU **13f-class** config; long clips / native
  resolution are deferred.
- **Scope.** CogVideoX-5b-I2V (native 49f), Sana-4Kpx VAE, Wan full-size f81,
  any large-VAE video model at native length/resolution. NOTE: CogVideoX-2b VAE
  **seams at 13f are correctness to close now** (not debt) — only its long-clip
  tiling is deferred here.
- **Status.** `DEFERRED` — final pass, together with D1.

---

**D1 + D2 are handled TOGETHER, at the very end, as one general Prism capability:
hardware-agnostic multi-GPU placement + 5D VAE temporal tiling, validated across
the `config/hardware/*.yml` profiles. Not before the family is 4/4 at 13f-class.**
