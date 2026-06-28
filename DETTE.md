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
- **Concrete sub-gap (added 2026-06-27, Wan-I2V-14B batch=2):**
  `triton/sequential.py` is **single-device only** — it takes one `device_idx`
  and routes every allocation to `cuda:{device_idx}`, with **none** of the
  cross-device machinery that `triton/sequence.py` already has
  (`compute_op_devices`, `_run_multi_device`, `device_transfer.transfer_tensor`).
  Any model whose component does not fit one GPU therefore cannot run in
  `triton_sequential`. First blocker: **Wan-I2V-14B at CFG batch=2** (31.25 GB
  transformer > 32 GB at batch=2 → needs multi-GPU placement; the other 3 modes
  run, triton_seq cannot). Fix = port the `sequence.py` multi-device path into
  `sequential.py` (R30 mirror), as part of this general Prism multi-GPU capability.
  NOTE: `triton` (compiled) is NOT blocked — it runs multi-GPU via
  `pipeline_parallel` (proven on Wan-I2V-14B batch=2, 2026-06-27).
- **Also under this debt (added 2026-06-27): Wan2.2-I2V-A14B** — dual-denoiser
  (2× 14B = 28B total). The compiled-mode close is single-GPU-feasible via expert
  lifecycle (one expert resident per boundary stage); the `triton`/`triton_seq`
  axes need the same multi-GPU/multi-device capability as Wan-I2V-14B → deferred
  here. (Its compiled close is itself OPEN on the i2v vae_encoder encode-pass — a
  chantier residual, not debt; see validation_outputs/video/Wan2.2-I2V-A14B/verdict.md.)
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
- **Also under this debt (added 2026-06-28): Allegro (T2V) — native-config locked.**
  Allegro is **out-of-distribution at 13f** (both NeuroBrix and the diffusers vendor
  degenerate at reduced frame counts; the model is native-720×1280×88-frame only).
  Its VAE decode already uses **~28 GB at latent T=4** and exceeds one 32 GB GPU at
  the native latent T=22. Therefore Allegro **cannot close at 13f-class** — its
  coherent frame needs the native 88-frame regime = this debt. The **#30 odd-H
  scanline** is NOT a symbolic-shape bug (proven: shapes resolve correctly at odd-H,
  no frozen literals, RoPE tables resize symbolically — see
  `validation_outputs/video/Allegro/verdict.md`); it is a **frame-count-dependent
  numerical artifact that only reproduces in this native regime**, so both its
  op-localization and its coherent verdict are deferred here with D2. Allegro-TI2V
  (same VAE/family) inherits the same native-regime constraint.
- **Status.** `DEFERRED` — final pass, together with D1.

---

**D1 + D2 are handled TOGETHER, at the very end, as one general Prism capability:
hardware-agnostic multi-GPU placement + 5D VAE temporal tiling, validated across
the `config/hardware/*.yml` profiles. Not before the family is 4/4 at 13f-class.**
