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
- **EMPIRICAL REFRAME (2026-07-01) — the D1 premise for Wan-I2V-14B is REFUTED by
  test.** Ran Wan-I2V-14B **batch=2 (cfg=5.0) triton_sequential** on the real 4-GPU
  box. It RUNS (exit 0, 550s, finite output, NBX_TRITON_TRACE_NAN clean): Prism
  chose `pipeline_parallel` **component-level** placement (transformer+vae+encoders
  → cuda:2, text_encoder → cuda:3) — the 31.25 GB transformer FITS one 32 GB card
  at batch=2, so the cross-device move is a COMPONENT-boundary handoff (handled by
  the driver's 2026-06-08 co-location block), NOT an intra-component split. The
  earlier "triton_seq cannot run batch=2" was INFERRED (the verdict row had
  dashes), never observed. **The real blocker is numerical, not placement:** a
  step-0 velocity drift-gate vs the sequential oracle (NBX_DUMP_DIT, seed 42,
  cfg=5.0) shows BOTH triton modes diverge ~identically — triton_seq corr 0.9753 /
  relL2 0.227, triton corr 0.9696 / relL2 0.254 (std 1.21 vs oracle 1.1776). Same
  divergence in both triton modes = a SHARED triton transformer numerical bug at
  batch=2 (TritonDtypeEngine / a kernel), unrelated to multi-GPU. So D1 is NOT the
  Wan-I2V-14B unblocker; the numerical bug is.
- **The intra-component co-location gaps ARE real in code but UNTRIGGERED:** the
  driver picks the FIRST weight (not largest) for `_target_dev`
  (graph_executor.py ~2016) and never updates `dispatcher.device_idx`
  (triton/sequential.py device attr + lse/cat-empty allocs pinned to a fixed
  device). These bite ONLY when a single component's weights span >1 GPU. Six
  proxy attempts (TinyLlama ×4 budgets, deepseek ×2) proved Prism strongly PREFERS
  single-GPU / component-placement / zero3-cpu and AVOIDS intra-component GPU
  splits — none triggered the gap. The gaps are latent; a fix is currently
  UNVALIDATABLE (no in-family model exercises them at single-GPU-per-component
  scale). Candidate genuine triggers: Wan2.2-I2V-A14B (solver shows intra-split of
  the 14B experts — but the dual-denoiser expert-lifecycle may make the RUNTIME
  component-level; needs a runtime run to confirm) and Qwen3-30B (57 GB > 32 GB,
  non-family). Do NOT implement the co-location fix until a real run triggers it.
- **Also seen (separate D1 facet, zero3/CPU-offload, NOT GPU↔GPU):** triton
  (compiled) crashes `custom.rope_fused::0` "Pointer argument cannot be accessed
  from Triton (cpu tensor?)" when Prism falls to `cpu_execution` (a weight stays
  CPU-resident but the fused-rope kernel is launched on it). triton_sequential
  handles cpu_execution fine. Named sub-item; off the GPU↔GPU critical path.
- **Sequential-CFG combine precision (low priority, SYMMETRIC — not R30):** the
  `_execute_sequential_cfg` combine skips the fp32 upcast that the BATCHED path
  does — in BOTH engines (`core/cfg/engine.py:443` and `triton/cfg/engine.py:458`
  both do `uncond + scale*(cond-uncond)` without a `.float()`). Because it is
  symmetric, there is NO cross-engine divergence (both modes match each other);
  it is only a shared "should upcast for stability" note for TP-strategy models
  (sequential CFG only fires under `tp_components`), which are rare. The BATCHED
  path (used by Wan and every pipeline_parallel model) DOES upcast to fp32 in both
  engines. Fix both sequential paths to `.float()` before the combine when the TP
  chantier lands.
- **Status.** `DEFERRED` + REFRAMED — the co-location capability is real but not on
  the video-family critical path (refuted for Wan-I2V-14B + Allegro; open for
  Wan2.2 pending a runtime check). Kept for the general Prism capability / Qwen3-class
  models. The Wan-I2V-14B triton 4/4 blocker moved to the numerical chantier below.

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
