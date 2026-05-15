# NeuroBrix v0.2.0 — P-PRISM-NEVER-REFUSE v2 closed (2026-05-15)

162 commits since `v0.1.6` (2026-04-20). Minor version bump per SemVer:
features added are rétrocompatibles, no breaking API changes. Major
acquis closes the P-PRISM-NEVER-REFUSE v2 mandate with 14/16 hardware ×
mode cells validated end-to-end on Sana 1.6B 4Kpx diffusion (4096×4096
output).

## Highlights

- **Doctrine R34 model-agnostic** ratified and audited clean — zero
  active hardcode-by-model-name violations across runtime, kernels,
  strategies, dispatchers.
- **Doctrine R35 "Prism never refuses"** implemented and empirically
  validated — cascade strategy covers every legitimate hardware ×
  model combination down to `cpu_execution` fallback.
- **14/16 cells matrice hardware × mode** validated end-to-end on
  Sana 4Kpx (single 32 GiB, single 16 GiB, 2× 16 GiB; compiled,
  sequential, triton, triton_sequential). The 2 ⏸ cells remain on
  upstream-blocked CPU triton modes (triton-cpu PyPI wheel pending).
- **18× speedup on 32g triton mode**: 1443 s → 79.81 s wall on
  Sana 4Kpx via DC-AE residual chain band-streaming (each of 9 VAE
  chains streams band-by-band, peak transient drops from 6.7 GiB to
  1.6 GiB per chain).

## Hardware coverage

| Config × Mode | compiled | sequential | triton | triton-seq |
|---------------|----------|------------|--------|------------|
| Single GPU 32 GiB   | ✓ 23.9 s | ✓        | ✓ 79.8 s | ✓        |
| Single GPU 16 GiB   | ✓        | ✓        | ✓ 79.4 s | ✓ 84.5 s |
| Multi-GPU 2× 16 GiB | ✓        | ✓        | ✓ 81.8 s | ✓ 84.1 s |
| CPU pure            | ✓        | ✓        | ⏸ S3     | ⏸ S3     |

Wall times measured on Sana 1.6B 4Kpx single-step at 4096×4096 spatial
output. CPU triton cells wait on upstream `triton-cpu` (meta-pytorch
project) shipping PyPI wheels.

## Architecture

### Hybrid CPU+GPU per-component placement
`core/strategies/lazy_sequential.py` + `core/runtime/executor.py` now
correctly route mixed-device plans through `strategy.execute_component`
so a CPU component consuming a GPU producer no longer stalls in
implicit transfer or raises device-mismatch. Unlocked 16g compiled +
sequential on models whose VAE alone exceeds the GPU budget.

### Intra-component multi-device cascade
The `single_gpu` strategy correctly picks `cuda:0` on 2× 16 GiB
hardware when a component fits there post-S5 chain tiling. No solver
extension required — the cascade naturally handles the case.

### DC-AE residual chain tiling — band-streamed (R30 dual-branch)
- Compiled mode: `band_streamed_chain_torch` materialises the residual
  chain band-by-band in PyTorch ATen, writes back to T_base in place.
- Triton mode: `band_streamed_chain_nbx` (143 lines, R33-pure) mirrors
  the algorithm using NBX wrappers + NBXTensor methods only.
- Mode-aware dispatcher in `core/module/tiling_engine.py` selects the
  variant per `graph_executor.mode`.
- Chain wrapper is default-ON on triton modes; `NBX_TRITON_CHAIN_WRAPPER=0`
  reverts to the c9d2581 skip path as emergency rollback.

### Edge-padding + halo-offset wrapper math fix (`_tiled_conv2d_spatial_*`)
Two coupled bugs in the 4 tiled-conv wrappers (`_tiled_conv2d_spatial_torch`,
`_tiled_conv2d_spatial_nbx`, `_fused_upsample_conv2d_torch`,
`_fused_upsample_conv2d_nbx`):
- Edge-band double-padding (max(0, -in_read_start) already provides
  pad_h; the extra `+ pad_h if is_top_band` term over-padded by
  pad_h rows).
- Internal-frontier halo offset on the conv_band read side
  (output slice was offset by halo_top rows on every internal band).
Validated bit-exact via `scripts/microtest_tiled_conv2d_small_scale.py`
sweeping (kh ∈ {1,3,5}, pad_h ∈ {0,1,2}, tile_factor ∈ {1,2,4,8}) at
1024² and 2048² — cos=1.0000 max_abs=0.0000 universally.

### NBXTensor.__getitem__ negative-slice fix — ROOT FIX for chain wrapper
`NBXTensor.__getitem__` slice path used `k.start or 0` which silently
left negative starts untouched (`-2 or 0` evaluates to `-2`, not 0).
For `t[-N:]`, the resulting `narrow(dim, -N, length)` produced an
`_offset = parent_offset - N*stride` → `data_ptr()` pointed BEFORE the
cudaMalloc'd block → Triton's `cuPointerGetAttribute` rejected the
pointer with "Pointer argument (at 0) cannot be accessed from Triton
(cpu tensor?)".

The Sana 4Kpx VAE chain wrapper's `halo_carry[:, :, -top_size:, :]`
triggered this at the 3rd iteration of every 4096² chain. Fix restores
the universal Python/torch slice contract:
```python
start = k.start if k.start is not None else 0
stop = k.stop if k.stop is not None else shape[dim]
if start < 0: start = max(0, shape[dim] + start)
if stop < 0:  stop  = shape[dim] + stop
```
**8-line fix, 4 cells unlocked.** Latent bug across the codebase, now closed.

### Triton-CPU integration stage 1
`triton-cpu` (meta-pytorch project) is not on PyPI today and requires
build-from-source. Stage 1 ships the install gate (raises clean
`ImportError` with actionable message when missing), coverage docs,
and the activation flow. CPU triton + CPU triton-sequential cells
remain ⏸ until upstream ships PyPI wheels — automatic unblock then.

## Doctrine ratified

- **R34 model-agnostic strict**: structural pattern detection, no
  hardcode-by-model-name in runtime / kernels / strategies / dispatchers.
- **R35 Prism never refuses**: cascade fully implemented to
  `cpu_execution` fallback; every legitimate plan finds a strategy.
- **R30 dualité runtime**: the 4 execution modes (compiled, sequential,
  triton, triton_sequential) cover the 14 cells symmetrically;
  `band_streamed_chain_*` mirror restored chain coverage on triton path.
- **R33 zero torch in `triton/`**: preserved; chain wrapper NBX
  variant uses only NBX wrappers + NBXTensor methods, no torch
  import on the execute path.
- **R29 inspectable artefacts**: every cell ✓ has a coherent red
  apple PNG in `docs/verdicts/p_triton_chain_cpu_pointer/` and
  `docs/verdicts/p_nbx_tiled_conv2d_small_scale/`.

## Known limitations (transparency)

- **2/16 cells ⏸** : CPU pure × triton and CPU pure × triton_sequential
  blocked on absence of `triton-cpu` PyPI wheel
  (`meta-pytorch/triton-cpu`, build-from-source only at v0.2.0 release
  time). Backlog item `P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP`
  unblocks automatically when upstream ships wheels.
- **Op-level cross-device split (Gap B)** out of scope of v2 mandate:
  for models where a single op exceeds per-device VRAM. Backlog item
  `P-OP-LEVEL-CROSS-DEVICE-SPLIT` opens when a concrete model demands
  it (no Sana 4Kpx need it now that VAE fits 16 GiB post-S5 tiling).

## Validation artefacts

The mandate v2 verdict and per-cell R29 artefacts are committed:
- `docs/verdicts/p_prism_never_refuse_v2_closed.md` — full
  verdict, sub-chantier history, commit hashes, anti-régression table.
- `docs/verdicts/p_triton_chain_cpu_pointer/` — 6 coherent red
  apple PNGs (32g compiled / 32g triton / 16g compiled / 16g triton /
  16g triton-seq / 2× 16g triton / 2× 16g triton-seq).
- `docs/verdicts/p_nbx_tiled_conv2d_small_scale/` — microtest
  logs + 16g compiled anti-regression PNG + 32g triton baseline PNG.

## Key commits

Sub-chantiers and bugfixes that shaped this release:

| Sub-chantier / fix | Commits |
|---|---|
| S1 hybrid CPU+GPU dispatch | `de5fb9e` |
| S2 native sequential CPU (RoPE fix) | `8b4d020` |
| S3 triton-cpu install gate | `b3e479f`, `d0974e6` |
| S5 residual chain detection + wrapper | `198ab1b` → `33c6b21` |
| P-S5 depthwise tile-skip | `8af7848` |
| R30 chain skip on triton (interim) | `c9d2581` |
| S4 closure (cascade docs) | `f58f6cc` |
| JSONL dump O(N) | `da484ae` |
| P-NBX-TILED-CONV2D-SMALL-SCALE | `176bc7e`, `63edb03` |
| L1 NBX_LIVE_WATERMARK_TRACE | `993181f` |
| L2 per-slot LIVE_DUMP_SLOT | `5a714a5` |
| L4 band_streamed_chain_nbx + dispatcher | `f8a8ad8` |
| L4b NBXTensor isinstance + _set_device | `f997479` |
| C1 NBX_CHAIN_DIAG per-chain device-state | `9ddcc7b` |
| **C3d ROOT FIX — NBXTensor.__getitem__** | **`8a6daf2`** |
| C4 chain wrapper default-ON triton | `23de696` |
| MANDATE CLOSED verdict + R29 | `77571b7` |

## Tag

- `p-prism-never-refuse-v2-closed` posted on commit `77571b7` (mandate
  closure), pushed to `origin` and `gitlab`.
- `v0.2.0` semver release tag posted on this commit, pushed to both
  remotes.

## Upgrading from v0.1.x

No breaking API changes. The chain wrapper is default-ON on triton
modes (set `NBX_TRITON_CHAIN_WRAPPER=0` to fall back to the v0.1.x
behaviour). Other internal changes are transparent to call sites.
