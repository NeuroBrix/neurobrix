# VIDEO first diagnosis — start from the existing SANA-Video container (2026-05-26)

**Scope**: diagnose the existing video runtime by starting from the one
already-built video container (SANA-Video_2B_720p). Investigation only — no
new tracing, no blind fixes. Method: static inspection of the built container
(`manifest.json`, `topology.json`, per-component `graph.json` /
`weights_index.json`, `runtime/variables.json`), a direct reproduction of the
runtime break with full traceback, and cross-reference against the vendor
diffusers SANA-Video classes in `/home/mlops/ml/venv` (`0.38.0.dev0`).

## Headline verdict

**The SANA-Video trace is HEALTHY — do NOT re-trace.** The only thing
standing between the existing container and a runtime is one bounded Prism
break (`P-PRISM-VIDEO-5D-UNPACK`, a single 4-tuple shape unpack). The build
side already produces a correct 5D NCTHW video container with a symbolic
temporal dimension, standard-normalized weight keys, and the standard
`iterative_process` diffusion flow. **First build target = SANA-Video**
(capitalise on the existing healthy trace), run at the trace frame count
first.

## Axis 1 — SANA-Video container trace health: HEALTHY

Container: the built SANA-Video_2B_720p (cache). Components: `text_encoder`,
`transformer`, `vae`, plus `tokenizer` + `scheduler` modules.

| Check | Result | Evidence |
|---|---|---|
| Flow type | **`iterative_process`** (the existing image-diffusion flow) | `topology.json` flow: pre_loop=[text_encoder], loop driver=scheduler comp=[transformer], post_loop=[vae] |
| 5D NCTHW captured | **Yes** | transformer `hidden_states` shape `[2,128,11,22,40]`; vae `z` `[1,128,4,8,8]` |
| Temporal dim symbolic? | **Yes — `s6="time"`** from `hidden_states::dim_2` (trace 11); `s7=height`, `s8=width` | transformer `graph.json` `symbolic_context.symbols` (9 symbols: batch×4, seq_len×2, time, height, width) |
| Core token flatten symbolic? | **Yes** — `s5·s6·s7·s8` | `aten.view::17` args use `{symbol s5}·{symbol s6}·{s7}·{s8}` → `[19360,2240]` = 2·11·22·40 |
| Standard weight-key naming | **Yes** | 480 `block.*` weight names (`block.0.self_attn.query.weight`, …), **0 vendor names** (no `transformer_blocks`/`to_q`); attn=340, ffn=120, norm=81 |
| Components correct vs vendor | **Yes** | text_encoder = **Gemma2** (hidden 2304, 26 layers); transformer = SanaVideo DiT (20L/20H×112, cross 2240); vae = 3D temporal (latent 128, spatial_compression 32, **temporal_compression 8**) |
| VAE genuinely 3D? | **Yes** | vae `graph.json`: `aten::reflection_pad3d`×45 paired 1:1 with `aten::convolution`×45 → 3D temporal convs (the manifest `backbone:"unet_2d"` is a taxonomy-label gap, not a trace defect) |
| Graph health (bypassed forward) | **Clean** | well-formed DAG (`execution_order`, `input/output_tensor_ids`, `forks`, `joins`); no openaudio-style orphan-consumer pattern |
| fp32 protection (fp16-runtime safety) | **Structural** | transformer `custom::rms_norm`×81 (engine fp32-variance seam) + `aten::_to_copy`×11; the manifest `dtype:"float32"` does not strand the runtime because the norm protection lives in the reassembled op, not in traced `_to_copy` |

### One caveat (non-blocking): partial temporal reshape freeze

The temporal dim is symbolic in the **core** token flow, but a subset of
setup/positional reshapes bake the trace literal `11`: of 1048 view/reshape
ops, 213 reference a symbol token and ~288 carry a bare literal `11`
(e.g. `aten.view::0` `value:[11,1,1,-1]` → positional/axis setup on the
temporal axis — the two sampled, `view::0`/`::3`, clearly are; full coverage
of the ~288 not individually verified). This is a granite-style **partial**
freeze: the container is runnable **at the trace frame count** (11 latent
frames), but an arbitrary frame count would hit at least some of these baked
reshapes. Maturing them to the `s6` symbol is a follow-on (same class as the
audio symbolic-maturation work), **not** a first-run blocker.

### Cosmetic notes
- `manifest.origin.url` is scrape-garbled (`…/Sana"><img`). Nothing in the
  runtime parses it; ignore.
- `manifest.components.*.backbone` = `"unknown"`/`"unet_2d"` are taxonomy
  label gaps; `topology.extracted_values` carry the correct types
  (gemma2, the DiT params, the 3D-VAE compression ratios).

## Axis 2 — P-PRISM-VIDEO-5D-UNPACK: localized, BOUNDED

Reproduced directly (full traceback):

```
solver.solve_smart → solve (core/prism/solver.py:746)
  → _detect_op_level_tiling_pairs (solver.py:793)
  → profiler.estimate_peak_memory (core/prism/profiler.py:361)
  → estimate_op_workspace_bytes (core/prism/memory_estimator.py:61)
ValueError: too many values to unpack (expected 4)
```

**Exact site**: `core/prism/memory_estimator.py:61` — `n, in_c, _, _ = in_shape`.
The cuDNN conv-workspace estimator guards `if len(in_shape) < 4: return 0`
(line 60) but a **5D** conv input `[N,C,T,H,W]` (the 3D-VAE conv) passes that
guard (5 ≥ 4) and then fails the **4-target** unpack. The error string is
generic Python; the reproduction confirms this is the firing site, not an
inference.

**Bounded crash; downstream unverified** (proved: the crash is one site;
inferred by static scan: the rest looks ndim-agnostic — but the repro stops
at the first 5D-sensitive site, so any 4D assumption downstream of the
conv-workspace estimator is invisible to this evidence):
- It is the **only** 4-tuple shape unpack in the whole `core/prism/` module.
- `profiler.py` has no 4D-indexing assumptions (delegates conv workspace to
  the estimator); the placement strategies (`core/strategies/*`) have no
  4D shape unpacks; tensor-byte budgeting uses `prod(shape)` (ndim-agnostic);
  the SDPA-workspace branch uses indexing (`q_shape[0..2]`), 5D-safe.
- Fix shape: extend `estimate_op_workspace_bytes` to handle a 5D conv input
  (compute the im2col workspace with the temporal extent), i.e. add `T` to
  the formula — a localized change at one function. The build mandate should
  still re-run after the fix to confirm no further 4D assumption surfaces
  downstream (the reproduction stops at the first 5D-sensitive site).

**Secondary gap (not this crash)**: the R31 tiling engine
(`core/module/tiling_engine.py:8`, `:151` `if len(shape)==4`) is explicitly
4D-only. It **gracefully skips** 5D (no crash) — so video spatial-temporal
tiling is unsupported, deferred until a video op actually overflows one GPU.

## Axis 3 — capability reusability inventory (universal-engine doctrine)

| Capability | Status for video | Detail |
|---|---|---|
| `iterative_process` diffusion flow | **Reused as-is** | the SANA-Video container already declares it; the flow orchestrates WHEN (scheduler loop), the 5D compute lives in the graph — no 3D flow variant needed |
| Scheduler engine | **Reused** | SANA-Video DPM++ flow scheduler in `modules/scheduler/`; the existing scheduler factory handles it |
| 3D DiT execution | **No new runtime capability** | "executing a 3D transformer" = running the traced ATen DAG on 5D tensors; the graph executor is dimension-agnostic (view/mm/addmm/convolution/rms_norm on 5D). The 3D-ness is in the trace, not a runtime gap (modulo the Prism estimator crash) |
| 3D VAE — **compiled** mode | **Expected to work — confirm at first run** | inference: `aten::convolution` on a 5D input dispatches via PyTorch's existing path to cuDNN's nd-conv; `aten::reflection_pad3d` is ATen-native, dimension-agnostic. No new compiled-mode kernel needed — but not yet run |
| 3D VAE — **triton** mode | **New kernels needed** | Triton-pure `conv3d` (+ `reflection_pad3d`) wrappers are unbuilt (a `conv3d` reference exists under `kernels/triton_kernels_ref/`); triton-mode video is a follow-on, compiled mode first |
| DtypeEngine video ops | **Covered** | `conv3d`/`conv_transpose3d` already in `AMP_FP16_OPS` (`core/dtype/engine.py`); norm protection via `engine.rms_norm_fp32`. `reflection_pad3d` is a dimension-agnostic pad (no dtype protection needed). No new engine upcast required |
| Symbolic 5D NCTHW | **Anticipated + partially matured** | the symbolic contract defines the NCTHW axis roles (`docs/architecture/symbolic-shapes-contract.md`); the trace symbolizes T/H/W; the partial reshape freeze (Axis 1 caveat) is the maturation gap |

**Doctrine note (universal/reusable)**: nothing here is SANA-Video-specific.
The bounded Prism estimator fix, the reused `iterative_process` flow, and the
DtypeEngine coverage all generalise to the Wan / CogVideoX / Allegro / Mochi
families once they are traced. The VibeVoice next-token-diffusion flow stays a
banked, reusable acquisition for a *future* autoregressive-token video model —
none of the current 11 use it (all latent diffusion).

## Axis 4 — sequencing verdict

1. **Do NOT re-trace SANA-Video.** The container is healthy (5D, symbolic
   T/H/W, standard-normalized weight keys, correct components, `iterative_process` flow,
   structurally fp32-protected). Re-tracing would be wasted work.
2. **Unblock Prism 5D first** — the bounded fix at
   `core/prism/memory_estimator.py:61` (5D conv-workspace). This is the single
   gate to a first end-to-end video run.
3. **First build target = SANA-Video**, run at the **trace frame count**
   (11 latent frames) initially. Rationale: it is already traced and healthy,
   so unblocking one bounded site validates the entire video runtime path
   (5D Prism placement → 3D DiT exec → 3D VAE exec → `iterative_process`
   flow) against the vendor `pipeline_sana_video.py` oracle (§5.8) at minimum
   cost. **Wan2.1-T2V-1.3B** (untraced) becomes the natural **second** target —
   and SANA-Video's linear-attention lineage already reuses the proven Sana
   image-DiT patterns.
4. **First-run punch list**: (a) confirm fp16-runtime activations on the
   fp32-traced graph do not NaN at any DiT block (the openaudio post-mortem
   check — risk assessed low, the 81 `custom::rms_norm` ops protect the
   variance, but verify); (b) confirm compiled-mode 3D-VAE 5D conv dispatch
   actually runs (Axis 3); (c) re-run the solve after the line-61 fix to
   confirm no further 4D assumption surfaces downstream.
5. **Follow-ons** (named, not blockers for the first run): frame-count
   symbolic maturation (~288 baked temporal reshapes); triton-mode 3D-VAE
   kernels (`conv3d` + `reflection_pad3d`); R31 video tiling (4D→5D) when a
   video op first overflows one GPU.

**Net**: video is much closer than the raw inventory implied. One healthy
traced model + one bounded Prism fix away from a first end-to-end run.
P-PRISM-VIDEO-5D-UNPACK is the whole gate.
