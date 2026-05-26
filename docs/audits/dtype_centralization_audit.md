# dtype centralization audit

Goal: every dtype *decision* (precision protection, overflow upcast, range
clamp) is owned by the dtype engine (`core/dtype/`), not duplicated or
shadowed elsewhere. Format conversions and mathematical-correctness fp32
(diffusion betas, kernel-internal accumulation) are NOT dtype decisions and
stay where they are.

Audit surface: every `.float()` / `.to(torch.float32)` / `65504` / string→
`torch.dtype` site outside `core/dtype/`. The raw grep returns ~970 hits;
the table below is the de-noised set after removing vendored reference
kernels (`kernels/triton_kernels_ref/`), the `NBXTensor` substrate, and
diffusion-math fp32.

## Verdict table

| Site | Manipulation | Verdict | Engine rule |
|---|---|---|---|
| `core/runtime/dtype_adapter.py` | Parallel hardware→dtype authority (`supports_dtypes`, fp16/bf16 selection) | **REPATRIATE → delete** | Dead code, 0 callers, not exported. The live authority is `core/dtype/config.py` (`HARDWARE_DTYPE_SUPPORT`, `parse_dtype`) reached through Prism. A second copy is an R30/R23 divergence risk. |
| `core/runtime/graph/compiled_ops.py` `_rms_norm_fn` | Hand-rolled fp32-variance RMSNorm | **REPATRIATE** | → `engine.rms_norm_fp32`. Byte-identical to prior inline. |
| `core/runtime/graph/sequential_dispatcher.py` `_dispatch_rms_norm` | Hand-rolled fp32-variance RMSNorm | **REPATRIATE** | → `engine.rms_norm_fp32`. Byte-identical to prior inline. |
| `core/flow/stages/vibevoice.py` `_vv_rms_norm` | Hand-rolled fp32-variance RMSNorm | **REPATRIATE** | → `engine.rms_norm_fp32`. Equivalent (uniform-fp16 stage: cast-back `x.dtype`≡`weight.dtype`). |
| `core/runtime/graph_executor.py` (720–761) | AMP enable/disable per component, op-level upcast | **KEEP** | Proper seam: `graph_executor` decides the per-component `amp_enabled` policy; `DtypeEngine` owns the op-level upcast. Already engine-delegated. |
| `core/runtime/graph_executor.py:3144` + `compiled_sequence.py:2130` `gate_scores.float()` | MoE router fp32 upcast | **DONE → `engine.routing_upcast_fp32`** | Both torch sites routed through the engine seam (single authority). Determinism-sensitive — proven byte-identical on DeepSeek-MoE compiled+sequential (tokens IDENTICAL + MoE op-level dump byte-identical, 27 layers). Verdict: `validation_outputs/p_dtype_moe_router_fp32/`. Triton-path symmetry posted as P-DTYPE-MOE-ROUTER-FP32-TRITON-MIRROR. |
| `core/prism/solver.py:2858,2904` (`FP16_MAX=65504`) | bf16-weights→fp16 range scan | **KEEP** | Prism's offline placement decision (which compute dtype a component can take). Prism's domain, not a runtime protection hack. |
| `core/runtime/graph/compiled_sequence.py:3323` `nan_to_num(±65504)` | NaN/Inf guard | **KEEP** | Diagnostic, gated behind `nan_guard_verbose`. Debug tooling (§5.8 toolkit), not the precision policy. |
| `kernels/ops/topk.py:28,29` (`±65504`) | fp16 min/max masking constants | **KEEP** | Triton kernel-internal (R33-pure). Kernel layer's domain. |
| `core/module/scheduler/{diffusion,consistency,utils}` fp32 | betas / alphas / sigmas in fp32 | **KEEP** | Mathematical correctness of the diffusion schedule, not a dtype-protection decision. |
| `core/flow/stages/{kokoro,vibevoice}.py` `_coerce_torch_dtype` | string→`torch.dtype` at the triton-engine boundary | **KEEP** | CLAUDE.md doctrine: the triton engine returns a dtype *string*; stage handlers coerce it. Sanctioned boundary. |
| `core/flow/{audio_utils,next_token_diffusion,audio}.py` local string→dtype maps | string→`torch.dtype` duplication (3 inline copies of `DTYPE_MAP`) | **REPATRIATE** | → `config.get_torch_dtype`. The `manifest.get("dtype","float16")` default guarantees a valid key, so byte-identical for every real model. |
| `core/cfg/engine.py` `_resolve_torch_dtype` | `Union[str, torch.dtype]` resolver | **KEEP** | Already delegates to `config.get_torch_dtype`; a thin Union-tolerant adapter, not a duplicated map. |

## Repatriated (R23-proven)

- `dtype_adapter.py` deleted (dead parallel authority).
- fp32-RMSNorm triplet → single `engine.rms_norm_fp32`.
- 3 inline `DTYPE_MAP` copies in the audio flows → `config.get_torch_dtype`.

R23 evidence: TinyLlama coherent in **compiled** and **sequential** ("The
capital of France is Paris."); VibeVoice STT-faithful via the `_vv_rms_norm`
**and** `_compute_dtype` paths ("Hello, this is a final validation."); Whisper
STT correct on the reference clip (audio.py + audio_utils `get_compute_dtype`).
The two LLM-fleet RMSNorm sites are byte-identical to their prior inline code
by construction; the VibeVoice RMSNorm site is equivalent under the
uniform-fp16 stage; the 3 compute-dtype readers are byte-identical for every
real model (manifest dtype is always a valid key).

## Resolved

- **P-DTYPE-MOE-ROUTER-FP32** (closed 2026-05-26): the MoE router fp32
  upcast now lives in `engine.routing_upcast_fp32`, the single authority for
  that dtype-protection policy. Both torch sites
  (`graph_executor.py:3144` sequential + `compiled_sequence.py:2130`
  compiled) route through it; body is `x.float()`, byte-identical to the
  inline upcast. Hocine corrected the original DEFER: a single-site
  dtype-treatment is still out-of-doctrine where it is — the engine must be
  the unique authority, so even a mono-site upcast belongs in the engine.
  Determinism proven byte-identical on DeepSeek-MoE compiled+sequential
  (tokens IDENTICAL + compiled MoE op-level dump byte-identical across all 27
  MoE layers; sequential run-to-run deterministic). Qwen3-30B skipped as
  overdetermination (provable 1:1 rename). Verdict:
  `validation_outputs/p_dtype_moe_router_fp32/`. Triton-path symmetry is open
  as **P-DTYPE-MOE-ROUTER-FP32-TRITON-MIRROR** (R33-sealed second engine).
