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
| `core/runtime/graph_executor.py:3144` `gate_scores.float()` | MoE router fp32 upcast | **DEFER (fingerprint plan)** | Determinism-sensitive (P-TRITON-MOE-DETERMINISM). Moving it requires `NBX_OP_FINGERPRINT` before/after on DeepSeek-MoE + Qwen3-30B to prove byte-identity. Posted as follow-up, not an orphan TODO. |
| `core/prism/solver.py:2858,2904` (`FP16_MAX=65504`) | bf16-weights→fp16 range scan | **KEEP** | Prism's offline placement decision (which compute dtype a component can take). Prism's domain, not a runtime protection hack. |
| `core/runtime/graph/compiled_sequence.py:3323` `nan_to_num(±65504)` | NaN/Inf guard | **KEEP** | Diagnostic, gated behind `nan_guard_verbose`. Debug tooling (§5.8 toolkit), not the precision policy. |
| `kernels/ops/topk.py:28,29` (`±65504`) | fp16 min/max masking constants | **KEEP** | Triton kernel-internal (R33-pure). Kernel layer's domain. |
| `core/module/scheduler/{diffusion,consistency,utils}` fp32 | betas / alphas / sigmas in fp32 | **KEEP** | Mathematical correctness of the diffusion schedule, not a dtype-protection decision. |
| `core/flow/stages/{kokoro,vibevoice}.py` `_coerce_torch_dtype` | string→`torch.dtype` at the triton-engine boundary | **KEEP** | CLAUDE.md doctrine: the triton engine returns a dtype *string*; stage handlers coerce it. Sanctioned boundary. |
| `core/flow/{audio_utils,next_token_diffusion,audio}.py`, `core/cfg/engine.py` local string→dtype maps | string→`torch.dtype` duplication | **DEFER (cosmetic)** | Not protection logic — duplicated mapping. Could route to one `config` helper. No behaviour risk; 4 files; low priority. |

## Repatriated (R23-proven)

- `dtype_adapter.py` deleted (dead parallel authority).
- fp32-RMSNorm triplet → single `engine.rms_norm_fp32`.

R23 evidence: TinyLlama coherent in **compiled** and **sequential** ("The
capital of France is Paris."); VibeVoice STT-faithful via the `_vv_rms_norm`
path. The two LLM-fleet sites are byte-identical to their prior inline code
by construction; the VibeVoice site is equivalent under the uniform-fp16
stage and empirically validated.

## Deferred (named, with protocol — not fictional sub-chantiers)

- **P-DTYPE-MOE-ROUTER-FP32**: move `gate_scores.float()` into the engine
  only after an `NBX_OP_FINGERPRINT` before/after run proves byte-identity
  on DeepSeek-MoE + Qwen3-30B (determinism-sensitive path).
- **P-DTYPE-STRING-COERCE-DEDUP**: fold the 4 flow-handler string→dtype
  maps into one `config` helper (cosmetic; the 2 stage-handler ones stay
  per the triton-boundary doctrine).
