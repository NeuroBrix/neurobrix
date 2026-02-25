# Triton Kernels Reference - Inventory

**Total: 1,097 kernel files across 20 projects**
**Disk usage: ~12MB (cleaned)**

---

## Kernel Type Classification

### Definition:
- **Pure Triton** (`P`): Files with only `@triton.jit` kernels, no PyTorch wrapper code
- **With Wrapper** (`W`): Files with `@triton.jit` + PyTorch entry point (torch.empty, grid launch, etc.)
- **With Autograd** (`A`): Files with `@triton.jit` + `torch.autograd.Function` (forward/backward)

---

## Projects Overview

| Project | Source | Files | @jit | Pure | Wrapper | Autograd | Primary Type |
|---------|--------|-------|------|------|---------|----------|--------------|
| flaggems | FlagOpen/FlagGems | 560 | 1474 | 188 | 350 | 22 | W (63%) |
| fla | flash-linear-attention | 159 | 378 | 21 | 81 | 57 | A+W (87%) |
| vllm | vllm-project/vllm | 21 | 48 | 5 | 16 | 0 | W (76%) |
| liger | linkedin/Liger-Kernel | 21 | 46 | 0 | 1 | 20 | A (95%) |
| unsloth | unslothai/unsloth | 10 | 20 | 5 | 2 | 3 | Mixed |
| mlstm_kernels | NX-AI/mlstm_kernels | 19 | 52 | 12 | 0 | 7 | P (63%) |
| flash_attn | Dao-AILab | 14 | 68 | 1 | 8 | 5 | W+A (93%) |
| mamba | state-spaces/mamba | 10 | 32 | 1 | 2 | 7 | A (70%) |
| attorch | BobMcDear/attorch | 14 | 82 | 13 | 0 | 1 | P (93%) |
| applied_ai | pytorch-labs | 12 | 23 | 0 | 10 | 2 | W (83%) |
| kernl | ELS-RD/kernl | 13 | 28 | 1 | 6 | 6 | W+A (92%) |
| triton_tutorials | triton-lang | 11 | 28 | 0 | 9 | 2 | W (82%) |
| gemlite | dropbox/gemlite | 9 | 18 | 0 | 9 | 0 | W (100%) |
| sageattention | thu-ml/SageAttention | 8 | 15 | 0 | 8 | 0 | W (100%) |
| flagattention | FlagOpen | 6 | 14 | 0 | 4 | 2 | W+A |
| qattn | IBM/qattn | 3 | 5 | 1 | 0 | 2 | A (67%) |
| mamba | Mamba SSM | 10 | 32 | 1 | 2 | 7 | A (70%) |
| llm_note | harleyszhang | 1 | 1 | 0 | 1 | 0 | W |
| conch | stackav-oss | 16 | - | - | - | - | (Mixed) |
| kernelheim | debashishc | 1 | 1 | 0 | 1 | 0 | W |
| scattermoe | shawntan | 2 | 6 | 0 | 2 | 0 | W (100%) |

### Summary by Type:
- **Pure Triton (P):** 248 files (ideal for learning kernel implementation)
- **With Wrapper (W):** 510 files (ready-to-use with PyTorch)
- **With Autograd (A):** 139 files (full forward/backward support)

---

## 1. FLAGGEMS (FlagOpen/FlagGems) - 560 files, 1474 @triton.jit

**Most comprehensive - Multi-vendor support**

### Kernel Types:
| Type | Files | @jit | Description |
|------|-------|------|-------------|
| Pure Triton | 188 | 370 | `angle.py, fill.py, flip.py, pow.py, sort.py...` |
| With Wrapper | 350 | 1038 | `cumsum.py, multinomial.py, add.py, addmm.py...` |
| With Autograd | 22 | 66 | `gelu.py, embedding.py, outer.py, rms_norm.py...` |

### Structure:
```
flaggems/
├── nvidia/
│   ├── common/     # ~169 kernels (main ops)
│   ├── ampere/     # A100 specific
│   └── hopper/     # H100 specific
├── amd/            # AMD ROCm
├── ascend/         # Huawei NPU (61 kernels)
├── cambricon/      # MLU (133 kernels)
├── kunlunxin/      # Baidu Kunlun (164 kernels)
├── metax/          # 31 kernels
├── mthreads/       # Moore Threads (33 kernels)
├── hygon/          # DCU (13 kernels)
├── iluvatar/       # 3 kernels
├── arm/            # 3 kernels
└── aipu/           # Arm Ethos-U (3 kernels)
```

### NVIDIA Common Ops:
```
abs, add, addcdiv, addcmul, addmm, addmv, addr, all, amax, angle, any,
arange, argmax, argmin, atan, attention, avg_pool2d, baddbmm, batch_norm,
bitwise_*, bmm, cat, celu, clamp, contiguous, conv1d, conv2d, conv3d,
conv_depthwise2d, copy, cos, count_nonzero, cummax, cummin, cumsum,
diag, diag_embed, diagonal, div, dot, dropout, elu, embedding, eq, erf,
exp, exp2, exponential_, eye, fill, flash_api, flash_kernel, flip, full,
full_like, gather, ge, gelu, glu, groupnorm, gt, hstack, index, index_add,
index_put, index_select, isclose, isfinite, isin, isinf, isnan, kron,
layernorm, le, lerp, linspace, log, log_sigmoid, log_softmax, logical_*,
logspace, lt, masked_fill, masked_select, max, max_pool2d, maximum, mean,
min, minimum, mm, mm_streamk, mse_loss, mul, multinomial, mv, nan_to_num,
ne, neg, nllloss, nonzero, normal, ones, ones_like, pad, polar, pow,
prod, quantile, rand, rand_like, randn, randn_like, randperm, reciprocal,
relu, repeat, repeat_interleave, resolve_conj, resolve_neg, rms_norm,
rsqrt, scaled_softmax, scatter, sigmoid, silu, sin, slice_scatter,
softmax, softplus, sort, sqrt, stack, std, sub, sum, tan, tanh, threshold,
tile, to, topk, trace, triu, uniform, unique, upsample_*, var_mean, vdot,
vector_norm, vstack, weightnorm, where, zeros, zeros_like
```

---

## 2. FLA - Flash Linear Attention (sustcsonglin/flash-linear-attention) - 159 files, 378 @triton.jit

**Largest single collection for linear/subquadratic attention**

### Kernel Types:
| Type | Files | @jit | Key Files |
|------|-------|------|-----------|
| Pure Triton | 21 | 49 | `logcumsumexp.py, logsumexp.py, wy_fast.py...` |
| With Wrapper | 81 | 140 | `chunk_A_fwd.py, chunk_bwd.py, chunk_O_fwd.py...` |
| With Autograd | 57 | 189 | `channel_mixing.py, compression.py, gate.py, recurrent.py...` |

### Structure:
```
fla/nvidia/ops/
├── attn/               # Standard attention
├── based/              # Based attention
├── abc/                # ABC attention
├── delta_rule/         # Delta rule
├── gated_delta_rule/   # Gated delta
├── gla/                # Gated Linear Attention
├── gsa/                # GSA attention
├── hgrn/               # HGRN
├── kda/                # KDA attention
├── mesa_net/           # MESA-Net
├── nsa/                # NSA compression
├── path_attn/          # Path attention
├── rebased/            # Rebased
├── rwkv4/              # RWKV-4
├── rwkv6/              # RWKV-6
├── rwkv7/              # RWKV-7
├── simple_gla/         # Simple GLA
├── ttt/                # Test-Time Training
├── generalized_delta_rule/  # DPLR/IPLR variants
├── gated_delta_product/
├── comba/
├── log_linear_attn/
├── deltaformer/
└── utils/              # Cumsum, softmax, matmul, etc.

fla/nvidia/modules/
├── activations.py
├── convolution.py
├── layernorm.py
├── layernorm_gated.py
├── rotary.py
├── l2norm.py
├── fused_cross_entropy.py
├── fused_linear_cross_entropy.py
├── fused_kl_div.py
└── token_shift.py
```

---

## 3. VLLM (vllm-project/vllm) - 43 kernels

**Production LLM inference engine**

```
triton_attn.py              # Triton attention
triton_decode_attention.py  # Decode-phase attention
triton_unified_attention.py # Unified attention kernel
triton_mla.py               # Multi-head Latent Attention
flashmla.py                 # Flash MLA
paged_attn.py               # Paged attention
prefix_prefill.py           # Prefix caching
triton_scaled_mm.py         # Scaled matmul
awq_triton.py               # AWQ quantization
fused_moe_lora_op.py        # Fused MoE + LoRA
triton_deep_gemm_moe.py     # MoE GEMM
gpt_oss_triton_kernels_moe.py
lora_expand_op.py / lora_shrink_op.py
rocm_aiter_mla_sparse.py    # AMD ROCm support
ops/                        # Additional ops
```

---

## 4. LIGER (linkedin/Liger-Kernel) - 24 kernels

**LLM training optimization**

```
cross_entropy.py          fused_add_rms_norm.py     fused_linear_cross_entropy.py
fused_linear_jsd.py       fused_neighborhood_attention.py
dyt.py                    geglu.py                  group_norm.py
jsd.py                    kl_div.py                 layer_norm.py
llama4_rope.py            multi_token_attention.py  poly_norm.py
qwen2vl_mrope.py          rms_norm.py               rope.py
softmax.py                sparsemax.py              swiglu.py
tiled_mlp.py              tvd.py
```

---

## 5. UNSLOTH (unslothai/unsloth) - 23 kernels

**Efficient LLM finetuning**

```
cross_entropy_loss.py   rms_layernorm.py        layernorm.py
rope_embedding.py       swiglu.py               geglu.py
fast_lora.py            fp8.py                  flex_attention.py
forward.py              backward.py             autotuning.py
tuning.py               llama4_moe.py           qwen3_moe.py
moe_block.py            moe_ops.py              moe_utils.py
```

---

## 6. MLSTM_KERNELS (NX-AI/mlstm_kernels) - 22 kernels

**mLSTM/linear attention**

```
chunkwise/          # 13 kernels (mLSTM core)
attention/          # Flash attention variants
linear_attention/   # GLA variants
```

---

## 7. FLASH_ATTN (Dao-AILab/flash-attention) - 19 kernels

**Official Flash Attention implementation**

```
nvidia/
├── flash_attn_triton.py     # Main Triton implementation
├── flash_attn_triton_og.py  # Original version
└── ops/
    ├── cross_entropy.py
    ├── layer_norm.py
    ├── linear.py
    ├── mlp.py
    ├── rotary.py
    └── k_activations.py

amd/                         # AMD ROCm support
├── bwd_prefill*.py          # Backward prefill variants
├── fwd_*.py                 # Forward kernels
├── fp8.py                   # FP8 support
└── interface_fa.py
```

---

## 8. MAMBA (state-spaces/mamba) - 18 kernels

**State Space Models (Mamba/Mamba2)**

```
mamba_simple.py             # Mamba-1 simple
mamba2.py                   # Mamba-2
mamba2_simple.py            # Mamba-2 simple
selective_scan_interface.py # Core SSM scan
selective_state_update.py
ssd_chunk_scan.py           # SSD chunk operations
ssd_chunk_state.py
ssd_combined.py
ssd_bmm.py
ssd_state_passing.py
ssd_minimal.py
layer_norm.py               # Mamba LayerNorm
layernorm_gated.py          # Gated LayerNorm
mixer_seq_simple.py
softplus.py
k_activations.py
```

---

## 9. CONCH (stackav-oss/conch) - 16 kernels

**vLLM-compatible kernels**

```
attention/          # paged_attention.py, varlen_attention.py
normalization/      # rms_norm.py, gemma_rms_norm.py
activation/         # gelu_tanh_and_mul.py, silu_and_mul.py
embedding/          # rotary_embedding.py
quantization/       # gemm.py, int8.py, fp8.py
vision/             # bev_pool.py, nms.py, voxelization.py
vllm/               # copy_blocks.py, reshape_and_cache.py
```

---

## 10. ATTORCH (BobMcDear/attorch) - 14 files, 82 @triton.jit

**PyTorch nn.Module reimplemented in Triton - 93% Pure Triton!**

### Kernel Types:
| Type | Files | @jit | Key Files |
|------|-------|------|-----------|
| Pure Triton | 13 | 78 | All kernels except multi_head_attention |
| With Autograd | 1 | 4 | `multi_head_attention_kernels.py` |

**Best for learning**: Clean pure Triton implementations without wrapper code.

```
act_kernels.py              # Activations (ReLU, GELU, SiLU, etc.) [P - 46 kernels]
batch_norm_kernels.py       # BatchNorm
layer_norm_kernels.py       # LayerNorm
rms_norm_kernels.py         # RMSNorm
linear_kernels.py           # Linear/Dense
conv_kernels.py             # Conv2D
dropout_kernels.py          # Dropout
softmax_kernels.py          # Softmax
glu_kernels.py              # GLU variants
cross_entropy_loss_kernels.py
nll_loss_kernels.py
p_loss_kernels.py           # L1/L2 loss
multi_head_attention_kernels.py
math.py
```

---

## 11. APPLIED_AI (pytorch-labs/applied-ai) - 14 kernels

**PyTorch Labs research kernels**

```
moe/                    # Column-major MoE GEMM (4x faster)
flash_attention/        # Flash Attention variants
fp8/                    # FP8 quantization
gptq/                   # GPTQ quantization
paged_attention/        # Paged attention for KV cache
```

---

## 12. KERNL (ELS-RD/kernl) - 13 kernels

**Transformer optimization**

```
attention.py            # Main attention kernel
attention_skinny.py     # Skinny matrix attention
attention_vec_mat.py    # Vector-matrix attention
fused_kernel_attention.py
fused_kernel_ff.py      # Fused feed-forward
fused_kernel_proj_qkv.py # Fused QKV projection
fused_kernel_fp8.py     # FP8 support
layer_norm.py
linear_layer.py
batched_matmul.py
kernel.py
activation_func.py
transpose.py
```

---

## 13. TRITON_TUTORIALS (triton-lang/triton) - 11 kernels

**Official Triton tutorial examples**

```
01-vector-add.py            # Vector addition
02-fused-softmax.py         # Fused softmax
03-matrix-multiplication.py # GEMM
04-low-memory-dropout.py    # Dropout
05-layer-norm.py            # LayerNorm
06-fused-attention.py       # Flash Attention tutorial
07-extern-functions.py      # External function calls
08-grouped-gemm.py          # Grouped GEMM
09-persistent-matmul.py     # Persistent kernel
10-block-scaled-matmul.py   # Block-scaled GEMM
11-programmatic-dependent-launch.py
```

---

## 14. LLM_NOTE (harleyszhang/llm_note) - 11 kernels

**Transformer components**

```
normalization/  # layernorm.py, rmsnorm.py, batchnorm.py
attention/      # mha.py, deepseekv_mla.py, softmax.py
embedding/      # rope.py, sinusoidal.py
activation/     # FFN_SwiGLU.py
misc/           # moe_gate.py
```

---

## 15. GEMLITE (dropbox/gemlite) - 10 kernels

**Quantized GEMM kernels**

```
gemm_kernels.py               # Standard GEMM
gemm_splitK_kernels.py        # Split-K for tall-skinny
gemm_splitK_persistent_kernels.py
gemv_kernels.py               # Matrix-vector ops
gemv_splitK_kernels.py
gemv_revsplitK_kernels.py
A16W3_gemm.py                 # 3-bit weight GEMM
A16W5_gemm.py                 # 5-bit weight GEMM
config.py
utils.py
```

---

## 16. SAGEATTENTION (thu-ml/SageAttention) - 10 kernels

**INT8/FP8 quantized attention (2-5x faster than FlashAttention)**

```
core.py                      # Main quantized attention
quant.py                     # Quantization utilities
quant_per_block.py           # Per-block quantization
quant_per_block_varlen.py    # Variable length support
quant_per_thread.py          # Per-thread quantization
attn_qk_int8_per_block.py    # INT8 Q/K attention
attn_qk_int8_per_block_causal.py
attn_qk_int8_block_varlen.py
api.py
```

---

## 17. FLAGATTENTION (FlagOpen/FlagAttention) - 6 kernels

**Memory-efficient attention operators**

```
flash.py                # Flash attention in Triton
piecewise.py            # Piecewise attention (Aquila-2 model)
paged.py                # Paged attention
split_kv.py             # Split K/V attention
dropout.py              # Attention dropout
total.py                # Combined operations
```

---

## 18. QATTN (IBM/qattn) - 6 kernels

**Mixed-precision Vision Transformer attention**

```
_flash_attention.py     # Quantized flash attention
_matmul.py              # Quantized matmul
_matmul_configs.py      # Matmul configurations
_quantize.py            # Quantization utilities
backends.py             # Backend selection
lower.py                # Lowering passes
```

---

## 19. KERNELHEIM (debashishc/kernelheim) - 2 kernels

**Educational implementations**

```
flash_attention.py      # Educational FA implementation
softmax.py              # Softmax kernel
```

---

## 20. SCATTERMOE (shawntan/scattermoe) - 2 kernels

**Sparse Mixture of Experts**

```
ops.py                  # Main MoE operations
single.py               # Single-expert ops
```

---

## Quick Reference by Op Type

| Op Type | Best Source | Path |
|---------|-------------|------|
| **Normalization** | | |
| GroupNorm | Liger | `liger/nvidia/group_norm.py` |
| LayerNorm | FlagGems | `flaggems/nvidia/common/layernorm.py` |
| RMSNorm | Unsloth | `unsloth/nvidia/rms_layernorm.py` |
| BatchNorm | attorch | `attorch/nvidia/batch_norm_kernels.py` |
| **Attention** | | |
| Flash Attention | flash_attn | `flash_attn/nvidia/flash_attn_triton.py` |
| Quantized Attention | SageAttention | `sageattention/nvidia/core.py` |
| Linear Attention (GLA) | FLA | `fla/nvidia/ops/gla/` |
| Paged Attention | vLLM | `vllm/nvidia/paged_attn.py` |
| MLA (DeepSeek) | vLLM | `vllm/nvidia/triton_mla.py` |
| **State Space** | | |
| Mamba-1 | mamba | `mamba/nvidia/mamba_simple.py` |
| Mamba-2 | mamba | `mamba/nvidia/mamba2.py` |
| RWKV | FLA | `fla/nvidia/ops/rwkv6/`, `fla/nvidia/ops/rwkv7/` |
| **Activations** | | |
| SwiGLU | Unsloth | `unsloth/nvidia/swiglu.py` |
| GELU | attorch | `attorch/nvidia/act_kernels.py` |
| GeGLU | Unsloth | `unsloth/nvidia/geglu.py` |
| **GEMM** | | |
| Standard GEMM | FlagGems | `flaggems/nvidia/common/mm.py` |
| Split-K GEMM | GemLite | `gemlite/nvidia/gemm_splitK_kernels.py` |
| Quantized GEMM | GemLite | `gemlite/nvidia/A16W3_gemm.py` |
| MoE GEMM | applied_ai | `applied_ai/nvidia/moe/` |
| **Embedding** | | |
| RoPE | Unsloth | `unsloth/nvidia/rope_embedding.py` |
| Embedding | FlagGems | `flaggems/nvidia/common/embedding.py` |
| **Loss Functions** | | |
| Cross Entropy | Unsloth | `unsloth/nvidia/cross_entropy_loss.py` |
| KL Divergence | FLA | `fla/nvidia/modules/fused_kl_div.py` |
| **MoE** | | |
| Fused MoE | vLLM | `vllm/nvidia/fused_moe_lora_op.py` |
| ScatterMoE | scattermoe | `scattermoe/nvidia/ops.py` |
| Col-Major MoE | applied_ai | `applied_ai/nvidia/moe/` |
| **Quantization** | | |
| FP8 | Unsloth | `unsloth/nvidia/fp8.py` |
| INT8 Attention | SageAttention | `sageattention/nvidia/attn_qk_int8_per_block.py` |
| AWQ | vLLM | `vllm/nvidia/awq_triton.py` |
| GPTQ | applied_ai | `applied_ai/nvidia/gptq/` |

---

## Vendor Coverage Matrix

| Vendor | Projects | Coverage |
|--------|----------|----------|
| NVIDIA (Common) | ALL | Full coverage (1000+ kernels) |
| NVIDIA Ampere (A100) | FlagGems, SageAttention | Optimized kernels |
| NVIDIA Hopper (H100) | FlagGems, SageAttention | FP8, TMA support |
| AMD ROCm | flash_attn, FlagGems, vLLM | Growing support |
| Huawei Ascend | FlagGems, Liger | 61+ kernels |
| Cambricon MLU | FlagGems | 133 kernels |
| Baidu Kunlun | FlagGems | 164 kernels |
| Moore Threads | FlagGems | 33 kernels |
| MetaX | FlagGems | 31 kernels |
| Hygon DCU | FlagGems | 13 kernels |

---

## Statistics

- **Total Kernel Files:** 1,097
- **Total Projects:** 20
- **@triton.jit Decorators:** 2,300+
- **Pure Triton Kernels:** 248 files
- **With Wrapper (PyTorch):** 510 files
- **With Autograd (Forward/Backward):** 139 files
- **NVIDIA Kernels:** ~1,000+
- **Multi-Vendor Kernels:** ~500 (via FlagGems)
- **Linear Attention Variants:** 159 files (FLA)
- **State Space Models:** 10 files (Mamba)
- **Quantization Kernels:** 40+ (SageAttention, GemLite, vLLM)

---

## Best Sources by Use Case

### For Learning Pure Triton Kernel Writing:
| Project | % Pure | Reason |
|---------|--------|--------|
| **attorch** | 93% | Clean kernel implementations, no wrapper clutter |
| **mlstm_kernels** | 63% | Well-structured chunkwise kernels |
| **unsloth** | 50% | Simple forward/backward patterns |

### For Production PyTorch Integration:
| Project | Type | Reason |
|---------|------|--------|
| **liger** | 95% A | Full autograd support, training ready |
| **vllm** | 76% W | Production-tested, inference optimized |
| **gemlite** | 100% W | Quantized GEMM, drop-in replacement |
| **sageattention** | 100% W | INT8 attention, easy integration |

### For Research/Experimentation:
| Project | Type | Reason |
|---------|------|--------|
| **fla** | 87% A+W | Most complete attention variants |
| **mamba** | 70% A | State space model reference |
| **flash_attn** | 93% W+A | Official flash attention |

### For Multi-Vendor Support:
| Project | Type | Reason |
|---------|------|--------|
| **flaggems** | 63% W | 11 hardware backends |

---

## Sources

- [FlagGems](https://github.com/FlagOpen/FlagGems)
- [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [Unsloth](https://github.com/unslothai/unsloth)
- [mLSTM Kernels](https://github.com/NX-AI/mlstm_kernels)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Mamba](https://github.com/state-spaces/mamba)
- [Conch](https://github.com/stackav-oss/conch)
- [attorch](https://github.com/BobMcDear/attorch)
- [Applied AI](https://github.com/pytorch-labs/applied-ai)
- [Kernl](https://github.com/ELS-RD/kernl)
- [Triton](https://github.com/triton-lang/triton)
- [LLM Note](https://github.com/harleyszhang/llm_note)
- [GemLite](https://github.com/dropbox/gemlite)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [FlagAttention](https://github.com/FlagOpen/FlagAttention)
- [qattn](https://github.com/IBM/qattn)
- [Kernelheim](https://github.com/debashishc/kernelheim)
- [ScatterMoE](https://github.com/shawntan/scattermoe)
