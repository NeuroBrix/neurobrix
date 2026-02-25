# kernels/adapter.py
"""
Kernel Adapter Layer - Enterprise Grade
NeuroBrix - NVIDIA Common - Zero Wrapper Architecture
"""

import torch
import triton
import triton.language as tl
import numpy as np
import json
import zipfile
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from .resolver import get_kernel, get_kernel_module
from .mapping import get_kernel_op_name

# Lazy import to avoid circular dependency
_vendor_config = None

def _get_vendor_config():
    """Lazy load vendor config module."""
    global _vendor_config
    if _vendor_config is None:
        from neurobrix.core.config import get_vendor_config, get_block_sizes
        _vendor_config = {"load": get_vendor_config, "get_blocks": get_block_sizes}
    return _vendor_config


# =============================================================================
# PERFORMANCE CACHES - Critical for runtime speed
# =============================================================================

# Cache: (module_id, kernel_op, op_family) -> kernel_function
# Avoids repeated getattr() searches on every op call
_KERNEL_FUNC_CACHE: Dict[Tuple[int, str, str], Any] = {}


def _get_cached_kernel(kernel_module: Any, op_type: str, op_family: str = "unary") -> Optional[Any]:
    """
    Get kernel function from module with caching.

    Searches for kernel function once, caches result for subsequent calls.
    This eliminates per-op getattr() overhead.

    Args:
        kernel_module: The module containing kernel functions
        op_type: ATen op type (e.g., "aten::gelu")
        op_family: "unary", "binary", "gemm", "reduction", etc.
    """
    k_op = get_kernel_op_name(op_type)
    if k_op:
        k_op = k_op.replace("aten::", "")
    else:
        k_op = op_type.replace("aten::", "")

    cache_key: Tuple[int, str, str] = (id(kernel_module), k_op, op_family)

    if cache_key in _KERNEL_FUNC_CACHE:
        return _KERNEL_FUNC_CACHE[cache_key]

    # Build candidate list based on op family
    if op_family == "binary":
        candidates = [f"_{k_op}_kernel", "_binary_kernel", f"{k_op}_kernel"]
    elif op_family == "gemm":
        candidates = [f"_{k_op}_kernel", "_mm_kernel", "_matmul_kernel", f"{k_op}_kernel"]
    elif op_family == "reduction":
        candidates = [f"_{k_op}_kernel", "_reduce_kernel", f"{k_op}_kernel"]
    else:  # unary default
        candidates = [f"_{k_op}_kernel", f"_{k_op}_impl_kernel", "_unary_kernel", f"{k_op}_kernel"]

    kernel = None
    for name in candidates:
        k = getattr(kernel_module, name, None)
        if k and hasattr(k, "run"):
            kernel = k
            break

    _KERNEL_FUNC_CACHE[cache_key] = kernel
    return kernel


# =============================================================================
# TENSOR HELPERS
# =============================================================================

def allocate_output(shape_or_like: Any, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Universal output tensor allocation - SINGLE POINT OF ALLOCATION.

    Uses empty_like/empty for performance. Kernels MUST write ALL elements.
    This is the kernel contract - any kernel that doesn't write all elements is buggy.

    Args:
        shape_or_like: Either a tensor (uses empty_like) or a shape tuple
        device: Device for allocation (required if shape_or_like is tuple)
        dtype: Dtype for allocation (required if shape_or_like is tuple)

    Returns:
        Uninitialized tensor ready for kernel to fill
    """
    if isinstance(shape_or_like, torch.Tensor):
        return torch.empty_like(shape_or_like)
    else:
        assert device is not None, "device must be provided when shape_or_like is not a tensor"
        assert dtype is not None, "dtype must be provided when shape_or_like is not a tensor"
        return torch.empty(shape_or_like, device=device, dtype=dtype)


def ensure_tensor(x: Any, ref: torch.Tensor) -> torch.Tensor:
    """Convert scalar to tensor using reference tensor's device/dtype."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, device=ref.device, dtype=ref.dtype)


def prepare_tensor(t: Any, device: str) -> Optional[torch.Tensor]:
    """Ensure tensor is on correct device and contiguous. Converts scalars to tensors."""
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        try:
            return torch.tensor(t, device=device)
        except Exception:
             return t

    if str(t.device) != device:
        t = t.to(device)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def get_device_index(device: str) -> int:
    """Extract device index from device string."""
    if device == "cpu":
        return -1
    if ":" in device:
        return int(device.split(":")[1])
    return 0


def extract_attr(attrs: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Extract attribute, handling nested dict format."""
    if key in attrs:
        val = attrs[key]
        if isinstance(val, dict) and "value" in val:
            return val["value"]
        return val

    # Check in args list
    args = attrs.get("args", [])
    for arg in args:
        if isinstance(arg, dict) and arg.get("name") == key:
            return arg.get("value", default)

    return default


# =============================================================================
# OP FAMILY CLASSIFICATION
# =============================================================================

GEMM_OPS = frozenset({
    "aten::linear", "aten::addmm", "aten::matmul", "aten::mm", "aten::bmm",
})

BINARY_OPS = frozenset({
    "aten::add", "aten::add_", "aten::sub", "aten::sub_", "aten::rsub",
    "aten::mul", "aten::mul_", "aten::div", "aten::div_",
    "aten::pow", "aten::remainder", "aten::fmod",
    "aten::minimum", "aten::maximum",
})

COMPARISON_OPS = frozenset({
    "aten::eq", "aten::ne", "aten::lt", "aten::le", "aten::gt", "aten::ge",
    "aten::equal", "aten::not_equal", "aten::less", "aten::less_equal",
    "aten::greater", "aten::greater_equal",
})

UNARY_OPS = frozenset({
    "aten::relu", "aten::relu_", "aten::gelu", "aten::silu", "aten::silu_",
    "aten::sigmoid", "aten::tanh", "aten::exp", "aten::log", "aten::sqrt",
    "aten::neg", "aten::abs", "aten::sin", "aten::cos", "aten::erf",
    "aten::leaky_relu", "aten::elu", "aten::hardswish", "aten::mish",
    "aten::rsqrt",
})

# Ops that REQUIRE float dtype - Triton math ops only work on fp32/fp64
# When input is int, we convert to float, compute, then convert back if needed
FLOAT_REQUIRED_OPS = frozenset({
    "aten::sin", "aten::cos", "aten::tan",
    "aten::exp", "aten::log", "aten::log2", "aten::log10",
    "aten::sqrt", "aten::rsqrt",
    "aten::sigmoid", "aten::tanh",
    "aten::gelu", "aten::silu", "aten::mish",
    "aten::erf", "aten::erfc",
})

REDUCTION_OPS = frozenset({
    "aten::sum", "aten::mean", "aten::max", "aten::min", "aten::prod",
    "aten::amax", "aten::amin",
    "aten::softmax", "aten::_softmax", "aten::log_softmax",
    "aten::argmax", "aten::argmin",
})

NORM_OPS = frozenset({
    "aten::layer_norm", "aten::group_norm", "aten::batch_norm",
    "aten::instance_norm", "aten::rms_norm",
    "aten::native_layer_norm", "aten::native_group_norm",
})

CONV_OPS = frozenset({
    "aten::convolution", "aten::conv2d", "aten::conv1d", "aten::conv3d",
    "aten::conv_transpose2d", "aten::conv_transpose1d", "aten::conv_transpose3d",
})

RESIZE_OPS = frozenset({
    "aten::upsample_nearest2d", "aten::upsample_bilinear2d",
    "aten::resize_nearest_neighbor", "aten::resize_bilinear",
})

SPECIAL_OPS = frozenset({
    "aten::embedding", "aten::gather", "aten::scatter", "aten::index_select",
    "aten::constant_pad_nd", "aten::pad",
    "aten::upsample_nearest2d", "aten::upsample_bilinear2d",
    "aten::where", "aten::clamp", "aten::clamp_min", "aten::clamp_max",
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention_for_cpu",
    "aten::multi_head_attention_forward",
})

FUSED_OPS = frozenset({
    "fused_mul_add", "fused_add_gelu", "fused_add_silu",
    "fused_linear_gelu", "fused_linear_silu", "fused_linear_relu", "fused_linear_bias",
    "fused_layernorm_residual", "fused_rmsnorm_residual",
})


# =============================================================================
# FAMILY-SPECIFIC ADAPTERS
# =============================================================================

class KernelAdapter:
    """
    Universal Adapter Layer - Enterprise Grade.
    Translates ATen convention → Pure kernel calls.

    ZERO HARDCODE: Block sizes come from vendor config, epsilon from graph attrs.
    """

    def __init__(self, family: str, vendor: str, arch: str, device: str, topology: Optional[Dict] = None, dtype: Optional[torch.dtype] = None, graph_dtype: Optional[torch.dtype] = None):
        self.family = family
        self.vendor = vendor
        self.arch = arch
        self.device = device
        self._topology = topology or {}

        # DtypeEngine for Triton branch dtype management
        from neurobrix.core.dtype.engine import DtypeEngine
        self.dtype_engine = DtypeEngine(dtype, graph_dtype=graph_dtype)

        # Load vendor-specific block sizes
        try:
            vc = _get_vendor_config()
            self._block_sizes = vc["get_blocks"](vendor, arch)
        except Exception:
            # Fallback to empty dict if config not found
            self._block_sizes = {}

    def _get_block_size(self, n: int, cap: Optional[int] = None) -> int:
        """
        Heuristic for BLOCK_SIZE: next power of 2, capped.

        ZERO HARDCODE: Cap from vendor config if not explicitly provided.
        """
        if n <= 0:
            return 1

        # Use vendor config default cap if not specified
        final_cap: int
        if cap is None:
            final_cap = self._block_sizes.get("default", 1024)
        else:
            final_cap = cap

        return min(final_cap, 1 << (max(0, n - 1)).bit_length())

    def _get_config_cap(self, key: str, default: int = 1024) -> int:
        """Get specific block size cap from vendor config."""
        return self._block_sizes.get(key, default)

    def _get_epsilon(self, attrs: Dict[str, Any], inputs: List[Any], input_idx: int = 4, component_name: Optional[str] = None) -> float:
        """
        Get epsilon value from graph attrs or topology.

        ZERO HARDCODE: Priority order:
        1. inputs[input_idx] if it's a float/int
        2. attrs["eps"] from graph.json
        3. topology.extracted_values.<component>.layer_norm_epsilon
        4. Raise error (ZERO FALLBACK)

        Note: For now we allow a fallback to 1e-5 for backward compatibility,
        but log a warning. In strict mode this should crash.
        """
        # 1. Check inputs
        if len(inputs) > input_idx and isinstance(inputs[input_idx], (float, int)):
            return float(inputs[input_idx])

        # 2. Check attrs from graph
        eps = attrs.get("eps")
        if eps is not None:
            return float(eps)

        # 3. Check topology extracted_values
        if component_name and self._topology:
            extracted = self._topology.get("extracted_values", {}).get(component_name, {})
            for key in ["layer_norm_epsilon", "layer_norm_eps", "epsilon", "eps"]:
                if key in extracted:
                    return float(extracted[key])

        # 4. Backward compatibility fallback with warning
        # TODO: In strict mode, this should raise RuntimeError
        import sys
        print(f"[WARNING] epsilon not found in attrs, using default 1e-5", file=sys.stderr)
        return 1e-5

    def launch(
        self,
        op_type: str,
        inputs: List[Any],
        attrs: Dict[str, Any],
    ) -> torch.Tensor:
        """Universal entry point."""
        # DtypeEngine: cast inputs based on op classification before kernel launch
        inputs = list(self.dtype_engine.cast_inputs(op_type, tuple(inputs)))  # type: ignore[attr-defined]

        # 0. High priority specialized dispatch
        if "scaled_dot_product" in op_type and "attention" in op_type:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            result = self._launch_sdpa(op_type, kernel_module, inputs, attrs)
            return result[0] if isinstance(result, tuple) else result

        if op_type in FUSED_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            return self._launch_fused(op_type, kernel_module, inputs, attrs)

        if op_type in GEMM_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if op_type == "aten::bmm":
                 return self._launch_bmm(op_type, kernel_module, inputs, attrs)
            return self._launch_gemm(op_type, kernel_module, inputs, attrs)

        if op_type in NORM_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if "layer_norm" in op_type:
                 result = self._launch_layernorm(op_type, kernel_module, inputs, attrs)
                 return result[0] if isinstance(result, tuple) else result
            if "instance_norm" in op_type or "instancenormalization" in op_type:
                 return self._launch_instancenorm(op_type, kernel_module, inputs, attrs)
            result = self._launch_norm(op_type, kernel_module, inputs, attrs)
            return result[0] if isinstance(result, tuple) else result
        
        if op_type in CONV_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            return self._launch_conv(op_type, kernel_module, inputs, attrs)
            
        if op_type in RESIZE_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            return self._launch_resize(op_type, kernel_module, inputs, attrs)
            
        if "softmax" in op_type:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            return self._launch_softmax(op_type, kernel_module, inputs, attrs)

        if op_type in REDUCTION_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if kernel_module:
                 if "arg" in op_type: return self._launch_argmax_argmin(op_type, kernel_module, inputs, attrs)
                 return self._launch_reduction(op_type, kernel_module, inputs, attrs)

        if op_type in UNARY_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if kernel_module:
                 return self._launch_unary(op_type, kernel_module, inputs, attrs)

        if op_type in BINARY_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if kernel_module:
                 return self._launch_binary(op_type, kernel_module, inputs, attrs)

        if op_type in COMPARISON_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if kernel_module:
                 return self._launch_comparison(op_type, kernel_module, inputs, attrs)

        if op_type in SPECIAL_OPS:
            kernel_module = get_kernel_module(op_type, self.family, self.vendor, self.arch)
            if op_type == "aten::embedding":
                 return self._launch_embedding(op_type, kernel_module, inputs, attrs)
            if op_type == "aten::gather":
                 return self._launch_gather(op_type, kernel_module, inputs, attrs)

        raise RuntimeError(f"ZERO FALLBACK: No handler for op '{op_type}'")

    def _launch_unary(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Unary Element-wise Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"

        # [DTYPE CONVERSION] Kernels handle fp16→fp32 internally (register-level, no memory copy)
        # Only need to convert int/bool → float for math ops
        original_dtype = x.dtype
        needs_cast_back = False
        if op_type in FLOAT_REQUIRED_OPS and not x.is_floating_point():
            # int/bool → fp32 (kernels can't handle int for transcendental ops)
            x = x.to(torch.float32)
            needs_cast_back = True

        # Universal allocation - kernel contract: MUST write ALL elements
        out = allocate_output(x)
        n_elements = x.numel()
        if n_elements == 0:
            return out.to(original_dtype) if needs_cast_back else out

        BLOCK_SIZE = self._get_block_size(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

        # Use cached kernel lookup for performance
        kernel = _get_cached_kernel(kernel_module, op_type)
        if not kernel: raise RuntimeError(f"ZERO FALLBACK: No kernel for unary op '{op_type}'")

        extra_kwargs = {}
        if "leaky_relu" in op_type:
             extra_kwargs["negative_slope"] = float(extract_attr(attrs, "negative_slope", 0.01))
        elif "gelu" in op_type: pass
        elif "elu" in op_type:
             extra_kwargs["alpha"] = float(extract_attr(attrs, "alpha", 1.0))

        kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, **extra_kwargs)

        if needs_cast_back:
            out = out.to(original_dtype)

        return out

    def _launch_binary(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Binary Element-wise Launch - OPTIMIZED."""
        inp0, inp1 = inputs[0], inputs[1]

        # Fast path: both are tensors (most common case)
        if isinstance(inp0, torch.Tensor) and isinstance(inp1, torch.Tensor):
            x = prepare_tensor(inp0, self.device)
            y = prepare_tensor(inp1, self.device)
            assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
            assert y is not None and isinstance(y, torch.Tensor), "y must be a tensor"
        elif isinstance(inp0, torch.Tensor):
            x = prepare_tensor(inp0, self.device)
            assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
            y = ensure_tensor(inp1, x)
        elif isinstance(inp1, torch.Tensor):
            y = prepare_tensor(inp1, self.device)
            assert y is not None and isinstance(y, torch.Tensor), "y must be a tensor"
            x = ensure_tensor(inp0, y)
        else:
            x = torch.tensor(inp0, device=self.device, dtype=torch.float32)
            y = torch.tensor(inp1, device=self.device, dtype=torch.float32)

        # Handle rsub: swap operands
        if "rsub" in op_type:
            x, y = y, x

        # Broadcasting
        n_x, n_y = x.numel(), y.numel()
        if n_x != n_y:
            if n_x > n_y and n_y == 1: y = y.expand_as(x).contiguous()
            elif n_y > n_x and n_x == 1: x = x.expand_as(y).contiguous()
            else:
                x, y = torch.broadcast_tensors(x, y)
                x, y = x.contiguous(), y.contiguous()
        else:
            if not x.is_contiguous(): x = x.contiguous()
            if not y.is_contiguous(): y = y.contiguous()

        out = allocate_output(x)
        n_elements = x.numel()
        BLOCK_SIZE = self._get_block_size(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

        kernel = _get_cached_kernel(kernel_module, op_type, "binary")
        if not kernel: raise RuntimeError(f"ZERO FALLBACK: No kernel for binary op '{op_type}'")

        kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out

    def _launch_comparison(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Comparison Launch."""
        x = prepare_tensor(inputs[0], self.device)
        y = prepare_tensor(inputs[1], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        assert y is not None and isinstance(y, torch.Tensor), "y must be a tensor"
        n_x, n_y = x.numel(), y.numel()
        if n_x != n_y:
            if n_x > n_y and n_y == 1: y = y.expand_as(x).contiguous()
            elif n_y > n_x and n_x == 1: x = x.expand_as(y).contiguous()
            else:
                x, y = torch.broadcast_tensors(x, y)
                x, y = x.contiguous(), y.contiguous()
        else:
            if not x.is_contiguous(): x = x.contiguous()
            if not y.is_contiguous(): y = y.contiguous()
        
        out = torch.empty(x.shape, dtype=torch.bool, device=self.device)
        n_elements = x.numel()
        BLOCK_SIZE = self._get_block_size(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
        k_op = get_kernel_op_name(op_type)
        candidates = [f"_{k_op}_kernel", "_binary_kernel", f"{k_op}_kernel"]
        kernel = None
        for name in candidates:
            k = getattr(kernel_module, name, None)
            if k and hasattr(k, "run"):
                kernel = k
                break
        if not kernel: raise RuntimeError(f"ZERO FALLBACK: No kernel for unary op '{op_type}'")
        kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out

    def _launch_gemm(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate GEMM Launch."""
        args, kwargs = self._adapt_gemm(op_type, inputs, attrs)
        a = prepare_tensor(args[0], self.device)
        b = prepare_tensor(args[1], self.device)
        assert a is not None and isinstance(a, torch.Tensor), "a must be a tensor"
        assert b is not None and isinstance(b, torch.Tensor), "b must be a tensor"
        bias = kwargs.get("bias")
        alpha = float(kwargs.get("alpha", 1.0))
        beta = float(kwargs.get("beta", 1.0))

        # [DTYPE HARMONIZATION] Triton tl.dot requires matching dtypes
        # Convert to float if needed - prefer b's dtype (weight tensor)
        if a.dtype != b.dtype:
            target_dtype = b.dtype if b.is_floating_point() else torch.float32
            if not a.is_floating_point():
                a = a.to(target_dtype)
            elif not b.is_floating_point():
                b = b.to(a.dtype)
            else:
                # Both float but different - use higher precision
                a = a.to(target_dtype)

        # Handle Transpose flags from adapter
        trans_a = kwargs.get("trans_a", False)
        trans_b = kwargs.get("trans_b", False)

        # [CRITICAL FIX] Handle multi-dimensional tensors (Batch*Seq, Hidden)
        # If a is 3D [B, S, K], we treat it as [B*S, K] for GEMM
        orig_shape_a = a.shape
        if a.ndim > 2 and not trans_a:
            a = a.reshape(-1, a.shape[-1])
        elif a.ndim > 2 and trans_a:
            # Transposed 3D is rare but possible
            a = a.reshape(a.shape[0], -1)
            
        # Determine logical dimensions M, N, K
        M = a.shape[1] if trans_a else a.shape[0]
        K = a.shape[0] if trans_a else a.shape[1]
        N = b.shape[0] if trans_b else b.shape[1]
        
        # Verify K match
        K_b = b.shape[1] if trans_b else b.shape[0]
        if K != K_b:
            raise RuntimeError(f"K dimension mismatch in {op_type}: {K} vs {K_b} (A: {a.shape}, B: {b.shape}, trans_a={trans_a}, trans_b={trans_b})")
            
        # Determine strides based on transpose
        stride_am, stride_ak = (a.stride(1), a.stride(0)) if trans_a else (a.stride(0), a.stride(1))
        stride_bk, stride_bn = (b.stride(1), b.stride(0)) if trans_b else (b.stride(0), b.stride(1))
        
        # Universal allocation - kernel contract: MUST write ALL elements
        out_flat = allocate_output((M, N), device=self.device, dtype=a.dtype)
        stride_cm, stride_cn = out_flat.stride(0), out_flat.stride(1);
        
        # Prepare Bias
        has_bias = bias is not None
        stride_bias = 0
        if has_bias:
            bias = prepare_tensor(bias, self.device)
            assert bias is not None and isinstance(bias, torch.Tensor), "bias must be a tensor"
            # If beta != 1.0, we pre-scale bias (since kernel does alpha*mm + bias)
            if beta != 1.0:
                bias = bias * beta
            stride_bias = bias.stride(0) if bias.ndim > 0 else 0
        else:
            bias = out_flat # Dummy
            
        # Heuristics for Triton (CRITICAL: tl.dot requires blocks >= 16)
        BLOCK_M = max(16, self._get_block_size(M, cap=128))
        BLOCK_N = max(16, self._get_block_size(N, cap=128))
        BLOCK_K = max(16, self._get_block_size(K, cap=32))
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
        kernel = getattr(kernel_module, "_gemm_kernel")
        
        kernel[grid](
            A=a, B=b, C=out_flat, Bias=bias,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cm=stride_cm, stride_cn=stride_cn,
            stride_bias=stride_bias,
            alpha=alpha, beta=beta,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8
        )
        
        # Restore original shape for output
        if len(orig_shape_a) > 2 and not trans_a:
            new_shape = list(orig_shape_a[:-1]) + [N]
            out = out_flat.view(new_shape)
        else:
            out = out_flat

        # DEBUG: Check for inf/nan in GEMM output
        if os.environ.get("NBX_DEBUG_MM") == "1":
            has_inf = torch.isinf(out).any().item()
            has_nan = torch.isnan(out).any().item()
            if has_inf or has_nan:
                print(f"[DEBUG MM ANOMALY] {op_type}")
                print(f"  A: shape={a.shape}, dtype={a.dtype}, mean={a.float().mean():.4f}, range=[{a.min():.4f}, {a.max():.4f}]")
                print(f"  B: shape={b.shape}, dtype={b.dtype}, mean={b.float().mean():.4f}, range=[{b.min():.4f}, {b.max():.4f}]")
                print(f"  Out: shape={out.shape}, has_inf={has_inf}, has_nan={has_nan}")
                print(f"  Out range: [{out[~torch.isinf(out) & ~torch.isnan(out)].min():.4f}, {out[~torch.isinf(out) & ~torch.isnan(out)].max():.4f}]")

        if os.environ.get("NBX_DEBUG_ATTN") == "1":
            def safe_stats(t):
                if not isinstance(t, torch.Tensor): return 0.0, 0.0
                tf = t.float()
                return tf.mean().item(), (tf.std().item() if tf.numel() > 1 else 0.0)

            ma, sa = safe_stats(a)
            mb, sb = safe_stats(b)
            mo, so = safe_stats(out)
            print(f"[DEBUG GEMM] {op_type} | A: shape={a.shape}, mean={ma:.4f}, std={sa:.4f} | B: shape={b.shape}, mean={mb:.4f}, std={sb:.4f} | Out: shape={out.shape}, mean={mo:.4f}, std={so:.4f}")

        return out

    def _launch_fused(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Fused Operations Launch."""
        import triton
        if "linear" in op_type:
            res = self._launch_gemm("aten::linear", get_kernel_module("aten::linear", self.family, self.vendor, self.arch), inputs, attrs)
            if "gelu" in op_type: return self._launch_unary("aten::gelu", get_kernel_module("aten::gelu", self.family, self.vendor, self.arch), [res], attrs)
            if "silu" in op_type: return self._launch_unary("aten::silu", get_kernel_module("aten::silu", self.family, self.vendor, self.arch), [res], attrs)
            if "relu" in op_type: return self._launch_unary("aten::relu", get_kernel_module("aten::relu", self.family, self.vendor, self.arch), [res], attrs)
            return res
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        out = torch.empty_like(x); n_el = x.numel(); BLOCK_SIZE = self._get_block_size(n_el)
        kernel = getattr(kernel_module, f"_{op_type}_kernel", None)
        if not kernel: raise RuntimeError(f"ZERO FALLBACK: No kernel for unary op '{op_type}'")
        ptrs = [prepare_tensor(t, self.device) for t in inputs]
        kernel[(triton.cdiv(n_el, BLOCK_SIZE),)](*ptrs, out, n_el, BLOCK_SIZE=BLOCK_SIZE)
        return out

    def _launch_bmm(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate BMM Launch."""
        a = prepare_tensor(inputs[0], self.device)
        b = prepare_tensor(inputs[1], self.device)
        assert a is not None and isinstance(a, torch.Tensor), "a must be a tensor"
        assert b is not None and isinstance(b, torch.Tensor), "b must be a tensor"
        B, M, K = a.shape
        B2, K2, N = b.shape
        out = torch.empty((B, M, N), dtype=a.dtype, device=self.device)
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, B)
        kernel = getattr(kernel_module, "_bmm_kernel", None)
        assert kernel is not None, "BMM kernel not found"
        kernel[grid](a_ptr=a, b_ptr=b, c_ptr=out, B=B, M=M, N=N, K=K, stride_ab=a.stride(0), stride_am=a.stride(1), stride_ak=a.stride(2), stride_bb=b.stride(0), stride_bk=b.stride(1), stride_bn=b.stride(2), stride_cb=out.stride(0), stride_cm=out.stride(1), stride_cn=out.stride(2), BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=32, num_warps=4)
        return out

    def _launch_norm(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Orchestrate Normalization Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        if "group_norm" in op_type:
            # GroupNorm Signature: (input, num_groups, weight, bias, eps, ...)
            # native_group_norm signature can vary, but typically:
            # (input, weight, bias, N, C, H, W, group, eps)
            
            if "native" in op_type:
                # aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int H*W, int group, float eps)
                weight = prepare_tensor(inputs[1], self.device) if len(inputs) > 1 and isinstance(inputs[1], torch.Tensor) else None
                bias = prepare_tensor(inputs[2], self.device) if len(inputs) > 2 and isinstance(inputs[2], torch.Tensor) else None
                num_groups = int(inputs[6]) if len(inputs) > 6 else extract_attr(attrs, "num_groups", 32)
                # ZERO HARDCODE: epsilon from graph attrs or topology
                eps = self._get_epsilon(attrs, inputs, input_idx=7)
            else:
                # aten::group_norm
                num_groups = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], int) else extract_attr(attrs, "num_groups", 32)
                weight = prepare_tensor(inputs[2], self.device) if len(inputs) > 2 and isinstance(inputs[2], torch.Tensor) else None
                bias = prepare_tensor(inputs[3], self.device) if len(inputs) > 3 and isinstance(inputs[3], torch.Tensor) else None
                # ZERO HARDCODE: epsilon from graph attrs or topology
                eps = self._get_epsilon(attrs, inputs, input_idx=4)

            N, C = x.shape[0], x.shape[1]
            x_reshaped = x.view(N, num_groups, -1)
            hidden_size = x_reshaped.shape[-1]
            channels_per_group = C // num_groups
            out = torch.empty_like(x_reshaped)
            mean, rstd = torch.empty((N, num_groups), dtype=torch.float32, device=self.device), torch.empty((N, num_groups), dtype=torch.float32, device=self.device)
            # ZERO HARDCODE: cap from vendor config
            groupnorm_cap = self._get_config_cap("groupnorm_cap", 65536)
            BLOCK_SIZE = self._get_block_size(hidden_size, cap=groupnorm_cap)
            getattr(kernel_module, "_group_norm_forward_kernel")[(N, num_groups)](Y_ptr=out, Y_row_stride=out.stride(0), Y_col_stride=out.stride(1), X_ptr=x_reshaped, X_row_stride=x_reshaped.stride(0), X_col_stride=x_reshaped.stride(1), Mean_ptr=mean, Mean_row_stride=mean.stride(0), Mean_col_stride=mean.stride(1), RSTD_ptr=rstd, RSTD_row_stride=rstd.stride(0), RSTD_col_stride=rstd.stride(1), W_ptr=weight, B_ptr=bias, hidden_size=hidden_size, channels_per_group=channels_per_group, eps=eps, BLOCK_SIZE=BLOCK_SIZE)
            
            res = out.view_as(x)
            if "native" in op_type:
                dummy = torch.zeros(1, device=self.device, dtype=torch.float32)
                return res, dummy, dummy
            return res
        raise RuntimeError(f"ZERO FALLBACK: No kernel for norm op '{op_type}'")

    def _launch_layernorm(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Orchestrate LayerNorm Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"

        # Signature for both layer_norm and native_layer_norm:
        # (input, normalized_shape, weight, bias, eps, ...)
        # normalized_shape is inputs[1]
        weight = None
        bias = None
        if len(inputs) > 2 and isinstance(inputs[2], torch.Tensor):
            weight = prepare_tensor(inputs[2], self.device)
        if len(inputs) > 3 and isinstance(inputs[3], torch.Tensor):
            bias = prepare_tensor(inputs[3], self.device)
            
        # ZERO HARDCODE: epsilon from graph attrs or topology
        eps = self._get_epsilon(attrs, inputs, input_idx=4)

        normalized_shape = extract_attr(attrs, "normalized_shape")
        if normalized_shape is None and len(inputs) > 1:
            normalized_shape = inputs[1]
            
        if normalized_shape is None:
            normalized_shape = [x.shape[-1]]
            
        N = normalized_shape[-1]
        M = x.numel() // N
        x_2d = x.reshape(M, N)
        y = torch.empty_like(x_2d)
        
        if N <= 4096:
            TILE_N = 1 << (max(0, N - 1)).bit_length()
            getattr(kernel_module, "layer_norm_persistent_kernel")[(M,)](in_ptr=x_2d, out_ptr=y, weight_ptr=weight, bias_ptr=bias, out_mean_ptr=None, out_rstd_ptr=None, M=M, N=N, eps=eps, TILE_N=TILE_N)
        else:
            # ZERO HARDCODE: tile size from vendor config
            layernorm_tile = self._get_config_cap("layernorm_tile", 1024)
            getattr(kernel_module, "layer_norm_loop_kernel")[(M,)](in_ptr=x_2d, out_ptr=y, weight_ptr=weight, bias_ptr=bias, out_mean_ptr=None, out_rstd_ptr=None, M=M, N=N, eps=eps, TILE_N=layernorm_tile)
            
        res = y.view_as(x)
        
        if os.environ.get("NBX_DEBUG_ATTN") == "1":
            print(f"[DEBUG LN] In: mean={x.mean():.4f}, std={x.std():.4f} | Out: mean={res.mean():.4f}, std={res.std():.4f}")

        if "native" in op_type:
            # Return (output, mean, rstd)
            # We don't need mean/rstd for inference, but must satisfy signature
            dummy = torch.zeros(1, device=self.device, dtype=torch.float32)
            return res, dummy, dummy
            
        return res

    def _launch_instancenorm(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate InstanceNorm Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        weight, bias = prepare_tensor(inputs[1], self.device) if len(inputs) > 1 else None, prepare_tensor(inputs[2], self.device) if len(inputs) > 2 else None
        # ZERO HARDCODE: epsilon from graph attrs or topology
        eps = self._get_epsilon(attrs, inputs)
        N, C = x.shape[0], x.shape[1]
        HW = x.numel() // (N * C)
        out = torch.empty_like(x)
        BLOCK_SIZE = self._get_block_size(HW)
        getattr(kernel_module, "_instancenorm_kernel")[(N * C,)](x_ptr=x, out_ptr=out, gamma_ptr=weight, beta_ptr=bias, N=N, C=C, HW=HW, eps=eps, stride_n=x.stride(0), stride_c=x.stride(1), stride_hw=1, BLOCK_SIZE=BLOCK_SIZE)
        return out

    def _launch_conv(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Convolution Launch."""
        x = prepare_tensor(inputs[0], self.device)
        weight = prepare_tensor(inputs[1], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        assert weight is not None and isinstance(weight, torch.Tensor), "weight must be a tensor"
        bias = prepare_tensor(inputs[2], self.device) if len(inputs) > 2 else None
        stride, padding, dilation, groups = extract_attr(attrs, "stride", (1,1)), extract_attr(attrs, "padding", (0,0)), extract_attr(attrs, "dilation", (1,1)), extract_attr(attrs, "groups", 1)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        N, C_in, H_in, W_in = x.shape
        C_out, _, K_H, K_W = weight.shape
        H_out = (H_in + 2 * padding[0] - dilation[0] * (K_H - 1) - 1) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - dilation[1] * (K_W - 1) - 1) // stride[1] + 1
        kernel = getattr(kernel_module, "_conv2d_direct_kernel")
        BLOCK_H, BLOCK_W = 16, 16
        if groups > 1:
            C_in_g, C_out_g, outputs = C_in // groups, C_out // groups, []
            x_gs, w_gs, b_gs = torch.chunk(x, groups, dim=1), torch.chunk(weight, groups, dim=0), torch.chunk(bias, groups, dim=0) if bias is not None else [None]*groups
            for x_g, w_g, b_g in zip(x_gs, w_gs, b_gs):
                out_g = torch.empty((N, C_out_g, H_out, W_out), dtype=x.dtype, device=self.device)
                kernel[(C_out_g, N, triton.cdiv(H_out, BLOCK_H) * triton.cdiv(W_out, BLOCK_W))](x_ptr=x_g, w_ptr=w_g, out_ptr=out_g, bias_ptr=b_g, N=N, C_in=C_in_g, H_in=H_in, W_in=W_in, C_out=C_out_g, K_H=K_H, K_W=K_W, H_out=H_out, W_out=W_out, stride_h=stride[0], stride_w=stride[1], pad_h=padding[0], pad_w=padding[1], dil_h=dilation[0], dil_w=dilation[1], stride_xn=x_g.stride(0), stride_xc=x_g.stride(1), stride_xh=x_g.stride(2), stride_xw=x_g.stride(3), stride_wco=w_g.stride(0), stride_wci=w_g.stride(1), stride_wkh=w_g.stride(2), stride_wkw=w_g.stride(3), stride_on=out_g.stride(0), stride_oc=out_g.stride(1), stride_oh=out_g.stride(2), stride_ow=out_g.stride(3), BLOCK_N=1, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W)
                outputs.append(out_g)
            res = torch.cat(outputs, dim=1)
        else:
            out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=self.device)
            kernel[(C_out, N, triton.cdiv(H_out, BLOCK_H) * triton.cdiv(W_out, BLOCK_W))](x_ptr=x, w_ptr=weight, out_ptr=out, bias_ptr=bias, N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out, K_H=K_H, K_W=K_W, H_out=H_out, W_out=W_out, stride_h=stride[0], stride_w=stride[1], pad_h=padding[0], pad_w=padding[1], dil_h=dilation[0], dil_w=dilation[1], stride_xn=x.stride(0), stride_xc=x.stride(1), stride_xh=x.stride(2), stride_xw=x.stride(3), stride_wco=weight.stride(0), stride_wci=weight.stride(1), stride_wkh=weight.stride(2), stride_wkw=weight.stride(3), stride_on=out.stride(0), stride_oc=out.stride(1), stride_oh=out.stride(2), stride_ow=out.stride(3), BLOCK_N=1, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W)
            res = out
            
        if os.environ.get("NBX_DEBUG_ATTN") == "1":
            print(f"[DEBUG CONV] In: mean={x.mean():.4f}, std={x.std():.4f} | Out: mean={res.mean():.4f}, std={res.std():.4f}")
            
        return res

    def _launch_resize(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Resize Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        mode = "bilinear" if "bilinear" in op_type else "nearest"
        out_sz = extract_attr(attrs, "output_size") or extract_attr(attrs, "size") or (inputs[1] if len(inputs) > 1 else None)
        if isinstance(out_sz, torch.Tensor):
            out_sz = out_sz.cpu().tolist()
        assert out_sz is not None, "output_size must be provided"
        H_out, W_out = int(out_sz[0]), int(out_sz[1])
        N, C, H_in, W_in = x.shape
        out = torch.empty(N, C, H_out, W_out, dtype=x.dtype, device=self.device)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(H_out * W_out, BLOCK_SIZE), N, C)
        kernel = getattr(kernel_module, f"_resize_{mode}_kernel")
        kernel[grid](x_ptr=x, out_ptr=out, N=N, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out, scale_h=float(H_out)/H_in, scale_w=float(W_out)/W_in, stride_xn=x.stride(0), stride_xc=x.stride(1), stride_xh=x.stride(2), stride_xw=x.stride(3), stride_on=out.stride(0), stride_oc=out.stride(1), stride_oh=out.stride(2), stride_ow=out.stride(3), BLOCK_SIZE=BLOCK_SIZE)
        return out

    def _launch_sdpa(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Orchestrate SDPA Launch."""
        q = prepare_tensor(inputs[0], self.device)
        k = prepare_tensor(inputs[1], self.device)
        v = prepare_tensor(inputs[2], self.device)
        assert q is not None and isinstance(q, torch.Tensor), "q must be a tensor"
        assert k is not None and isinstance(k, torch.Tensor), "k must be a tensor"
        assert v is not None and isinstance(v, torch.Tensor), "v must be a tensor"
        
        # Check inputs for mask, dropout, causal, scale
        # Signature: q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        
        # 1. Extract Mask (Positional or Keyword)
        attn_mask = None
        if len(inputs) > 3 and inputs[3] is not None:
            attn_mask = inputs[3]
        elif "kwargs" in attrs and "attn_mask" in attrs["kwargs"]:
            attn_mask = attrs["kwargs"]["attn_mask"]
            
        if attn_mask is not None:
            attn_mask = prepare_tensor(attn_mask, self.device)
            
        # 2. Extract Causal
        is_causal = False
        if len(inputs) > 5:
            is_causal = inputs[5]
        elif "kwargs" in attrs and "is_causal" in attrs["kwargs"]:
            is_causal = attrs["kwargs"]["is_causal"]
            
        # 3. Extract Scale
        scale_arg = None
        if len(inputs) > 6:
            scale_arg = inputs[6]
        elif "kwargs" in attrs and "scale" in attrs["kwargs"]:
            scale_arg = attrs["kwargs"]["scale"]
            
        scale = extract_attr(attrs, "scale") 
        if scale is None and scale_arg is not None:
             scale = scale_arg
        if scale is None:
             scale = q.shape[-1] ** -0.5
             
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        batch, heads, seq_q, head_dim = q.shape
        blk_d = 1 << (max(0, head_dim - 1)).bit_length()
        if blk_d != head_dim:
            pad = blk_d - head_dim
            q, k, v = torch.nn.functional.pad(q, (0, pad)), torch.nn.functional.pad(k, (0, pad)), torch.nn.functional.pad(v, (0, pad))
            
        out = torch.empty(batch, heads, seq_q, blk_d, device=self.device, dtype=q.dtype)
        lse = torch.empty((batch, heads, seq_q), device=self.device, dtype=torch.float32)
        
        # Adaptive block sizes
        if blk_d >= 256:
            blk_m, blk_n = 16, 16
        elif blk_d >= 128:
            blk_m, blk_n = 32, 32
        else:
            blk_m, blk_n = 64, 64
            
        grid = (triton.cdiv(seq_q, blk_m), batch * heads)
        kernel = getattr(kernel_module, "_attn_fwd_kernel", None) if kernel_module else None

        # [FALLBACK] If no Triton kernel, use PyTorch's native SDPA (cuDNN/Flash Attention)
        if not kernel:
            import torch.nn.functional as F

            # Make inputs contiguous for SDPA
            q_orig = q[:, :, :, :head_dim].contiguous()
            k_orig = k[:, :, :, :head_dim].contiguous()
            v_orig = v[:, :, :, :head_dim].contiguous()

            # Convert attention mask to additive float mask (more reliable than bool)
            # PyTorch SDPA additive mask: 0 = attend, -inf = mask
            mask_for_sdpa = None
            if attn_mask is not None:
                attn_mask = attn_mask.contiguous()

                if attn_mask.dtype == torch.int64 or attn_mask.dtype == torch.int32:
                    # Check if already additive format (values like -10000, 0)
                    mask_min = attn_mask.min().item()
                    mask_max = attn_mask.max().item()

                    if mask_min < -1:
                        # Already additive format: 0 = attend, large_negative = mask
                        # Just cast to float (SDPA handles large negative as -inf)
                        mask_for_sdpa = attn_mask.to(q_orig.dtype)
                    else:
                        # Binary mask: 1 = attend, 0 = mask → additive: 0 = attend, -inf = mask
                        mask_for_sdpa = torch.zeros_like(attn_mask, dtype=q_orig.dtype)
                        mask_for_sdpa.masked_fill_(attn_mask == 0, float("-inf"))
                elif attn_mask.dtype == torch.bool:
                    # Bool mask: True = attend → additive: 0 = attend, -inf = mask
                    mask_for_sdpa = torch.zeros(attn_mask.shape, dtype=q_orig.dtype, device=attn_mask.device)
                    mask_for_sdpa.masked_fill_(~attn_mask, float("-inf"))
                else:
                    # Already float - assume it's additive format
                    mask_for_sdpa = attn_mask.to(q_orig.dtype)

                # Handle mask broadcasting for cross-attention [B, H, Q, K]
                # If mask is [B, 1, 1, K] or [B, H, 1, K], expand to match Q dimension
                if mask_for_sdpa.ndim == 4 and mask_for_sdpa.shape[2] == 1:
                    # Broadcast Q dimension
                    mask_for_sdpa = mask_for_sdpa.expand(-1, -1, q_orig.shape[2], -1)
                elif mask_for_sdpa.ndim == 4 and mask_for_sdpa.shape[1] == 1:
                    # Broadcast heads dimension
                    mask_for_sdpa = mask_for_sdpa.expand(-1, q_orig.shape[1], -1, -1)

            # PyTorch SDPA is highly optimized - cuDNN Flash Attention backend
            output = F.scaled_dot_product_attention(
                q_orig, k_orig, v_orig,
                attn_mask=mask_for_sdpa,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=scale
            )
            # Return format depends on op_type
            if "efficient" in op_type or "flash" in op_type:
                # These variants return (output, logsumexp, philox_seed, philox_offset)
                lse = torch.zeros((batch, heads, seq_q), device=self.device, dtype=torch.float32)
                philox_seed = torch.tensor(0, device=self.device, dtype=torch.int64)
                philox_offset = torch.tensor(0, device=self.device, dtype=torch.int64)
                return output, lse, philox_seed, philox_offset
            return output
             
        # Prepare mask strides and pointers
        if attn_mask is not None:
            # Convert non-floating mask to additive -inf/0 mask
            if not attn_mask.is_floating_point():
                # Convention: non-zero (True/1) is KEEP, zero (False/0) is MASK
                mask_f = torch.zeros_like(attn_mask, dtype=q.dtype)
                mask_f.masked_fill_(attn_mask == 0, float("-inf"))
                attn_mask = mask_f
            
            # Ensure mask is at least 4D for stride logic
            while attn_mask.ndim < 4:
                attn_mask = attn_mask.unsqueeze(0)
            
            # [CRITICAL] Handle broadcasting for Triton kernel
            # Expand to full 4D shape [B, H, Q, K] so broadcasted dimensions have 0 stride
            try:
                attn_mask = attn_mask.expand(batch, heads, seq_q, k.shape[2])
            except Exception:
                # Fallback: if expansion fails, try to broadcast only what we can
                pass
                
            mask_strides = (
                attn_mask.stride(0), attn_mask.stride(1), 
                attn_mask.stride(2), attn_mask.stride(3)
            )
            has_mask = True
        else:
            mask_strides = (0, 0, 0, 0)
            attn_mask = q # Dummy
            has_mask = False

        # DEBUG: Inspect inputs (after conversion)
        if os.environ.get("NBX_DEBUG_ATTN") == "1":
            print(f"[DEBUG SDPA] Q: shape={q.shape}, mean={q.mean():.4f}, std={q.std():.4f}, range=[{q.min():.4f}, {q.max():.4f}]")
            print(f"[DEBUG SDPA] K: shape={k.shape}, mean={k.mean():.4f}, std={k.std():.4f}, range=[{k.min():.4f}, {k.max():.4f}]")
            if has_mask:
                print(f"[DEBUG SDPA] Mask (Float): shape={attn_mask.shape}, mean={attn_mask.mean():.4f}, range=[{attn_mask.min():.4f}, {attn_mask.max():.4f}]")
            print(f"[DEBUG SDPA] sm_scale: {scale:.6f}")

        kernel[grid](
            Q=q, K=k, V=v, attn_mask=attn_mask, sm_scale=scale, LSE=lse, Out=out,
            stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qk=q.stride(3),
            stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
            stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
            stride_mask_b=mask_strides[0], stride_mask_h=mask_strides[1], 
            stride_mask_m=mask_strides[2], stride_mask_n=mask_strides[3],
            stride_ob=out.stride(0), stride_oh=out.stride(1), stride_om=out.stride(2), stride_ok=out.stride(3),
            batch=batch, heads=heads, seq_q=seq_q, seq_k=k.shape[2],
            BLOCK_M=blk_m, BLOCK_N=blk_n, BLOCK_DMODEL=blk_d,
            HAS_ATTN_MASK=has_mask
        )
        
        if blk_d != head_dim:
            out = out[:, :, :, :head_dim]
            
        return out

    def _launch_softmax(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Softmax Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        dim = extract_attr(attrs, "dim", -1)
        ndim = x.ndim
        if dim < 0: dim += ndim
        orig_shape, need_unperm = list(x.shape), dim != ndim - 1
        if need_unperm:
            perm = list(range(ndim)); perm[dim], perm[-1] = perm[-1], perm[dim]
            x = x.permute(perm).contiguous()
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out_flat = torch.empty_like(x.view(n_rows, n_cols))
        blk_size = self._get_block_size(n_cols, cap=8192)
        getattr(kernel_module, "_softmax_kernel")[(n_rows,)](output_ptr=out_flat, input_ptr=x.view(n_rows, n_cols), input_row_stride=x.stride(-2), output_row_stride=out_flat.stride(0), n_cols=n_cols, BLOCK_SIZE=blk_size, num_warps=4)
        
        if os.environ.get("NBX_DEBUG_ATTN") == "1":
            print(f"[DEBUG SOFTMAX] In: range=[{x.min():.4f}, {x.max():.4f}] | Out: range=[{out_flat.min():.4f}, {out_flat.max():.4f}]")

        out = out_flat.view(x.shape)
        if need_unperm:
            perm = list(range(ndim)); perm[dim], perm[-1] = perm[-1], perm[dim]
            out = out.permute(perm).contiguous()
        return out.view(orig_shape)

    def _launch_reduction(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Reduction Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        dim, keepdim = extract_attr(attrs, "dim"), extract_attr(attrs, "keepdim", False)
        if dim is None:
            out = torch.zeros(1, dtype=x.dtype, device=self.device)
            kernel = getattr(kernel_module, f"_{op_type.split('::')[-1]}_1d_kernel", None) or getattr(kernel_module, "_reduce_sum_1d_kernel", None)
            if kernel:
                blk_size = 1024
                kernel[(triton.cdiv(x.numel(), blk_size),)](x, out, x.numel(), BLOCK_SIZE=blk_size, num_warps=4)
                return out.view([1]*x.ndim) if keepdim else out.squeeze()
            dim = list(range(x.ndim))
        if isinstance(dim, int): dim = [dim]
        dim = sorted([(d if d >= 0 else x.ndim + d) for d in dim])
        if not all(dim[i] + 1 == dim[i+1] for i in range(len(dim)-1)) and len(dim) > 1:
            res = x
            for d in reversed(dim): res = self._launch_reduction(op_type, kernel_module, [res], {"dim": d, "keepdim": keepdim})
            return res
        o_s, r_s, i_s = self._compute_reduce_dims(list(x.shape), dim)
        out_sh = [s for i, s in enumerate(x.shape) if i not in dim] or [1]
        if keepdim:
            out_sh = list(x.shape); 
            for d in dim: out_sh[d] = 1
        out = torch.zeros(out_sh, dtype=x.dtype, device=self.device)
        blk_size = self._get_block_size(r_s)
        op_n = op_type.split("::")[-1]
        kernel = getattr(kernel_module, f"_{op_n}_axis_kernel", None) or getattr(kernel_module, "_reduce_sum_axis_kernel", None)
        if not kernel: raise RuntimeError(f"ZERO FALLBACK: No kernel for unary op '{op_type}'")
        kernel[(o_s * i_s,)](in_ptr=x, out_ptr=out, outer_size=o_s, reduce_size=r_s, inner_size=i_s, BLOCK_SIZE=blk_size, num_warps=4)
        return out

    def _compute_reduce_dims(self, shape: List[int], dim: List[int]):
        ndim = len(shape)
        dim = sorted([(d if d >= 0 else ndim + d) for d in dim])
        o_s, r_s, i_s = 1, 1, 1
        for i, s in enumerate(shape):
            if i < dim[0]: o_s *= s
            elif i <= dim[-1]: r_s *= s
            else: i_s *= s
        return o_s, r_s, i_s

    def _launch_argmax_argmin(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate ArgMax/ArgMin Launch."""
        x = prepare_tensor(inputs[0], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        dim, keepdim = extract_attr(attrs, "dim", None), extract_attr(attrs, "keepdim", False)
        if dim is None: x = x.flatten().unsqueeze(0); dim = 1
        ndim = x.ndim
        if dim < 0: dim += ndim
        orig_shape = list(x.shape)
        if dim != ndim - 1:
            perm = list(range(ndim)); perm[dim], perm[-1] = perm[-1], perm[dim]
            x = x.permute(perm).contiguous()
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty(n_rows, dtype=torch.int64, device=self.device)
        blk_size = self._get_block_size(n_cols)
        op_name = op_type.split("::")[-1]
        kernel = getattr(kernel_module, f"_{op_name}_last_dim_kernel", None)
        if not kernel: raise RuntimeError(f"ZERO FALLBACK: No kernel for unary op '{op_type}'")
        kernel[(n_rows,)](x_ptr=x, out_ptr=out, n_rows=n_rows, n_cols=n_cols, stride_row=x.stride(-2), BLOCK_SIZE=blk_size)
        out_shape = [s for i, s in enumerate(orig_shape) if i != dim] or [1]
        if keepdim: out_shape = list(orig_shape); out_shape[dim] = 1
        return out.view(out_shape)

    def _launch_embedding(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Embedding Launch."""
        weight = prepare_tensor(inputs[0], self.device)
        indices = prepare_tensor(inputs[1], self.device)
        assert weight is not None and isinstance(weight, torch.Tensor), "weight must be a tensor"
        assert indices is not None and isinstance(indices, torch.Tensor), "indices must be a tensor"
        indices_flat = indices.flatten()
        emb_d = weight.shape[1]
        out_flat = torch.empty((indices_flat.numel(), emb_d), dtype=weight.dtype, device=self.device)
        blk_size = self._get_block_size(emb_d)
        getattr(kernel_module, "_embedding_kernel")[(indices_flat.numel(),)](indices_ptr=indices_flat, weight_ptr=weight, out_ptr=out_flat, num_indices=indices_flat.numel(), embedding_dim=emb_d, stride_w0=weight.stride(0), stride_w1=weight.stride(1), BLOCK_SIZE=blk_size, num_warps=4)
        return out_flat.view(list(indices.shape) + [emb_d])

    def _launch_gather(self, op_type: str, kernel_module: Any, inputs: List[Any], attrs: Dict[str, Any]) -> torch.Tensor:
        """Orchestrate Gather Launch."""
        x = prepare_tensor(inputs[0], self.device)
        indices = prepare_tensor(inputs[1], self.device)
        assert x is not None and isinstance(x, torch.Tensor), "x must be a tensor"
        assert indices is not None and isinstance(indices, torch.Tensor), "indices must be a tensor"
        dim = extract_attr(attrs, "dim", 0)
        ndim = x.ndim
        if dim < 0: dim += ndim
        if dim == 0 and ndim == 2:
            n_idx, emb_d = indices.numel(), x.shape[1]
            out_f = torch.empty((n_idx, emb_d), dtype=x.dtype, device=self.device)
            blk_size = self._get_block_size(emb_d)
            getattr(kernel_module, "_gather_nd_kernel")[(n_idx,)](x_ptr=x, indices_ptr=indices.flatten(), out_ptr=out_f, n_indices=n_idx, index_dim=emb_d, stride_x=x.stride(0), stride_out=out_f.stride(0), BLOCK_SIZE=blk_size, num_warps=4)
            return out_f.view(list(indices.shape) + [emb_d])
        raise RuntimeError(f"ZERO FALLBACK: Gather op '{op_type}' requires dim=0, ndim=2 for Triton kernel")

    def _adapt_gemm(self, op_type: str, inputs: List[Any], attrs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        if op_type == "aten::linear":
            x, weight, bias = inputs[0], inputs[1], inputs[2] if len(inputs) > 2 and isinstance(inputs[2], torch.Tensor) else None
            return [x, weight], {"trans_a": False, "trans_b": True, "alpha": 1.0, "beta": 1.0 if bias is not None else 0.0, "bias": bias}
        if op_type == "aten::addmm":
            bias, mat1, mat2 = inputs[0] if isinstance(inputs[0], torch.Tensor) else None, inputs[1], inputs[2]
            return [mat1, mat2], {"trans_a": False, "trans_b": False, "alpha": extract_attr(attrs, "alpha", 1.0), "beta": extract_attr(attrs, "beta", 1.0), "bias": bias}
        if op_type in ("aten::matmul", "aten::mm"): return [inputs[0], inputs[1]], {"trans_a": False, "trans_b": False, "alpha": 1.0, "beta": 0.0, "bias": None}
        if op_type == "aten::bmm": return [inputs[0], inputs[1]], {}
        raise RuntimeError(f"Unknown GEMM op: {op_type}")