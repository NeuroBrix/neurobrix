"""
NeuroBrix Kernel System.

Architecture:
- dispatch.py: aten_op → Triton kernel wrapper (--triton mode)
- wrappers.py: PyTorch wrappers for Triton kernels
- ops/: Pure @triton.jit kernels (ZERO import torch)
- classification.py: Op classification (TRITON vs METADATA)
- metadata_ops.py: Shape/view ops (PyTorch native)

Usage:
    from neurobrix.kernels.dispatch import dispatch
    kernel = dispatch("aten::relu")  # Returns Triton wrapper or None
"""

from .classification import OpExecution, get_execution_type, ATEN_CLASSIFICATION

# `metadata_ops` is the COMPILED-mode metadata op set and imports torch at
# module level. Re-exporting it eagerly here pulled torch into every process
# that imported ANY kernel — `ops/_configs` alone, and therefore rmsnorm,
# softmax, conv2d, matmul, baddbmm, depthwise_conv2d — i.e. into the branch
# that exists to contain no torch at all. It was the widest torch surface
# under kernels/ (R33 inventory, 2026-09-03), and the activation proof found
# it: removing torch from _configs.py alone changed nothing, because the
# package __init__ was pulling it anyway.
#
# The real consumer (`core/runtime/graph_executor`) imports it from the
# module directly, so the re-export is kept only for API compatibility and
# is resolved on first access (PEP 562).
def __getattr__(name):
    if name == "execute_metadata_op":
        from .metadata_ops import execute_metadata_op
        return execute_metadata_op
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



__all__ = [
    # Classification
    "OpExecution",
    "get_execution_type",
    "ATEN_CLASSIFICATION",
    # Metadata ops
    "execute_metadata_op",
]

# The NeuroBrix launcher seam is installed HERE, in the package every kernel
# module imports before it can launch anything — not in dispatch.py, which a
# weight load (`bf16_to_fp16_kernel`) or a dtype cast (`copy_kernel`) reaches
# long before the first dispatch. Installed there, those launches went through
# upstream's launcher and its torch-importing driver probe (whole-zoo probe
# 2026-09-05: Sana, VibeVoice, Voxtral, canary, chatterbox).
from .launcher import install as _install_launcher  # noqa: E402

_install_launcher()
