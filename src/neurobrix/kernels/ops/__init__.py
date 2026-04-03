"""NeuroBrix Triton Kernels — Pure @triton.jit, One File Per Op.

Each file contains the @triton.jit kernel only (pure Triton compute).
ALL PyTorch wrappers are in kernels/wrappers.py.
Routing is in kernels/dispatch.py.

NO import torch in this package.
"""
