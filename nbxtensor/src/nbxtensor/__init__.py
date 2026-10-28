"""nbxtensor — an inference-first tensor library, torch-free.

The tensor library of NeuroBrix, built to stand on its own: dtypes, shapes,
strides and views, memory that is planned rather than discovered, device
backends (CUDA, Apple Metal, CPU) and Triton kernels that declare their own
cost. NeuroBrix's Triton engine is its first client; it imports this package
and never re-implements what lives here.

Status: M0 — the package is the boundary. The code moves in during M1 (see
docs/nbxtensor/MIGRATION.md in the NeuroBrix repository). Until then nothing
is exported, on purpose: an export appears when its module has moved in with
its tests and its proof of identical outputs.
"""

__version__ = "0.0.0"
__all__: list = []
