"""Exchange with other array libraries: DLPack, the one protocol nbxtensor speaks (never torch, never a producer's
own API)."""
from nbxtensor.interop.dlpack import from_dlpack

__all__ = ["from_dlpack"]
