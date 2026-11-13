"""Exchange with other array libraries: DLPack, the one protocol nbxtensor speaks (never torch, never a producer's
own API)."""
from nbxtensor.interop.dlpack import from_dlpack
from nbxtensor.interop.handoff import copy_to_address, host_bytes

__all__ = ["from_dlpack", "copy_to_address", "host_bytes"]
