"""Triton Memory Pool — arena-based GPU memory allocator.

Pre-allocates one contiguous block per component, sub-allocates inside
via bump pointer. No individual free, no fragmentation.

Zero torch. Uses DeviceAllocator (hardware-agnostic: CUDA/ROCm).
"""

from neurobrix.kernels.nbx_tensor import DeviceAllocator


class ComponentArena:
    """Single contiguous GPU memory block for one component's weights.

    Allocates one big block via DeviceAllocator.malloc_cuda, then
    sub-allocates inside via bump pointer with 256-byte alignment.

    Usage:
        arena = ComponentArena(total_bytes, device_idx)
        ptr1 = arena.alloc(1024)   # sub-allocate 1024 bytes
        ptr2 = arena.alloc(4096)   # sub-allocate 4096 bytes
        arena.free()               # release entire block at once
    """

    ALIGNMENT = 256  # GPU memory alignment (matches CUDA spec)

    def __init__(self, total_bytes: int, device_idx: int):
        self.device_idx = device_idx
        self._total = self._align(total_bytes)
        self._offset = 0

        DeviceAllocator.set_device(device_idx)
        self._base_ptr = DeviceAllocator.malloc_cuda(self._total)
        if self._base_ptr == 0 and self._total > 0:
            raise RuntimeError(
                f"ComponentArena: failed to allocate {self._total / 1e9:.2f} GB "
                f"on device {device_idx}")

    def alloc(self, nbytes: int) -> int:
        """Sub-allocate nbytes from the arena. Returns GPU pointer."""
        if nbytes == 0:
            return 0
        aligned = self._align(nbytes)
        if self._offset + aligned > self._total:
            raise RuntimeError(
                f"ComponentArena: out of memory. "
                f"Requested {nbytes} bytes, "
                f"used {self._offset}/{self._total} bytes "
                f"on device {self.device_idx}")
        ptr = self._base_ptr + self._offset
        self._offset += aligned
        return ptr

    def free(self):
        """Release the entire arena block."""
        if self._base_ptr:
            DeviceAllocator.set_device(self.device_idx)
            DeviceAllocator.free_cuda(self._base_ptr)
            self._base_ptr = 0
            self._offset = 0
            self._total = 0

    def reset(self):
        """Reset bump pointer without freeing GPU memory.

        Allows reuse of the same arena for a new component load
        (e.g., lifecycle reload in lazy_sequential strategy).
        """
        self._offset = 0

    @property
    def used(self) -> int:
        return self._offset

    @property
    def capacity(self) -> int:
        return self._total

    def _align(self, n: int) -> int:
        return (n + self.ALIGNMENT - 1) & ~(self.ALIGNMENT - 1)

    def __del__(self):
        self.free()
