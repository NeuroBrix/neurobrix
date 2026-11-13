"""A tensor's values handed out to another array library without that library's API: copied into a buffer it owns on
the same device heap, or as host bytes in C order (the reader types them itself). The engine's torch hand-off used
raw copies and `ctypes.string_at` for both until 2026-11-13."""
import ctypes


def copy_to_address(tensor, address: int) -> None:
    """The tensor's values, in C order, copied into device memory at `address`, a buffer another library allocated
    on the same heap and device with room for them (a strided view is made contiguous first)."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if tensor.nbytes() == 0:
        return
    src = tensor.contiguous()
    DeviceAllocator.memcpy(int(address), src.data_ptr(), src.numel() * src.element_size())


def host_bytes(tensor) -> bytearray:
    """The tensor's values, in C order, as host bytes (read through the host when the tensor is on a device)."""
    if tensor.nbytes() == 0:
        return bytearray()
    host = tensor.contiguous()
    if not host.is_cpu:
        host = host.to_cpu()
    return bytearray(ctypes.string_at(host.data_ptr(), host.numel() * host.element_size()))
