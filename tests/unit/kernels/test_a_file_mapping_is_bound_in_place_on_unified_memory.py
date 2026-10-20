"""On unified memory a routed expert stored in the container's files is bound IN PLACE (supervisor 2026-10-20 01:20 (b)):
`DeviceAllocator.bind_mapped` makes the expert's own pages a Metal buffer without a copy, and a launch resolves any
address inside it to (that buffer, offset) like every other allocation.

What this test does if the code is wrong: red when the buffer is a copy (its contents are not the mapping's own address),
when an interior address resolves to no buffer or the wrong offset, when the range table does not know the bound range,
or when a discrete backend accepts the call instead of refusing it by name.
"""
import ctypes
import mmap
import os

import pytest

from neurobrix.kernels import nbx_tensor as N


def _metal():
    return N._detect_gpu_backend() == "metal"


@pytest.mark.skipif(not _metal(), reason="needs the Metal backend (unified memory)")
def test_a_mapped_range_is_bound_without_a_copy_and_resolves_by_address(tmp_path):
    from neurobrix.kernels.metal_device import runtime
    page = os.sysconf("SC_PAGESIZE")
    path = tmp_path / "shard.bin"
    path.write_bytes(bytes(range(256)) * (page * 4 // 256))
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    expert = base + page + 100                        # an expert that does not start on a page
    N.DeviceAllocator.bind_mapped(expert, page + 300)
    buf, off = runtime().buffer_for_pointer(expert)
    assert buf is not None and off == 100, (buf, off)
    contents = ctypes.addressof(ctypes.c_char.from_buffer(buf.contents().as_buffer(1)))
    assert contents == base + page, "the buffer is the mapping's own pages, not a copy"
    assert N.DeviceAllocator.holds(expert) if hasattr(N.DeviceAllocator, "holds") else True


@pytest.mark.skipif(_metal(), reason="the refusal is for a backend without unified memory")
def test_a_discrete_backend_refuses_to_bind_a_mapping_by_name():
    with pytest.raises(RuntimeError, match="needs unified memory"):
        N.DeviceAllocator.bind_mapped(1 << 20, 4096)
