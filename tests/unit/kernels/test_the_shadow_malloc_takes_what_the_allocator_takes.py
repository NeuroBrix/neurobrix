"""The census shadow replaces `DeviceAllocator.malloc_cuda` (kernels/census.py) and must take every argument the real
one takes: a weight arena asks for a block outside the pool (`pooled=False`, 2026-10-28) and the shadow without it
raised inside the arena's constructor — the 8 GB census of mochi-1-preview died at its VAE (rc 139).

What this test would do if the code were wrong: a parameter of the real allocator missing from the shadow (red)."""
import inspect

from neurobrix.kernels import census
from neurobrix.kernels.nbx_tensor import DeviceAllocator


def test_the_shadow_takes_every_parameter_of_the_real_malloc():
    real = inspect.signature(DeviceAllocator.malloc_cuda).parameters
    shadow = inspect.signature(census._shadow_malloc).parameters
    missing = [n for n in real if n not in shadow]
    assert not missing, f"the census shadow refuses {missing}, which the allocator takes"
    assert census._shadow_malloc(4096, 0, pooled=False)
