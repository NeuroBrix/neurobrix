"""The runtime and everything compiled against it have ONE lifetime.

Three caches hold the Metal runtime:

* `metal_device._RUNTIME` itself,
* the compiled kernels in `metal_driver`, which hold the device and the
  pipelines they were built on,
* `nbx_tensor._gpu_runtime`, an `lru_cache` the allocator resolves every
  malloc through.

Clearing any subset is worse than clearing none. The tensors then come from
one runtime and the kernels from another, and a launch refuses a perfectly
good tensor with "which the Metal allocator did not hand out" — a loud,
correct refusal of a situation that should never have arisen.

Measured 2026-09-05: this showed up as `test_cumsum_is_correct[128]` failing
in the full kernels suite and passing in every smaller selection. The
contaminating file was `test_metal_allocator.py`, the only caller of
`reset_runtime_for_tests`. Repairing the lifetime turned 15 further failures
in that neighbourhood green.
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_metal():
    try:
        from neurobrix.kernels import nbx_tensor
        return nbx_tensor._detect_gpu_backend() == "metal"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="no Apple GPU here")


def test_a_launch_still_works_after_the_runtime_is_reset():
    """Allocate, launch, reset, allocate, launch. Both must compute."""
    from neurobrix.kernels import metal_device
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import add

    def one_round():
        x = NBXTensor.from_numpy(np.arange(512, dtype=np.float32))
        y = NBXTensor.from_numpy(np.ones(512, dtype=np.float32))
        return add(x, y).numpy()

    expected = np.arange(512, dtype=np.float32) + 1.0
    assert np.array_equal(one_round(), expected)

    metal_device.reset_runtime_for_tests()

    assert np.array_equal(one_round(), expected), (
        "a launch after reset_runtime_for_tests produced wrong results: the "
        "tensors and the kernels are bound to different runtimes")


def test_resetting_the_runtime_clears_every_cache_that_holds_it():
    """The invariant itself, stated rather than inferred from behaviour."""
    from neurobrix.kernels import metal_device, nbx_tensor
    from neurobrix.triton import metal_driver

    # Populate all three.
    x = nbx_tensor.NBXTensor.from_numpy(np.ones(64, dtype=np.float32))
    from neurobrix.kernels.wrappers import add
    add(x, x)
    assert metal_driver._KERNEL_CACHE, "no compiled kernel to invalidate"
    assert nbx_tensor._gpu_runtime.cache_info().currsize == 1

    metal_device.reset_runtime_for_tests()

    assert not metal_driver._KERNEL_CACHE, (
        "compiled kernels outlived the runtime they were built on")
    assert not metal_driver._LIBRARY_CACHE, (
        "compiled libraries outlived the device they were built on")
    assert nbx_tensor._gpu_runtime.cache_info().currsize == 0, (
        "the allocator still resolves mallocs through the old runtime")
