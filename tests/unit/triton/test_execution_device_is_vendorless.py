"""The Triton strategies must accept every GPU Prism can name.

Prism emits a device string from `DeviceSpec.brand.to_device_prefix()`:
`cuda:0` for NVIDIA, `hip:0` for AMD, `xpu:0` for Intel, `mps:0` for Apple.
`LazySequentialStrategy._execution_device` matched `startswith("cuda")`, so on
AMD and Apple the plan device was rejected, the executor's device — which
keeps the plan's prefix, see `_load_weights_triton`'s `if ':' in
str(self.device)` branch — was rejected too, and the strategy raised
ZERO FALLBACK on a machine that has a GPU. Every model Prism assigned
lazy_sequential was therefore unreachable on the Triton branch off NVIDIA.

`transfer_tensor` had the same shape: it matched only the literal `cuda:` and
let every other accelerator fall through to `return tensor`, a silent no-op
that is indistinguishable from a successful transfer. `to_cuda` is the
GENERIC accelerator move in this layer — an NBXTensor reports `_device ==
'cuda'` on Metal too — so routing the other prefixes into it is the fix, not
a special case.

Both changes are inert on NVIDIA: `cuda:0` was accepted before and is
accepted now, `cpu` was rejected before and is rejected now.
"""

from __future__ import annotations

import pytest

from neurobrix.core.strategies.triton.lazy_sequential import _names_accelerator


NVIDIA = ["cuda:0", "cuda:1", "cuda:7"]
OTHER_GPUS = ["hip:0", "hip:3", "mps:0", "xpu:0", "tt:0"]
NOT_A_GPU = ["cpu", "", "cuda", "hip", "mps", "meta", "cpu:0"]


@pytest.mark.parametrize("dev", NVIDIA)
def test_nvidia_is_accepted_exactly_as_before(dev):
    assert _names_accelerator(dev)


@pytest.mark.parametrize("dev", OTHER_GPUS)
def test_every_other_accelerator_prism_can_name_is_accepted(dev):
    assert _names_accelerator(dev), (
        f"Prism emits {dev} for a real GPU; rejecting it raises ZERO FALLBACK "
        f"on hardware that has one"
    )


@pytest.mark.parametrize("dev", NOT_A_GPU)
def test_a_staging_location_is_still_refused(dev):
    """A bare prefix carries no index, which is what 'CPU-staged' looks like.

    This is the distinction `_execution_device` exists to make: a plan device
    without a GPU index is a weight-staging slot, never an execution device.
    """
    assert not _names_accelerator(dev)


class _T:
    """Minimal NBXTensor stand-in: it must look like one to be transferred."""

    def __init__(self):
        self._device = "cuda"
        self.moved_to = None

    def to_cuda(self, idx=0):
        self.moved_to = ("gpu", idx)
        return self

    def to_cpu(self):
        self.moved_to = ("cpu", None)
        return self


def _transfer(dev):
    from neurobrix.core.strategies.triton.base import TritonStrategy
    t = _T()
    out = TritonStrategy.transfer_tensor(None, t, dev)
    return out.moved_to


@pytest.mark.parametrize("dev,idx", [("cuda:0", 0), ("cuda:2", 2), ("cuda", 0)])
def test_transfer_on_nvidia_is_unchanged(dev, idx):
    assert _transfer(dev) == ("gpu", idx)


@pytest.mark.parametrize("dev,idx", [("hip:0", 0), ("hip:3", 3), ("mps:0", 0), ("xpu:1", 1)])
def test_transfer_routes_every_accelerator_instead_of_silently_doing_nothing(dev, idx):
    assert _transfer(dev) == ("gpu", idx), (
        f"{dev} fell through to a no-op that reads as a successful transfer"
    )


def test_transfer_to_cpu_still_goes_to_cpu():
    assert _transfer("cpu") == ("cpu", None)
