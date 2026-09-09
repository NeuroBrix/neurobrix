"""zero3's offload frees device memory only where host and device are disjoint.

The solver excluded a `zero3:` component's WEIGHTS from the device budget,
counting only activations, on the reasoning that zero3 keeps weights in CPU
pinned memory. That is exact on a discrete card, where host RAM and device
memory are different silicon. **On a unified device they are the same bytes**,
so the "offload" moves nothing and the budget must still see the weights.

Measured 2026-09-09, `DeepSeek-Coder-V2-Lite-Instruct` on a 24576 MB unified
device: Prism scored `lazy_sequential` "the only viable strategy" with its
`model` component on zero3 at **31259.5 MB** — 27% over the budget it had
just read. The run then planned 32104 MB; the eager arm was **SIGKILLed** by
the OS and the Triton arm's allocator refused at the budget, reporting a
5.5 MB pinned-allocation failure that was really a full working set.

This is the shape the mandate names: a plan accepted under one memory model
and executed under another. It is wrong on every vendor that has the
property, which is why the code asks the DEVICE and never the brand — an APU
on `hip:0` and an integrated `xpu:0` are answered by the same line.
"""

from __future__ import annotations

from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import _device_is_unified
from neurobrix.core.prism.structure import DeviceSpec, DeviceBrand


def test_unified_is_read_from_the_device_not_the_brand():
    """Brand cannot answer this: a discrete Radeon is not unified, an APU is.

    So the property is the architecture's, and the profile's own
    `unified_memory` overrides it whenever the profile says anything.
    """
    discrete = DeviceSpec(index=0, name="d", memory_mb=1024,
                          compute_capability="8.0", supports_dtypes=["float16"],
                          architecture="ampere", brand=DeviceBrand.NVIDIA)
    unified = DeviceSpec(index=0, name="u", memory_mb=1024,
                         compute_capability="0.0", supports_dtypes=["float16"],
                         architecture="apple_silicon", brand=DeviceBrand.APPLE)
    assert discrete.has_unified_memory is False
    assert unified.has_unified_memory is True

    # An explicit profile value wins in BOTH directions — the architecture
    # table is the fallback for profiles written before the field existed,
    # not an override of one that states it.
    said_yes = DeviceSpec(index=0, name="x", memory_mb=1024,
                          compute_capability="8.0", supports_dtypes=["float16"],
                          architecture="ampere", brand=DeviceBrand.NVIDIA,
                          unified_memory=True)
    said_no = DeviceSpec(index=0, name="y", memory_mb=1024,
                         compute_capability="0.0", supports_dtypes=["float16"],
                         architecture="apple_silicon", brand=DeviceBrand.APPLE,
                         unified_memory=False)
    assert said_yes.has_unified_memory is True
    assert said_no.has_unified_memory is False


def test_the_live_profile_is_unified_and_a_discrete_one_is_not():
    """The two real profiles, so this cannot pass on constructed specs alone."""
    apple = load_profile("default")
    assert apple.devices[0].has_unified_memory is True, (
        "this machine's profile must read as unified; if it does not, the "
        "zero3 accounting is back to freeing memory it cannot free")

    a100 = load_profile("a100-80g")
    assert a100.devices[0].has_unified_memory is False


def test_zero3_offload_frees_memory_only_on_a_discrete_device():
    """The predicate the budget consults, on both shapes of machine.

    `False` on the discrete card is the inertia proof: NVIDIA's zero3
    accounting is byte-unchanged by this, because the branch it guards is
    only ever taken where the device says it shares memory with the host.
    """
    apple = load_profile("default")
    a100 = load_profile("a100-80g")

    assert _device_is_unified("zero3:mps:0", apple) is True
    assert _device_is_unified("zero3:cuda:0", a100) is False

    # A device index the profile does not have is not a licence to guess.
    assert _device_is_unified("zero3:cuda:7", a100) is False
    assert _device_is_unified("not a device", apple) is False
    assert _device_is_unified("zero3:mps:0", None) is False
