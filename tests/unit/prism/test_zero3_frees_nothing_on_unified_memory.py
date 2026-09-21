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


def test_the_live_profile_answers_from_its_own_device_and_a_discrete_one_is_not():
    """The two real profiles, so this cannot pass on constructed specs alone.

    Written first as "this machine's profile must read as unified", which was
    true on the Mac that wrote it and false on the Dell that merged it — a test
    that asserts what one machine is, run on another, is the semantic conflict
    no git conflict signals (2026-09-13). So the live profile is asked what ITS
    device declares, and the assertion is that the predicate agrees with the
    device — on either kind of machine.
    """
    # THE MACHINE'S OWN profile through the door every run uses (autodetect), never the
    # literal "default": on the Dell the file is default-<hash>.yml and load_profile("default")
    # raised FileNotFoundError (2026-09-21) — a cell that reads the host must ask the host.
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    live = load_profile(get_or_create_default_profile())
    if not live.devices:
        import pytest
        pytest.skip("this host has no device in its live profile (a masked or CPU-only run)")
    dev = live.devices[0]
    assert dev.has_unified_memory == _device_is_unified(f"zero3:x:{dev.index}", live), (
        "the budget's predicate must read the same answer as the device itself"
    )
    if getattr(dev, "unified_memory", None) is not None:
        assert dev.has_unified_memory is bool(dev.unified_memory)

    a100 = load_profile("a100-80g")
    assert a100.devices[0].has_unified_memory is False


def test_zero3_offload_frees_memory_only_on_a_discrete_device():
    """The predicate the budget consults, on both shapes of machine.

    `False` on the discrete card is the inertia proof: NVIDIA's zero3
    accounting is byte-unchanged by this, because the branch it guards is
    only ever taken where the device says it shares memory with the host.
    """
    unified = load_profile("a10-24g")                # a FIXTURE in the tree, the same on every machine
    unified.devices[0].unified_memory = True          # declared unified by the test, whatever the file says
    a100 = load_profile("a100-80g")

    assert _device_is_unified("zero3:mps:0", unified) is True
    assert _device_is_unified("zero3:cuda:0", a100) is False

    # A device index the profile does not have is not a licence to guess.
    assert _device_is_unified("zero3:cuda:7", a100) is False
    assert _device_is_unified("not a device", unified) is False
    assert _device_is_unified("zero3:mps:0", None) is False
