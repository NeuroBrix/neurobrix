"""One vendorless predicate, derived from the brand mapping itself.

`startswith("cuda")` guards silently take the no-GPU branch on every other
brand. The replacement derives its prefix set from `DeviceBrand` rather than
listing prefixes by hand, so adding a brand cannot leave a guard behind — which
is precisely how the existing ones came to exclude AMD, Apple and Intel.

Inert on NVIDIA by construction: `cuda:N` was accepted before and is accepted
now, `cpu` was refused and is refused.
"""

from __future__ import annotations

import pytest

from neurobrix.core.prism.structure import DeviceBrand, names_accelerator


def test_the_prefix_set_is_derived_from_the_brands():
    """Hand-listing prefixes is what let brands fall through the cracks."""
    assert DeviceBrand.device_prefixes() == frozenset(
        b.to_device_prefix() for b in DeviceBrand)


@pytest.mark.parametrize("brand", list(DeviceBrand))
def test_every_brand_prism_can_emit_is_recognised(brand):
    dev = f"{brand.to_device_prefix()}:0"
    assert names_accelerator(dev), (
        f"Prism emits {dev} for a {brand.name} GPU; a guard that misses it "
        f"takes the no-GPU branch on hardware that has one"
    )


@pytest.mark.parametrize("dev", ["cuda:0", "cuda:1", "cuda:7"])
def test_nvidia_is_unchanged(dev):
    assert names_accelerator(dev)


@pytest.mark.parametrize("dev", ["cpu", "cpu:0", "", None, "cuda", "hip", "mps"])
def test_host_memory_and_bare_prefixes_are_refused(dev):
    """A bare prefix carries no index — that is what CPU-staged looks like."""
    assert not names_accelerator(dev)


def test_executor_hybrid_detection_sees_every_gpu():
    """The live site: a cpu+GPU plan must read as hybrid on any brand.

    `_is_hybrid_dispatch` gates the explicit transfers a host-resident
    producer output needs when it crosses a graph-executor boundary. With mps
    missing from the tuple, an Apple plan mixing cpu and GPU was not detected
    and those transfers were skipped.
    """
    import inspect
    from neurobrix.core.runtime import executor as ex

    src = inspect.getsource(ex)
    idx = src.index("seen_gpu = False")
    window = src[idx:idx + 700]
    code = "\n".join(l for l in window.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "_names_accelerator(dev)" in code, (
        "hybrid detection no longer uses the vendorless predicate"
    )
    assert "('cuda', 'hip', 'xpu')" not in code, (
        "hybrid detection lists prefixes by hand again — it will miss a brand"
    )
