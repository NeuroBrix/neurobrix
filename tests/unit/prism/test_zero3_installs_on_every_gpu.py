"""zero3 must install its ratchet on the GPU it just resolved.

`Zero3Strategy._get_exec_device` resolves against ("cuda", "hip", "xpu",
"mps") and even falls back to `mps:0`. `install_for_executor` then gated the
install on `startswith("cuda")`, contradicting the resolution twelve lines
above it: on Apple and AMD the class picked the right device and declined to
install on it.

The consequence was not a slowdown. zero3 is the cascade's offload rung —
weights on CPU, compute on GPU, at most two blocks resident via the ratchet.
With the install skipped, `_pin_cpu_weights` still ran and nothing managed GPU
residency, so the strategy Prism had selected did nothing at all. Prism
assigns it to the largest components there are: a 31 GB one landed on `mps:0`
in the Apple budget run.

One tuple now serves both the resolution and the install decision, so they
cannot drift apart again.
"""

from __future__ import annotations

import pytest

from neurobrix.core.strategies.zero3 import _ZERO3_GPU_PREFIXES


@pytest.mark.parametrize("dev", ["cuda:0", "cuda:7"])
def test_nvidia_installs_exactly_as_before(dev):
    assert dev.startswith(_ZERO3_GPU_PREFIXES)


@pytest.mark.parametrize("dev", ["hip:0", "mps:0", "xpu:0"])
def test_every_gpu_the_class_resolves_also_installs(dev):
    assert dev.startswith(_ZERO3_GPU_PREFIXES), (
        f"_get_exec_device resolves {dev}; refusing to install on it makes "
        f"the strategy a no-op on hardware Prism assigns it to"
    )


def test_cpu_does_not_install():
    """The ratchet needs a GPU; CPU-only is the useless case it must skip."""
    assert not "cpu".startswith(_ZERO3_GPU_PREFIXES)


def test_resolution_and_install_use_the_same_set():
    """They disagreed before, and the disagreement was the bug."""
    import inspect
    from neurobrix.core.strategies import zero3

    src = inspect.getsource(zero3)
    assert src.count("_ZERO3_GPU_PREFIXES") >= 3, (
        "the resolution and the install decision must share one tuple"
    )
    body = src[src.index("def install_for_executor"):]
    body = body[:body.index("def ", 10)] if "def " in body[10:] else body
    # Only CODE counts: the comment in that method quotes the old form to
    # explain what was wrong, and matching prose would fail on the fix itself.
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))
    assert 'self.exec_device.startswith("cuda")' not in code, (
        "install_for_executor branches on cuda again — zero3 is a no-op on "
        "every other GPU"
    )
