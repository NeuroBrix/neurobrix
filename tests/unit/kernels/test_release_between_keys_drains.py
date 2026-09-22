"""The certifier's between-key drain must resolve to a method the allocator REALLY has.

Red on 2026-09-22: `_release_between_keys` probed `empty_cache` then `device_empty_cache`.
DeviceAllocator has neither — its pool drain is `empty_cache_pool` — so `drain` resolved to
None, `callable(drain)` was False, and the pool was never drained. No exception was raised and
nothing was logged, so a whole Apple certification campaign ran with the drain inert while its
docstring claimed the device was given back between keys. The certifier reached a 21.9 GB
physical footprint on a 24 GB machine, drove swap to 11.6 GB of 12.3 GB, and the stability
witness then refused sweeps for 8.8 % drift because the GPU regime moved under memory pressure.

This test would have failed the day the probe was written, which is the only reason to have it.
"""
import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator
from neurobrix.kernels import autotune_certify as AC


def test_allocator_exposes_the_drain_the_certifier_asks_for():
    """Whatever name the certifier probes, the allocator must actually answer to it."""
    resolved = AC._resolve_pool_drain()
    assert callable(resolved), (
        "the certifier's pool drain resolved to nothing: it is a no-op and the pool is "
        f"never returned between keys. DeviceAllocator exposes: "
        f"{sorted(n for n in dir(DeviceAllocator) if 'cache' in n or 'pool' in n)}"
    )


def test_a_missing_drain_is_refused_not_ignored():
    """ZERO FALLBACK: an allocator with no drain at all must raise, never pass silently."""
    class NoDrain:
        pass

    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        AC._resolve_pool_drain(allocator=NoDrain)


def test_release_between_keys_actually_calls_it():
    """The drain is not merely resolvable — the between-key hook must invoke it."""
    calls = []

    class Spy:
        @staticmethod
        def empty_cache_pool():
            calls.append(1)

    AC._release_between_keys(allocator=Spy)
    assert calls == [1], "_release_between_keys did not call the allocator's drain"
