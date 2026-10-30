"""The Triton branch's release fence on Metal (state engine step 2, docs/design/state-engine.md §4.3).

Not in use today: the Apple vendor files declare `memory.async_release: false`, because every launch through nbxtensor
waits for its kernel (docs/nbxtensor/BACKENDS.md, "Releasing state pages on Metal"), so no kernel is running when a page
is released. The day launches stop waiting, a released page may be granted again only once every kernel that read it
has completed (core/state/pages.py, the release queue): the Apple backend then numbers each committed command buffer
(`fence()`) and reports the highest completed without waiting (`completed_fence()`), and this class reads them. The
backend module is handed in by the caller; one without those two calls is refused by name. Torch-free (R33).
"""
from __future__ import annotations


class MetalStateFence:
    def __init__(self, native):
        if not hasattr(native, "completed_fence"):
            raise RuntimeError(
                "ZERO FALLBACK: the Apple backend has no fence (fence / completed_fence): state pages cannot be released "
                "asynchronously on Metal without one — docs/nbxtensor/BACKENDS.md, 'Releasing state pages on Metal'")
        self._native = native

    def record(self) -> int:
        """The fence past every kernel launched so far: what a page they read is released under."""
        return int(self._native.fence())

    def completed(self) -> int:
        """The highest fence whose kernels have all completed, without waiting."""
        return int(self._native.completed_fence())

    def drain(self, pool) -> int:
        """Return to `pool` the pages whose fence has completed; how many came back."""
        return pool.fence_completed(self.completed())
