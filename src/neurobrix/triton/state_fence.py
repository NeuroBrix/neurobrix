"""The Triton branch's release fence on Metal (state engine step 2, docs/design/state-engine.md §4.3).

A state page released by its last holder may be granted again only once every kernel that read it has completed
(core/state/pages.py, the release queue). On Metal the kernels run on the Apple backend's own command queue, whose
buffers complete in commit order; the backend numbers each committed buffer (`metal_native.fence`) and reports the
highest completed without waiting (`metal_native.completed_fence`). A page copy for copy-on-write is a kernel on the
same queue, so the same fence covers it. Torch-free (R33).
"""
from __future__ import annotations


class MetalStateFence:
    def __init__(self, native=None):
        if native is None:
            from neurobrix.triton.triton_ext_driver import _native
            native = _native()
        if not hasattr(native, "completed_fence"):
            raise RuntimeError(
                "ZERO FALLBACK: the Apple backend has no fence (metal_native.fence / completed_fence, triton-ext-nbx "
                "branch state-fences): state pages cannot be released on Metal without one — rebuild the backend")
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
