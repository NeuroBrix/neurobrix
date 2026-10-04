"""The grouped GEMM reads every expert of a projection through ONE stride pair,
taken from expert 0 (`_build_ptr_tables`). Stacked-slab views share it by
construction; a list mixing layouts — a per-expert [F, H] matrix beside a
transposed view of a slab — would be read through the wrong strides in silence.
The table builder refuses it by name before any table is built.

What this test would do if the code were wrong: without the door the builder
goes on toward the device — which this test replaces by a sentinel that raises
"past the door" — and never names the mismatch: the `pytest.raises` fails. No
card: the refusal happens before any allocation.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triton")


class _W:
    def __init__(self, shape, strides):
        self.shape, self._strides, self._device_idx = shape, strides, 0

    def stride(self, d):
        return self._strides[d]


def _past_the_door(*_a, **_k):
    raise RuntimeError("past the door: the builder went on to the device")


def test_a_projection_mixing_layouts_is_refused_by_name(monkeypatch):
    from neurobrix.triton import moe as M
    from neurobrix.triton.moe import _build_ptr_tables
    monkeypatch.setattr(M, "_detect_gpu_backend", lambda: "cuda")
    monkeypatch.setattr(M.DeviceAllocator, "set_device", staticmethod(_past_the_door))
    F, H = 4, 6
    linear = [_W((F, H), (H, 1)) for _ in range(3)]               # (out x in), k contiguous
    mixed = linear[:2] + [_W((F, H), (1, 2 * F))]                  # a transposed slab view
    with pytest.raises(RuntimeError, match=r"'up' expert 2 has \(shape, strides\)"):
        _build_ptr_tables(linear, mixed, linear)


_PROMOTE = r'''
import ctypes, os, sys
import numpy as np
import neurobrix.kernels.nbx_tensor as NT
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
calls = []
keep = []
def malloc(nbytes, dev_idx=None):
    b = ctypes.create_string_buffer(max(nbytes, 1)); keep.append(b); return ctypes.addressof(b)
def memcpy(dst, src, nbytes, kind=3):
    calls.append(nbytes); ctypes.memmove(dst, src, nbytes)
NT.DeviceAllocator.malloc_cuda = staticmethod(malloc)
NT.DeviceAllocator.memcpy = staticmethod(memcpy)
NT.DeviceAllocator.set_device = staticmethod(lambda *a, **k: None)
NT.DeviceAllocator.free_cuda = staticmethod(lambda *a, **k: None)
from neurobrix.triton.moe import promote_stacked_slabs
from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
E, H, F = 4, 6, 3
src = np.arange(E * H * 2 * F, dtype=np.float16).reshape(E, H, 2 * F)
slab = NBXTensor.empty_cpu(src.shape, NBXDtype.float16)
ctypes.memmove(slab.data_ptr(), src.ctypes.data, src.nbytes)
out = NBXTensor.empty_cpu((E, F, H), NBXDtype.float16)
attrs = {"num_experts": E, "stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out",
         "ffn_dim": F, "input_linear_in_axis": 0, "gate_offset": 0, "output_linear_in_axis": 0},
         "expert_gate_weight_ids": [], "expert_up_weight_ids": [], "expert_down_weight_ids": []}
g, u, d = expert_weight_lists(attrs, promote_stacked_slabs({"in": slab, "out": out}.get, 0))
assert all(v._device == "cuda" for v in g + u + d), "a view stayed on the host"
assert calls == [slab.nbytes(), out.nbytes()], ("one whole transfer per slab", calls)
assert (g[1]._strides, g[1].shape) == ((1, 2 * F), (F, H)), (g[1]._strides, g[1].shape)
print("PROMOTED OK"); sys.stdout.flush(); os._exit(0)
'''


def test_a_host_slab_is_promoted_whole_once():
    """The triton dispatchers promote a host-resident stacked slab ONCE and WHOLE
    before cutting the per-expert views (never 3E per-view copies). Wrong code —
    per-view promotion, or none — fails the transfer count or the device check.
    No card: host memory stands in for the device allocator in a subprocess."""
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    r = subprocess.run([sys.executable, "-c", _PROMOTE], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-2500:])
    assert "PROMOTED OK" in r.stdout
