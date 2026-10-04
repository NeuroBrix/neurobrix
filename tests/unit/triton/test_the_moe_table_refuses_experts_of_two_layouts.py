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


_PROMOTE = r"""
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
NT.DeviceAllocator.ensure_triton_device = staticmethod(lambda *a, **k: None)
NT.DeviceAllocator.free_cuda = staticmethod(lambda *a, **k: None)
import neurobrix.triton.moe as M
from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
E, H, F = 4, 6, 3
src = np.arange(E * H * 2 * F, dtype=np.float16).reshape(E, H, 2 * F)

def host(shape):
    t = NBXTensor.empty_cpu(shape, NBXDtype.float16)
    a = np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape)
    ctypes.memmove(t.data_ptr(), a.ctypes.data, a.nbytes)
    return t

attrs = {"num_experts": E, "top_k": 2, "norm_topk_prob": True,
         "gate_scores_tid": "gs", "hidden_states_tid": "h",
         "stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out",
         "ffn_dim": F, "input_linear_in_axis": 0, "gate_offset": 0, "output_linear_in_axis": 0},
         "expert_gate_weight_ids": [], "expert_up_weight_ids": [], "expert_down_weight_ids": []}

# 1. the promotion: whole slabs, once each, and it says so
slab, out = host((E, H, 2 * F)), host((E, F, H))
promo = M.StackedSlabPromotion({"in": slab, "out": out}.get, 0)
g, u, d = expert_weight_lists(attrs, promo)
assert promo.per_call, "a promoted slab must mark the call's weights per-call"
assert all(v._device == "cuda" for v in g + u + d), "a view stayed on the host"
assert calls == [slab.nbytes(), out.nbytes()], ("one whole transfer per slab", calls)
assert (g[1]._strides, g[1].shape) == ((1, 2 * F), (F, H)), (g[1]._strides, g[1].shape)
resident = M.StackedSlabPromotion({"in": g[0]._base, "out": d[0]._base}.get, 0)
expert_weight_lists(attrs, resident)
assert not resident.per_call, "resident slabs are not per-call"
print("PROMOTED OK")

# 2. the table rule: per-call weights never enter (nor read) the pointer-table cache
built = []
M._build_ptr_tables = lambda *a: built.append(1) or object()
ws = [NBXTensor.empty((F, H), "float16", 0) for _ in range(E)]
fp = M._ptr_cache_fingerprint(ws, ws, ws, E)
t1 = M._tables_for_call(ws, ws, ws, E, True, 0)
assert M._ptr_cache_get(fp) is None, "a per-call table entered the cache"
t2 = M._tables_for_call(ws, ws, ws, E, False, 0)
assert M._ptr_cache_get(fp) is t2 and len(built) == 2, "a resident table is cached"
t3 = M._tables_for_call(ws, ws, ws, E, True, 0)
assert t3 is not t2 and len(built) == 3, "a per-call call must not be served a cached table"
print("TABLES OK")

# 3. both triton dispatchers hand execute_moe_fused the flag
seen = []
def fake_exec(**kw):
    seen.append(kw.get("weights_per_call")); return None
M.execute_moe_fused = fake_exec
hs = NBXTensor.empty((5, H), "float16", 0)
gs = NBXTensor.empty((5, E), "float32", 0)
from neurobrix.core.runtime.graph_executor import GraphExecutor
ex = GraphExecutor.__new__(GraphExecutor)
ex._execute_moe_fused_triton_sequential(attrs, {"in": host((E, H, 2 * F)), "out": host((E, F, H)), "h": hs, "gs": gs})
ex._execute_moe_fused_triton_sequential(attrs, {"in": g[0]._base, "out": d[0]._base, "h": hs, "gs": gs})
from neurobrix.triton.sequence import TritonSequence
ts = TritonSequence.__new__(TritonSequence)
ts._tid_to_slot = {"gs": 0, "h": 1, "in": 2, "out": 3}
cop = ts._compile_moe_fused_op("moe_fused::block.0", {"attributes": attrs, "output_tensor_ids": ["o"]}, ())
cop.func([gs, hs, host((E, H, 2 * F)), host((E, F, H))] + [None])
cop.func([gs, hs, g[0]._base, d[0]._base] + [None])
assert seen == [True, False, True, False], seen
print("DISPATCH OK"); sys.stdout.flush(); os._exit(0)
"""


def test_a_host_slab_is_promoted_whole_once_and_its_call_is_per_call():
    """The triton dispatchers promote a host-resident stacked slab ONCE and WHOLE
    before cutting the per-expert views, and TELL execute_moe_fused the weights
    live for this call (`weights_per_call`), so its zero3 rule holds: a per-call
    address never enters — nor is served from — the pointer-table cache (a reused
    address meeting a stale table reads zeros on Metal, in silence). Wrong code —
    per-view promotion, a dropped flag in either dispatcher, a per-call table
    cached — fails a named assertion. No card: host memory stands in for the
    device allocator in a subprocess."""
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    r = subprocess.run([sys.executable, "-c", _PROMOTE], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-2500:])
    for line in ("PROMOTED OK", "TABLES OK", "DISPATCH OK"):
        assert line in r.stdout, (line, r.stdout[-1500:])
