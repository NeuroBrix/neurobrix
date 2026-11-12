"""The triton dispatchers promote a host-resident stacked expert slab ONCE and WHOLE, and tell `execute_moe_fused` its
weights live for this call (see the test's own text below). The table builder's refusal of a projection mixing two
layouts, which stood here, is the library's test: `nbxtensor/tests/test_the_moe_table_refuses_experts_of_two_layouts.py`;
so is the rule that a per-call table never enters the pointer-table cache, and the promotion itself
(`nbxtensor/tests/test_a_promoted_slab_crosses_once_and_its_table_is_never_cached.py`). What stays here is the
engine's: the views it cuts from a promoted slab, and the flag both of its dispatchers hand on.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triton")


_PROMOTE = r"""
import ctypes, os, sys
import numpy as np
import neurobrix  # the engine first: it puts its own tree's nbxtensor on the path
import nbxtensor.tensor as NT
import nbxtensor.memory.allocator as _nbx_allocator
from nbxtensor.dtype.nbx_dtype import NBXDtype; from nbxtensor.tensor import NBXTensor
calls = []
keep = []
def malloc(nbytes, dev_idx=None):
    b = ctypes.create_string_buffer(max(nbytes, 1)); keep.append(b); return ctypes.addressof(b)
def memcpy(dst, src, nbytes, kind=3):
    calls.append(nbytes); ctypes.memmove(dst, src, nbytes)
_nbx_allocator.DeviceAllocator.malloc_cuda = staticmethod(malloc)
_nbx_allocator.DeviceAllocator.memcpy = staticmethod(memcpy)
_nbx_allocator.DeviceAllocator.set_device = staticmethod(lambda *a, **k: None)
_nbx_allocator.DeviceAllocator.ensure_triton_device = staticmethod(lambda *a, **k: None)
_nbx_allocator.DeviceAllocator.free_cuda = staticmethod(lambda *a, **k: None)
import neurobrix.triton.moe as M
from nbxtensor.ops.moe_fused import StackedSlabPromotion
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
promo = StackedSlabPromotion({"in": slab, "out": out}.get, 0)
g, u, d = expert_weight_lists(attrs, promo)
assert promo.per_call, "a promoted slab must mark the call's weights per-call"
assert all(v._device == "cuda" for v in g + u + d), "a view stayed on the host"
assert calls == [slab.nbytes(), out.nbytes()], ("one whole transfer per slab", calls)
assert (g[1]._strides, g[1].shape) == ((1, 2 * F), (F, H)), (g[1]._strides, g[1].shape)
resident = StackedSlabPromotion({"in": g[0]._base, "out": d[0]._base}.get, 0)
expert_weight_lists(attrs, resident)
assert not resident.per_call, "resident slabs are not per-call"
print("PROMOTED OK")

# 2. both triton dispatchers hand execute_moe_fused the flag
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
    live for this call (`weights_per_call`), so the op's zero3 rule holds (the
    library's test holds the rule itself). Wrong code — per-view promotion, a
    dropped flag in either dispatcher — fails a named assertion. No card: host memory stands in for the
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
    for line in ("PROMOTED OK", "DISPATCH OK"):
        assert line in r.stdout, (line, r.stdout[-1500:])
