"""`expert_weight_lists` is the one place every dispatcher (compiled, sequential,
triton, triton_sequential) resolves a stacked-expert fused op's per-expert weights.
Every dispatcher consumes the nn.Linear layout — gate and up [F, H], down [H, F] —
and the spec says how the slabs hold it, as the matcher READ it off the graph:

  * granite (select -> t -> mm): W_in [E, 2F, H], W_out [E, H, F] — in_axis 1;
  * Qwen3-VL 4.57 (bmm over all experts): W_in [E, H, 2F], W_out [E, F, H] —
    in_axis 0, the per-expert matrices returned as transposed VIEWS;
  * the gate half at `gate_offset` along the 2F axis, the up half the other F.

What this test would do if the code were wrong: swapping the halves, reading the
slab along the wrong axis, or dropping the transpose changes the data_ptr / shape /
strides the grouped GEMM is handed — each is checked exactly, for both geometries
and both gate offsets. A spec without its geometry is refused by name (the reader
never assumes one). Views are NBXTensor select/narrow/t on a shadow allocation (no
card, no kernel: the census's own door)."""
from __future__ import annotations

import os
import subprocess
import sys

SCRIPT = r'''
from neurobrix.kernels import census; census.install()
from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
E, F, H = 3, 4, 5
elem = 2

def attrs(in_axis, g_off, out_in_axis):
    return {"num_experts": E,
            "stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out", "ffn_dim": F,
                                "input_linear_in_axis": in_axis, "gate_offset": g_off,
                                "output_linear_in_axis": out_in_axis},
            "expert_gate_weight_ids": [], "expert_up_weight_ids": [], "expert_down_weight_ids": []}

# granite: W_in [E, 2F, H] rows, W_out [E, H, F]
w_in = NBXTensor.empty((E, 2 * F, H), "float16", 0)
w_out = NBXTensor.empty((E, H, F), "float16", 0)
for g_off in (0, F):
    gate, up, down = expert_weight_lists(attrs(1, g_off, 1), {"in": w_in, "out": w_out}.get)
    assert len(gate) == len(up) == len(down) == E
    for e in range(E):
        base = w_in.data_ptr() + e * (2 * F * H) * elem
        assert tuple(gate[e].shape) == (F, H) and tuple(up[e].shape) == (F, H)
        assert (gate[e].stride(0), gate[e].stride(1)) == (H, 1), gate[e].stride(0)
        assert gate[e].data_ptr() == base + g_off * H * elem, ("granite gate", e, g_off)
        assert up[e].data_ptr() == base + (F - g_off) * H * elem, ("granite up", e, g_off)
        assert tuple(down[e].shape) == (H, F) and (down[e].stride(0), down[e].stride(1)) == (F, 1)
        assert down[e].data_ptr() == w_out.data_ptr() + e * (H * F) * elem
print("GRANITE OK")

# Qwen3-VL (transformers 4.57): W_in [E, H, 2F] columns, W_out [E, F, H]
w_in = NBXTensor.empty((E, H, 2 * F), "float16", 0)
w_out = NBXTensor.empty((E, F, H), "float16", 0)
for g_off in (0, F):
    gate, up, down = expert_weight_lists(attrs(0, g_off, 0), {"in": w_in, "out": w_out}.get)
    for e in range(E):
        base = w_in.data_ptr() + e * (H * 2 * F) * elem
        # (out x in) = [F, H]: a transposed view of W_in[e][:, g:g+F] — n is the
        # contiguous axis, k strides by the slab's 2F row
        assert tuple(gate[e].shape) == (F, H) and tuple(up[e].shape) == (F, H), gate[e].shape
        assert (gate[e].stride(0), gate[e].stride(1)) == (1, 2 * F), (gate[e].stride(0), gate[e].stride(1))
        assert (up[e].stride(0), up[e].stride(1)) == (1, 2 * F)
        assert gate[e].data_ptr() == base + g_off * elem, ("qwen gate", e, g_off)
        assert up[e].data_ptr() == base + (F - g_off) * elem, ("qwen up", e, g_off)
        assert tuple(down[e].shape) == (H, F) and (down[e].stride(0), down[e].stride(1)) == (1, H)
        assert down[e].data_ptr() == w_out.data_ptr() + e * (F * H) * elem
print("QWEN3VL OK")

bad = attrs(0, 0, 0)
del bad["stacked_experts"]["gate_offset"]
try:
    expert_weight_lists(bad, {"in": w_in, "out": w_out}.get)
except RuntimeError as exc:
    assert "gate_offset" in str(exc), exc
    print("REFUSED OK")
'''


def test_the_halves_are_the_views_the_graph_named():
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2500:]
    for line in ("GRANITE OK", "QWEN3VL OK", "REFUSED OK"):
        assert line in r.stdout, (line, r.stdout[-1500:])
