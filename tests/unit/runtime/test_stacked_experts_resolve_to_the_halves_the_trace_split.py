"""`expert_weight_lists` is the one place every dispatcher (compiled, sequential,
triton, triton_sequential) resolves a stacked-expert fused op's per-expert weights.
The stacked input slab is [E, 2F, H]: rows 0:F are the gate half, F:2F the up half
(the traced block splits the projection's output at F); the output slab is [E, H, F].

What this test would do if the code were wrong: swapping the halves, or reading the
down slab with the wrong index, changes which rows come back — checked by value on
a slab whose every row carries its own expert and row index. Views are NBXTensor
select/narrow on a shadow allocation (no card, no kernel: the census's own door)."""
from __future__ import annotations

import os
import subprocess
import sys

SCRIPT = r'''
import numpy as np
from neurobrix.kernels import census; census.install()
from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
E, F, H = 3, 4, 5
w_in = NBXTensor.empty((E, 2 * F, H), "float16", 0)
w_out = NBXTensor.empty((E, H, F), "float16", 0)
attrs = {"num_experts": E, "stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out", "ffn_dim": F},
         "expert_gate_weight_ids": [], "expert_up_weight_ids": [], "expert_down_weight_ids": []}
gate, up, down = expert_weight_lists(attrs, {"in": w_in, "out": w_out}.get)
assert len(gate) == len(up) == len(down) == E
for e in range(E):
    assert tuple(gate[e].shape) == (F, H) and tuple(up[e].shape) == (F, H), (gate[e].shape, up[e].shape)
    assert tuple(down[e].shape) == (H, F), down[e].shape
    # the halves are DISTINCT rows of the same slab: gate starts at row 0, up at row F
    elem = 2
    base = w_in.data_ptr() + e * (2 * F * H) * elem
    assert gate[e].data_ptr() == base, (gate[e].data_ptr(), base)
    assert up[e].data_ptr() == base + F * H * elem, (up[e].data_ptr(), base + F * H * elem)
    assert down[e].data_ptr() == w_out.data_ptr() + e * (H * F) * elem
print("HALVES OK")
'''


def test_the_halves_are_the_rows_the_trace_split_at_f():
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2500:]
    assert "HALVES OK" in r.stdout
