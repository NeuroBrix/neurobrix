"""D-TSEQ-ORPHEUS-STEP110 (2026-09-03): the per-op dump record carries a
LAST-POSITION window (`last_pos10`) beside `head10` — the first ten
elements of the last index along the sequence axis (axis 1 for rank >= 3,
batch 0; axis 0 for rank 2; None below). Pinned on the triton stats
producer (a static method on an NBXTensor) and on the CPU differential
tool that walks two dumps on that field.
"""
import json

import numpy as np
import pytest

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "dump_diff_lastpos.py"
_spec = importlib.util.spec_from_file_location("dump_diff_lastpos", _TOOL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
dev, load = _mod.dev, _mod.load


def _gpu():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="needs a CUDA device")
def test_triton_stats_last_position_window_rank3_and_rank2():
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.triton.sequence import TritonSequence
    a = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4)      # [B, S, D]
    rec = TritonSequence.nbx_tid_stats(NBXTensor.from_numpy(a))
    assert rec["head10"] == list(a[0, 0])                             # head10 = position 0's row (the producer's rule)
    assert rec["last_pos10"] == list(a[0, -1])                        # the LAST position's row (4 elements)
    b = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)               # [S, D]
    rec2 = TritonSequence.nbx_tid_stats(NBXTensor.from_numpy(b))
    assert rec2["last_pos10"] == list(b[-1])
    c = np.arange(6, dtype=np.float32)                                 # rank 1 → no window
    assert TritonSequence.nbx_tid_stats(NBXTensor.from_numpy(c))["last_pos10"] is None


def test_diff_tool_names_the_first_op_over_the_bound(tmp_path):
    A = tmp_path / "a.jsonl"
    B = tmp_path / "b.jsonl"
    recs_a = [{"engine": "x", "record": {"tid": f"op::{i}::out_0", "op_type": "aten::add",
                                          "shape": [1, 4, 8], "head10": [1.0] * 10,
                                          "last_pos10": [1.0] * 10}} for i in range(3)]
    recs_b = json.loads(json.dumps(recs_a))
    recs_b[1]["record"]["last_pos10"] = [1.0] * 9 + [2.0]     # position-local divergence at op 1
    A.write_text("\n".join(json.dumps(r) for r in recs_a) + "\n")
    B.write_text("\n".join(json.dumps(r) for r in recs_b) + "\n")
    ra, order = load(str(A))
    rb, _ = load(str(B))
    devs = [dev(ra[t]["last_pos10"], rb[t]["last_pos10"]) for t in order]
    assert devs[0] == 0.0 and devs[2] == 0.0 and devs[1] == pytest.approx(0.5)
    assert [dev(ra[t]["head10"], rb[t]["head10"]) for t in order] == [0.0, 0.0, 0.0]   # head10 is blind to it
