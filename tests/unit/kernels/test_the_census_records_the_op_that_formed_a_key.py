"""The census table's `ops` column: every key a shadow forms is recorded with the graph ops whose
dispatch formed it, and a key formed outside any op (a flow's own call) with None.

The owner's census table (2026-09-28) names the op of every key. The dispatch loops name the op they
run (`census.set_op`, only while keys are recorded), the recorder writes `<record>.ops` pairs beside
the unchanged `<record>` key lines, and the census tool turns them into rows.

What each test would do if the code were wrong: without the dispatcher's `set_op` the shadow writes
no `.ops` pair and the first test fails; an op left set after its dispatch returns would charge the
next direct call's key to it and the second assertion of the first test fails; `table_rows` dropping
op-less keys or splitting one key into a row per op fails the last test. (Seen red: `set_op` removed from
TritonSequentialDispatcher.dispatch.)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SCRIPT = r'''
from neurobrix.kernels import census
census.install()
from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.kernels import wrappers as W
from neurobrix.triton.sequential import TritonSequentialDispatcher
a = NBXTensor.empty((19, 2048), "float16", 0)
b = NBXTensor.empty((2048, 2048), "float16", 0)
d = TritonSequentialDispatcher(device_idx=0, stores_fp64=False)
d.dispatch("aten::mm", [a, b], {}, op_uid="aten.mm::7")
c = NBXTensor.empty((23, 2048), "float16", 0)
W.mm(c, b)                            # a flow's own call, after the op returned: no op
print("OK")
'''


def test_a_shadow_records_the_op_of_each_key(tmp_path):
    rec = tmp_path / "keys"
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
                "NBX_KEY_RECORD": str(rec), "PYTHONPATH": str(REPO / "src")})
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
    keys = rec.read_text().splitlines()
    ops = [l.split("\t", 1) for l in Path(str(rec) + ".ops").read_text().splitlines()]
    assert len(keys) == 2, keys
    assert ops == [["aten.mm::7", keys[0]]], ops          # the dispatched op, and ONLY it


def test_table_rows_carry_the_op_and_keep_the_op_less_key():
    sys.path.insert(0, str(REPO / "tools"))
    import certified_census as CC
    k1 = "neurobrix.kernels.ops.matmul.matmul_kernel::(19, 2048, 2048, True, True, 'fp16', 'fp16', 'fp16')"
    k2 = "neurobrix.kernels.ops.matmul.matmul_kernel::(23, 2048, 2048, True, True, 'fp16', 'fp16', 'fp16')"
    rows = CC.table_rows("M", "sha", "triton", 16384, [k1, k2],
                         [("aten.mm::7", k1), ("aten.mm::9", k1)])
    got = sorted((r["key"][:4], tuple(r["ops"]), tuple(r["rungs_mb"])) for r in rows)
    assert got == [("(19,", ("aten.mm::7", "aten.mm::9"), (16384,)), ("(23,", (None,), (16384,))]
