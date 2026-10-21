"""The profile's walk dumped op by op (`NBX_PRISM_CURVE_DUMP=<dir>`, default off): the in-op figure and the live set AFTER
the op's frees, one line per op, beside a header naming the peak and its largest live tensors — what step A of the Prism
work compares with an engine's measured live bytes (`NBX_NATIVE_LIVE_DUMP_EVERY`, `NBX_LIVE_DUMP_EVERY`). Real-ESRGAN x4
at 448 px: the walk matched the sequential engine within 1 MB at each of its 1 093 ops (2026-10-21).

Injection (seen red, then restored green, 2026-10-21): the line appended before the frees -> the "after" figure of an
op that frees its input equals its in-op figure and the after-assertion is red."""
from __future__ import annotations

import json

from neurobrix.core.prism.profiler import ActivationProfiler


def _t(shape):
    return {"shape": list(shape), "dtype": "float32"}


def _op(uid, op_type, ins, outs):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def test_each_op_is_dumped_with_its_in_op_figure_and_what_is_live_after_it(tmp_path, monkeypatch):
    T = {"input::x": {**_t([256]), "is_input": True}, "a": _t([256]), "b": _t([1024]), "c": _t([16])}
    O = [_op("neg::0", "aten::neg", ["input::x"], ["a"]), _op("neg::1", "aten::neg", ["a"], ["b"]),
         _op("sum::0", "aten::sum", ["b"], ["c"])]
    g = {"component_name": "toy", "tensors": T, "ops": {o["op_uid"]: o for o in O},
         "execution_order": [o["op_uid"] for o in O], "input_tensor_ids": ["input::x"],
         "output_tensor_ids": ["c"], "torch_dtype": "float32"}
    monkeypatch.setenv("NBX_PRISM_CURVE_DUMP", str(tmp_path))
    prof = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=4)
    files = list(tmp_path.iterdir())
    assert len(files) == 1, files
    lines = files[0].read_text().splitlines()
    head, rows = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    assert head["component"] == "toy" and head["peak_op"] == prof.peak_op_uid == "neg::1"
    assert [r[0] for r in rows] == ["neg::0", "neg::1", "sum::0"]
    # neg::1 holds a (1 KB) and b (4 KB) while it runs, then frees a: in-op 5 KB, after 4 KB
    assert rows[1][2] == (256 + 1024) * 4 and rows[1][3] == 1024 * 4, rows[1]
    assert head["peak_live"][0][:2] == ["b", 1024 * 4], head["peak_live"]


def test_nothing_is_written_without_the_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("NBX_PRISM_CURVE_DUMP", raising=False)
    g = {"component_name": "toy", "tensors": {"input::x": {**_t([4]), "is_input": True}, "a": _t([4])},
         "ops": {"neg::0": _op("neg::0", "aten::neg", ["input::x"], ["a"])}, "execution_order": ["neg::0"],
         "input_tensor_ids": ["input::x"], "output_tensor_ids": ["a"], "torch_dtype": "float32"}
    ActivationProfiler(g).estimate_peak_memory(dtype_bytes=4)
    assert not list(tmp_path.iterdir())
