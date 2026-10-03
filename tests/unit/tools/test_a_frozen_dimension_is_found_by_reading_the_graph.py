"""The frozen-dimension scan names a dimension the trace left at its own value — and only that.

Built from the named cases of 2026-10-03: mochi's rotary table viewed at the trace's 4 050 tokens
(`aten.view::17`, a literal where the graph itself carries `s1*((s2-2)//2+1)*((s3-2)//2+1)`),
Qwen3-VL's expert view annotated at 230 = s0*s1. On a scan that read nothing, counted weights,
matched a batch-at-1 product, lost the origin of a propagated literal, accepted an empty cache or
left no file behind, these fail.

Run: python -m pytest tests/unit/tools/test_a_frozen_dimension_is_found_by_reading_the_graph.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "frozen_dim_scan.py"


def _tool():
    spec = importlib.util.spec_from_file_location("frozen_dim_scan", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def SYM(i, t):
    return {"type": "symbol", "id": i, "trace": t}


def MUL(a, b, t):
    return {"type": "mul", "left": a, "right": b, "trace": t}


SYMBOLS = {"s0": {"name": "batch", "trace_value": 1, "source": "input::x::dim_0"},
           "s1": {"name": "seq_len", "trace_value": 23, "source": "input::x::dim_1"}}


def _t(tid, dims, producer=None, consumers=(), param=False, const=False, weight_name=None):
    concrete = [d if isinstance(d, int) else d["trace"] for d in dims]
    return {"tensor_id": tid, "shape": concrete, "symbolic_shape": {"dims": dims, "concrete": concrete},
            "producer_op_uid": producer, "consumer_op_uids": list(consumers), "is_parameter": param,
            "is_input": tid.startswith("input::"), "constant": const, "weight_name": weight_name}


def _op(uid, op_type, ins, outs, args):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": ins, "output_tensor_ids": outs,
            "attributes": {"args": args, "kwargs": {}}}


def _graph(frozen_view=True, weight_extent=64, batch_product=False):
    """input x [s0, s1, 64] -> view (args [s0*s1 | 23, 64]) -> mm with w [64, weight_extent]
    -> relu (inherits the view's literal, if any)."""
    flat = MUL(SYM("s0", 1), SYM("s1", 23), 23)
    view_dim = 23 if frozen_view else flat
    tensors = {
        "input::x": _t("input::x", [SYM("s0", 1), SYM("s1", 23), 64], consumers=["aten.view::0"]),
        "aten.view::0::out_0": _t("aten.view::0::out_0", [view_dim, 64], "aten.view::0", ["aten.mm::0"]),
        "param::w": _t("param::w", [64, weight_extent], consumers=["aten.mm::0"], param=True, weight_name="w"),
        "aten.mm::0::out_0": _t("aten.mm::0::out_0", [view_dim, weight_extent], "aten.mm::0", ["aten.relu::0"]),
        "aten.relu::0::out_0": _t("aten.relu::0::out_0", [view_dim, weight_extent], "aten.relu::0"),
    }
    ops = {
        "aten.view::0": _op("aten.view::0", "aten::view", ["input::x"], ["aten.view::0::out_0"],
                            [{"type": "tensor", "tensor_id": "input::x"}, {"type": "list", "value": [view_dim, 64]}]),
        "aten.mm::0": _op("aten.mm::0", "aten::mm", ["aten.view::0::out_0", "param::w"], ["aten.mm::0::out_0"],
                          [{"type": "tensor", "tensor_id": "aten.view::0::out_0"}, {"type": "tensor", "tensor_id": "param::w"}]),
        "aten.relu::0": _op("aten.relu::0", "aten::relu", ["aten.mm::0::out_0"], ["aten.relu::0::out_0"],
                            [{"type": "tensor", "tensor_id": "aten.mm::0::out_0"}]),
    }
    if batch_product:
        # an expression through batch-at-1: s0*48 is 48 at the trace; a literal 48 elsewhere
        tensors["aten.full::0::out_0"] = _t("aten.full::0::out_0", [MUL(SYM("s0", 1), 48, 48)], "aten.full::0", ["aten.add::0"])
        tensors["aten.add::0::out_0"] = _t("aten.add::0::out_0", [48], "aten.add::0")
        ops["aten.full::0"] = _op("aten.full::0", "aten::full", [], ["aten.full::0::out_0"], [])
        ops["aten.add::0"] = _op("aten.add::0", "aten::add", ["aten.full::0::out_0"], ["aten.add::0::out_0"], [])
    return {"symbolic_context": {"symbols": SYMBOLS, "expressions": {}}, "tensors": tensors, "ops": ops,
            "execution_order": list(ops), "output_tensor_ids": ["aten.relu::0::out_0"]}


def _cache(tmp_path, graphs: dict, name="model-a"):
    root = tmp_path / "cache"
    c = root / name
    for comp, g in graphs.items():
        d = c / "components" / comp
        d.mkdir(parents=True)
        (d / "graph.json").write_text(g if isinstance(g, str) else json.dumps(g))
    (c / "manifest.json").write_text(json.dumps({"model_name": name}))
    (c / "topology.json").write_text(json.dumps({"components": {}, "extracted_values": {}}))
    return root


def _read(out: Path):
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ what it finds

def test_a_frozen_activation_dim_is_reported_at_its_origin():
    m = _tool()
    hits, _ = m.scan_graph(_graph(frozen_view=True))
    frozen = [h for h in hits if h["class"] == m.FROZEN]
    assert {h["tensor"] for h in frozen} == {"aten.view::0::out_0", "aten.mm::0::out_0", "aten.relu::0::out_0"}
    origin = [h for h in frozen if not h["inherited"]]
    assert [(h["producer"], h["value"], h["position"], h["in_args"]) for h in origin] == [("aten.view::0", 23, 0, True)]
    assert {h["origin"] for h in frozen} == {"aten.view::0"}          # downstream literals inherit it
    assert origin[0]["matches"].startswith("s1")
    assert origin[0]["dropped"]                                      # x carries s1 = 23, the view writes 23
    assert not any(h["dropped"] for h in frozen if h["inherited"])


def test_a_symbolic_graph_is_clean():
    m = _tool()
    hits, _ = m.scan_graph(_graph(frozen_view=False))
    assert hits == []


def test_a_weight_dim_equal_to_a_symbol_value_is_not_reported():
    """A weight [64, 23]: the weight itself is never a hit, and an activation carrying 23 next to
    it is AMBIGUOUS (a model constant can coincide), never FROZEN."""
    m = _tool()
    hits, _ = m.scan_graph(_graph(frozen_view=False, weight_extent=23))
    assert not any(h["tensor"] == "param::w" for h in hits)
    assert hits and all(h["class"] == m.AMBIGUOUS and "weight extent" in h["also"] for h in hits)


def test_a_product_through_batch_at_one_is_ambiguous_never_frozen():
    m = _tool()
    hits, _ = m.scan_graph(_graph(frozen_view=False, batch_product=True))
    h48 = [h for h in hits if h["value"] == 48]
    assert h48 and all(h["class"] == m.AMBIGUOUS and "R39" in h["also"] for h in h48)


def test_the_symbol_id_encoding_is_read():
    """Some arguments name a symbol as `symbol_id` / `trace_value` (Sana-4K's text-encoder arange)."""
    m = _tool()
    node = {"type": "mul", "left": {"type": "symbol", "symbol_id": "s1", "trace_value": 23}, "right": 10}
    assert m.expr_trace(node, {"s1": 23}) == 230
    assert m._symbol_ids(node) == {"s1"}


# ------------------------------------------------------------------ the contract: inputs and the file

def test_an_empty_cache_is_refused_by_name_and_the_file_is_written(tmp_path):
    m = _tool()
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "stray").mkdir()                                        # a directory with no manifest
    out = tmp_path / "o" / "scan.jsonl"
    assert m.main(["--cache", str(empty), "--out", str(out)]) == 2
    rec = _read(out)
    assert len(rec) == 1 and "holds no container" in rec[0]["refused"] and str(empty) in rec[0]["refused"]


def test_a_missing_cache_and_an_unknown_model_are_refused_by_name(tmp_path):
    m = _tool()
    out = tmp_path / "scan.jsonl"
    assert m.main(["--cache", str(tmp_path / "nowhere"), "--out", str(out)]) == 2
    assert "nowhere" in _read(out)[0]["refused"]
    root = _cache(tmp_path, {"model": _graph()})
    assert m.main(["--cache", str(root), "--models", "model-a,not-a-model", "--out", str(out)]) == 2
    assert "not-a-model" in _read(out)[0]["refused"]


def test_the_output_is_written_for_a_scan_and_for_an_unreadable_graph(tmp_path):
    m = _tool()
    root = _cache(tmp_path, {"model": _graph(), "broken": "{ not json"})
    out = tmp_path / "scan.jsonl"
    assert m.main(["--cache", str(root), "--out", str(out)]) == 0
    (rec,) = _read(out)
    assert rec["container"] == "model-a" and rec["field"] == m.FIELD
    assert "unreadable" in rec["components"]["broken"]
    assert rec["verdict"] == "UNREADABLE"                            # never read as clean
    assert rec["counts"]["FROZEN_patterns"] == 1 and rec["patterns"][0]["first"] == "aten.view::0"
    assert (tmp_path / "scan_sites.jsonl").read_text().strip()
    assert (tmp_path / "scan_summary.txt").read_text().count("model-a") >= 1
