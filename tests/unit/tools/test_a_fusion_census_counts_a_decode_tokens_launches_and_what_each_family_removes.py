"""The fusion census counts a decode token's kernel launches from the container's graph, and what each family of
fused kernels would remove.

`tools/fusion_census.py` prepares a decoder's graph with the engine's own passes (no device, no weight) and counts
what each op launches for one decoded token. These tests hand it a two-layer decoder written here, small enough to
count by hand:

    per layer   cast to float32, norm                     2 launches (one group of F-N)
                q, k, v products                          3
                k and v repeated to the query heads       2 strided copies
                self-attention under the paged cache      6 (2 regroup copies, append, advance, split, reduce)
                layout copy of one position               a device copy, no kernel
                output product, residual add              2
                                                         15 a layer, 30 a token
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "fusion_census.py"
_spec = importlib.util.spec_from_file_location("fusion_census", _TOOL)
census = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = census
_spec.loader.exec_module(census)

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 8, 2, 1, 4


class _Graph:
    """A decoder graph in the container's format, written op by op."""

    def __init__(self):
        self.tensors, self.ops, self.order = {}, {}, []
        self._n = {}

    def tensor(self, tid, shape, dtype="bfloat16", parameter=False, is_input=False):
        self.tensors[tid] = {
            "tensor_id": tid, "shape": list(shape), "dtype": dtype, "device": "cuda", "producer_op_uid": None,
            "output_index": None, "consumer_op_uids": [], "is_parameter": parameter, "is_input": is_input,
            "weight_name": tid.split("::", 1)[1] if parameter else None,
            "input_name": tid.split("::", 1)[1] if is_input else None, "output_name": None,
            "symbolic_shape": {"dims": list(shape), "concrete": list(shape)}}
        return tid

    def op(self, kind, inputs, shape, dtype="bfloat16", extra=(), kwargs=None, namespace="aten"):
        n = self._n.get(kind, 0)
        self._n[kind] = n + 1
        uid, out = f"{namespace}.{kind}::{n}", f"{namespace}.{kind}::{n}::out_0"
        self.tensor(out, shape, dtype)
        self.tensors[out]["producer_op_uid"], self.tensors[out]["output_index"] = uid, 0
        for t in inputs:
            self.tensors[t]["consumer_op_uids"].append(uid)
        self.ops[uid] = {
            "op_uid": uid, "op_type": f"{namespace}::{kind}", "input_tensor_ids": list(inputs),
            "output_tensor_ids": [out], "input_shapes": [self.tensors[t]["shape"] for t in inputs],
            "output_shapes": [list(shape)], "input_dtypes": [self.tensors[t]["dtype"] for t in inputs],
            "output_dtypes": [dtype], "device": "cuda",
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in inputs] + list(extra),
                           "kwargs": dict(kwargs or {})}}
        self.order.append(uid)
        return out

    def dag(self, inputs, outputs):
        return {"component_name": "model", "tensors": self.tensors, "ops": self.ops, "execution_order": self.order,
                "input_tensor_ids": list(inputs), "output_tensor_ids": list(outputs), "forks": [], "joins": [],
                "symbolic_context": {}, "format": "tensor_dag", "version": "0.1", "torch_dtype": "bfloat16"}


def _scalar(v):
    return {"type": "scalar", "value": v}


def _shape(*dims):
    return {"type": "list", "value": list(dims)}


def _decoder(layers: int) -> dict:
    g = _Graph()
    x = g.tensor("input::inputs_embeds", [1, 1, HIDDEN], is_input=True)
    kv_width = KV_HEADS * HEAD_DIM
    group = HEADS // KV_HEADS
    for i in range(layers):
        w = {name: g.tensor(f"param::block.{i}.{name}", shape, parameter=True) for name, shape in (
            ("norm", [HIDDEN]), ("q", [HIDDEN, HIDDEN]), ("k", [HIDDEN, kv_width]), ("v", [HIDDEN, kv_width]),
            ("o", [HIDDEN, HIDDEN]))}
        wide = g.op("_to_copy", [x], [1, 1, HIDDEN], "float32",
                    kwargs={"dtype": {"type": "dtype", "value": "torch.float32"}})
        normed = g.op("rms_norm", [wide, w["norm"]], [1, 1, HIDDEN], kwargs={"epsilon": 1e-5}, namespace="custom")
        rows = g.op("view", [normed], [1, HIDDEN], extra=[_shape(1, HIDDEN)])
        q = g.op("mm", [rows, w["q"]], [1, HIDDEN])
        k = g.op("mm", [rows, w["k"]], [1, kv_width])
        v = g.op("mm", [rows, w["v"]], [1, kv_width])
        q = g.op("view", [q], [1, 1, HEADS, HEAD_DIM], extra=[_shape(1, 1, HEADS, HEAD_DIM)])
        q = g.op("transpose", [q], [1, HEADS, 1, HEAD_DIM], extra=[_scalar(1), _scalar(2)])
        repeated = []
        for t in (k, v):
            t = g.op("view", [t], [1, 1, KV_HEADS, HEAD_DIM], extra=[_shape(1, 1, KV_HEADS, HEAD_DIM)])
            t = g.op("transpose", [t], [1, KV_HEADS, 1, HEAD_DIM], extra=[_scalar(1), _scalar(2)])
            t = g.op("unsqueeze", [t], [1, KV_HEADS, 1, 1, HEAD_DIM], extra=[_scalar(2)])
            t = g.op("expand", [t], [1, KV_HEADS, group, 1, HEAD_DIM], extra=[_shape(1, KV_HEADS, group, 1, HEAD_DIM)])
            t = g.op("clone", [t], [1, KV_HEADS, group, 1, HEAD_DIM])
            repeated.append(g.op("view", [t], [1, HEADS, 1, HEAD_DIM], extra=[_shape(1, HEADS, 1, HEAD_DIM)]))
        attended = g.op("scaled_dot_product_attention", [q] + repeated, [1, HEADS, 1, HEAD_DIM])
        back = g.op("transpose", [attended], [1, 1, HEADS, HEAD_DIM], extra=[_scalar(1), _scalar(2)])
        back = g.op("clone", [back], [1, 1, HEADS, HEAD_DIM])
        back = g.op("view", [back], [1, HIDDEN], extra=[_shape(1, HIDDEN)])
        out = g.op("mm", [back, w["o"]], [1, HIDDEN])
        out = g.op("view", [out], [1, 1, HIDDEN], extra=[_shape(1, 1, HIDDEN)])
        x = g.op("add", [x, out], [1, 1, HIDDEN])
    return g.dag(["input::inputs_embeds"], [x])


def _container(path: Path, layers: int = 2, state: bool = True) -> Path:
    topology = {"flow": {"type": "autoregressive_generation", "generation": {"lm_component": "model"}},
                "state": {"model": [{"kind": "kv", "layers": layers, "kv_heads": KV_HEADS, "k_dim": HEAD_DIM,
                                     "v_dim": HEAD_DIM}]} if state else {}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps({"family": "llm"}))
        z.writestr("topology.json", json.dumps(topology))
        z.writestr("components/model/graph.json", json.dumps(_decoder(layers)))
    return path


@pytest.fixture(scope="module")
def counted(tmp_path_factory):
    return census.census_of(_container(tmp_path_factory.mktemp("zoo") / "small.nbx"))["decoders"][0]


def test_a_decode_token_launches_what_its_ops_launch(counted):
    assert counted["layers"] == 2
    assert counted["launches"] == 30
    assert counted["per_layer"] == 15
    assert counted["device_copies"] == 2                  # the one-position layout copy of each layer: no kernel
    kernels = counted["by_kernel"]
    assert kernels["mm"] == 8 and kernels["rms_norm"] == 2 and kernels["add"] == 2
    assert kernels["strided_copy"] == 4                   # k and v repeated to the query heads
    for kernel in census.PAGED_REGROUP_COPIES + census.PAGED_DECODE_KERNELS:
        assert kernels[kernel] == 2, kernel


def test_the_literal_families_fuse_nothing_where_a_view_or_a_copy_sits_between_two_ops(counted):
    after = counted["readings"]["strict"]["after"]
    assert after == {"F-N": 30, "F-R": 30, "F-S": 30, "F-E": 30, "all": 30}


def test_a_family_takes_the_casts_and_copies_inside_its_chain(counted):
    wide = counted["readings"]["wide"]
    assert wide["after"]["F-N"] == 28                     # cast + norm: one launch a layer
    assert wide["patterns"]["F-N"] == {"_to_copy > rms_norm": 2}
    assert wide["after"]["F-S"] == 22                     # two repeats, two regroup copies: gone; four kernels stay
    assert wide["after"]["F-R"] == 30 and wide["after"]["F-E"] == 30
    assert wide["after"]["all"] == 20
    assert wide["left"] == {"mm": 8, "add": 2}
    assert wide["unfused_chains"] == {"mm > add": 2}      # the next family: a residual add behind its product


def test_a_group_beside_a_matrix_product_folds_into_its_launch(counted):
    anchored = counted["readings"]["anchored"]
    assert anchored["after"]["F-N"] == 26                 # the norm formed on the read of q, k, v
    assert anchored["after"]["F-E"] == 28                 # the residual add as the output product's epilogue
    assert anchored["after"]["all"] == 16


def test_a_value_read_twice_ends_its_group(tmp_path):
    graph = census.decode_graph_of(census.Container(_container(tmp_path / "small.nbx")), "model", "llm", {},
                                   {"kv_heads": KV_HEADS})
    norm = next(op for op in graph.ops if op.name == "rms_norm")
    readers, clean = graph.sinks(norm, census.ORDER_KEEPING_VIEWS)
    assert clean and [graph.ops[r].name for r in readers] == ["mm", "mm", "mm"]
    wide = census.groups_of(graph, census.FAMILY["F-N"], "wide")
    assert all(group.last.name == "rms_norm" for group in wide)


def test_a_container_without_a_cached_decoder_is_refused_by_name(tmp_path):
    with pytest.raises(census.Refused, match="no key/value cache state"):
        census.census_of(_container(tmp_path / "stateless.nbx", state=False))


def test_the_table_has_one_row_per_decoder_and_names_a_refusal(tmp_path, capsys):
    good, bad = _container(tmp_path / "small.nbx"), _container(tmp_path / "stateless.nbx", state=False)
    out_json = tmp_path / "census.json"
    assert census.main([str(good), str(bad), "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    assert "| small | model | 2 | 15 | 30 | 30 | 22 | 28 | 30 | 20 |" in printed
    assert "| stateless | - | - | - | refused:" in printed
    rows = json.loads(out_json.read_text())
    assert [("decoders" in r, "refused" in r) for r in rows] == [(True, False), (False, True)]
    assert census.main([str(bad)]) == 2


def test_a_cast_whose_value_already_holds_its_target_at_run_time_launches_nothing_under_an_activation_dtype(tmp_path):
    path = _container(tmp_path / "small.nbx")
    recorded = census.census_of(path)["decoders"][0]
    half = census.census_of(path, "float16")["decoders"][0]
    wide = census.census_of(path, "float32")["decoders"][0]
    assert recorded["by_kernel"]["copy (bfloat16 to float32)"] == 2       # the recorded dtypes: both casts launch
    # float16 run: the first layer's input is held in float16 (its cast launches), the second layer's input is
    # the sum of a float32 product and it (float32 already: the cast finds its target)
    assert half["launches"] == 29 and half["by_kernel"] == {
        **{k: v for k, v in recorded["by_kernel"].items() if not k.startswith("copy (")}, "copy (float16 to float32)": 1}
    assert wide["launches"] == 28 and not any(k.startswith("copy (") for k in wide["by_kernel"])


_TIMES = {"rms_norm_forward_kernel": {"launches": 2.0, "ms": 0.5}, "triton_copy_kernel": {"launches": 2.0, "ms": 9.0},
          "add_forward_kernel": {"launches": 2.0, "ms": 0.2}, "gemv_vec_kernel": {"launches": 8.0, "ms": 4.0}}


def test_a_kernel_times_file_ranks_each_chain_by_the_measured_ms_of_its_members(tmp_path, capsys):
    path = _container(tmp_path / "small.nbx")
    ranked = census.ranked_chains(path, _TIMES, "wide")
    norm = next(r for r in ranked if r["chain"] == "_to_copy > rms_norm")
    # one group per layer, two layers: a cast and a norm become one launch, and each member launch costs its
    # measured kernel's mean time (the norm 0.25 ms, the cast's copy 4.5 ms: 2 x (0.25 + 4.5) = 9.5)
    assert norm["count"] == 2 and norm["launches_saved"] == 2 and abs(norm["ms"] - 9.5) < 1e-9
    assert ranked == sorted(ranked, key=lambda r: (-r["ms"], -r["launches_saved"], r["chain"]))
    kernel_times = tmp_path / "times.json"
    kernel_times.write_text(json.dumps(_TIMES))
    assert census.main([str(path), "--kernel-times", str(kernel_times)]) == 0
    assert "chains by measured ms a token" in capsys.readouterr().out


def test_a_profiled_runs_step_table_reads_as_kernel_times(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("[step] CSE: ONE DECODE STEP = x\n"
                   "[step]   kernel                                           launches/token   ms/token   mean us\n"
                   "[step]   gemv_vec_kernel                                          155.00     4.8732      31.4\n"
                   "[step]   copy host to device                                        2.00    19.7086    9854.3\n")
    assert census.read_kernel_times(log) == {"gemv_vec_kernel": {"launches": 155.0, "ms": 4.8732},
                                             "copy host to device": {"launches": 2.0, "ms": 19.7086}}
