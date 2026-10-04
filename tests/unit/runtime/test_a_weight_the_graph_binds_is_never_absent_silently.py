"""A weight the graph binds is never absent silently: both engines refuse it BY NAME before execution.

2026-10-04 02:10, Allegro on a staged container whose directory was replaced during the run
(`nbx/campaigns/2026_10_04_neurotax_ab/Allegro.b.log`), Triton engine:

    RuntimeError: Failed at aten.convolution::0 (aten::convolution): 'NoneType' object has no
    attribute 'ndim' | None args at positions [1, 2] of 9

The graph binds `param::post_quant_conv.weight` and `.bias`. The Triton loader globbed the shards
it found and never compared them with `weights_index.json`; a missing `weights/` directory was
`return {}`; the sequence's bind left the slots None without a word; the first reader met None.
The compiled loader had the same blindness (its index reader was never called, and Prism builds a
file-path shard map from the shards it finds on disk, so "load only the files in the map" skips a
missing one in silence).

The three absences, each refused naming the component, the key and the shard path expected:

  1. a shard the index lists, absent on disk;
  2. a key the index places in a shard whose header lacks it;
  3. a key the graph binds that the index lacks — at the loader when the load is asked for it,
     and at the WEIGHT BIND of both sequences, where a flow-injected weight (a tied head) and the
     trailing-suffix resolution have had their chance.

Every refusal test runs clean -> injected -> restored on the same container, the restored arm green.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES= python -m pytest tests/unit/runtime/test_a_weight_the_graph_binds_is_never_absent_silently.py
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from neurobrix.triton import weight_loader as W

COMP = "vae"
ARRAYS = {   # two shards, so a missing one leaves the other loadable
    "post_quant_conv.weight": (np.arange(16, dtype=np.float32).reshape(4, 4, 1, 1), "shard_000.safetensors"),
    "post_quant_conv.bias": (np.arange(4, dtype=np.float32) - 1.5, "shard_000.safetensors"),
    "decoder.conv_in.weight": (np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4, 1, 1),
                               "shard_001.safetensors"),
}


def _write_shard(path: Path, arrays: dict) -> None:
    """A safetensors file written byte by byte (torch-free): header, then the data."""
    header, blobs, off = {}, [], 0
    for k, a in arrays.items():
        b = np.ascontiguousarray(a).tobytes()
        header[k] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _container(root: Path, drop_from_shard=()) -> Path:
    comp = root / "components" / COMP
    (comp / "weights").mkdir(parents=True, exist_ok=True)
    shards: dict = {}
    for k, (a, s) in ARRAYS.items():
        if k not in drop_from_shard:
            shards.setdefault(s, {})[k] = a
    for s in sorted({s for _, s in ARRAYS.values()}):
        _write_shard(comp / "weights" / s, shards.get(s, {}))
    index = {"version": "1.0", "format": "nbx-weights-index", "component": COMP,
             "shards": {s: {} for s in sorted({s for _, s in ARRAYS.values()})},
             "tensors": {k: {"shard": s, "dtype": "float32", "shape": list(a.shape)}
                         for k, (a, s) in ARRAYS.items()}}
    (comp / "weights_index.json").write_text(json.dumps(index))
    # A container directory carries its manifest's model name (`refuse_misnamed`).
    (root / "manifest.json").write_text(json.dumps({"model_name": root.name}))
    return root


def _shard(root: Path, name="shard_000.safetensors") -> Path:
    return root / "components" / COMP / "weights" / name


# ---------------------------------------------------------------------------------------------
# The Triton loader. Its pre-flight runs before any device is touched, so a refusal is provable
# on a CPU; a complete container passes it and reaches the device — the first thing after it.
# ---------------------------------------------------------------------------------------------

class _ReachedTheDevice(Exception):
    pass


def _triton_load(root, monkeypatch, only=None):
    def _device(*a, **k):
        raise _ReachedTheDevice
    monkeypatch.setattr(W.DeviceAllocator, "set_device", staticmethod(_device))
    return W.load_component_weights(str(root), COMP, 0, only=only)


def _triton_passes(root, monkeypatch, only=None):
    with pytest.raises(_ReachedTheDevice):
        _triton_load(root, monkeypatch, only)


def _triton_refuses(root, monkeypatch, *needles, only=None):
    with pytest.raises(W.AbsentWeightError) as e:
        _triton_load(root, monkeypatch, only)
    msg = str(e.value)
    for n in (f"component '{COMP}'",) + needles:
        assert n in msg, f"the refusal does not name {n!r}:\n{msg}"
    return msg


def test_triton_a_shard_the_index_lists_absent_on_disk_is_refused_by_name(tmp_path, monkeypatch):
    root = _container(tmp_path)
    _triton_passes(root, monkeypatch)                                        # clean
    held = tmp_path / "held"; shutil.move(str(_shard(root)), str(held))      # injected
    _triton_refuses(root, monkeypatch, "post_quant_conv.weight", "post_quant_conv.bias",
                    str(_shard(root)), "absent on disk")
    shutil.move(str(held), str(_shard(root)))                                # restored
    _triton_passes(root, monkeypatch)


def test_triton_the_whole_weights_directory_absent_is_refused_not_returned_empty(tmp_path, monkeypatch):
    # The Allegro shape: the staged directory was replaced and `weights/` was gone; the loader
    # returned {} and the bind left every slot None.
    root = _container(tmp_path)
    wdir = root / "components" / COMP / "weights"
    held = tmp_path / "held"; shutil.move(str(wdir), str(held))
    _triton_refuses(root, monkeypatch, "post_quant_conv.weight", str(_shard(root)), "absent on disk")
    shutil.move(str(held), str(wdir))
    _triton_passes(root, monkeypatch)


def test_triton_a_key_the_index_places_in_a_shard_whose_header_lacks_it_is_refused(tmp_path, monkeypatch):
    root = _container(tmp_path)
    _triton_passes(root, monkeypatch)
    _container(root, drop_from_shard={"post_quant_conv.bias"})              # index untouched
    msg = _triton_refuses(root, monkeypatch, "post_quant_conv.bias", str(_shard(root)),
                          "header holds no such key")
    assert "post_quant_conv.weight" not in msg, "the present weight is named as absent"
    _container(root)
    _triton_passes(root, monkeypatch)


def test_triton_a_key_asked_for_that_the_index_lacks_is_refused(tmp_path, monkeypatch):
    root = _container(tmp_path)
    _triton_passes(root, monkeypatch, only={"post_quant_conv.weight"})
    _triton_refuses(root, monkeypatch, "post_quant_conv.scale",
                    str(root / "components" / COMP / "weights_index.json"),
                    "index lists no such key", only={"post_quant_conv.weight", "post_quant_conv.scale"})
    _triton_passes(root, monkeypatch, only={"post_quant_conv.weight"})


def test_triton_a_filtered_load_checks_only_what_it_asks_for(tmp_path, monkeypatch):
    # The consumed-weight filter: a shard holding only weights the load does not ask for may be
    # absent without a refusal (nothing will read them) — the door is about what is READ.
    root = _container(tmp_path)
    _shard(root, "shard_001.safetensors").unlink()
    _triton_passes(root, monkeypatch, only={"post_quant_conv.weight", "post_quant_conv.bias"})
    _triton_refuses(root, monkeypatch, "decoder.conv_in.weight",
                    str(_shard(root, "shard_001.safetensors")))


def test_triton_shards_without_an_index_are_refused(tmp_path, monkeypatch):
    root = _container(tmp_path)
    ip = root / "components" / COMP / "weights_index.json"
    held = ip.read_text(); ip.unlink()
    _triton_refuses(root, monkeypatch, "no index", str(ip))
    ip.write_text("{not json")
    _triton_refuses(root, monkeypatch, "cannot be read", str(ip))
    ip.write_text(held)
    _triton_passes(root, monkeypatch)


def test_triton_a_component_that_stores_no_weights_still_loads_nothing(tmp_path, monkeypatch):
    (tmp_path / "components" / COMP).mkdir(parents=True)
    assert _triton_load(tmp_path, monkeypatch) == {}


# ---------------------------------------------------------------------------------------------
# The compiled loader (CPU, end to end).
# ---------------------------------------------------------------------------------------------

def _torch():
    """The compiled engine's tests need torch; the triton ones above must not (a host without
    torch still proves the triton doors), so the skip is per test, never per module."""
    pytest.importorskip("safetensors")
    return pytest.importorskip("torch")


def _compiled_load(root, only=None, shard_map=None):
    torch = _torch()
    from neurobrix.core.io.weight_loader import WeightLoader
    with WeightLoader(str(root)) as loader:
        if shard_map is not None:
            return loader.load_component_with_shard_map(COMP, shard_map, torch.float32, only=only)
        return loader.load_component(COMP, "cpu", torch.float32, only=only)


def _compiled_complete(root, only=None, shard_map=None):
    got = _compiled_load(root, only, shard_map)
    want = set(ARRAYS) if only is None else set(only)
    assert set(got) == want, sorted(got)
    for k in want:
        np.testing.assert_array_equal(got[k].numpy(), ARRAYS[k][0])


def _compiled_refuses(root, *needles, only=None, shard_map=None):
    with pytest.raises(W.AbsentWeightError) as e:
        _compiled_load(root, only, shard_map)
    msg = str(e.value)
    for n in (f"component '{COMP}'",) + needles:
        assert n in msg, f"the refusal does not name {n!r}:\n{msg}"
    return msg


def _prism_map(root):
    """What Prism builds: a file-path map over the shards it FINDS on disk."""
    wdir = root / "components" / COMP / "weights"
    return {f"components/{COMP}/weights/{p.name}": "cpu" for p in sorted(wdir.glob("*.safetensors"))}


def test_compiled_a_shard_the_index_lists_absent_on_disk_is_refused_by_name(tmp_path):
    root = _container(tmp_path)
    _compiled_complete(root)
    _compiled_complete(root, shard_map=_prism_map(root))
    held = tmp_path / "held"; shutil.move(str(_shard(root)), str(held))
    _compiled_refuses(root, "post_quant_conv.weight", str(_shard(root)), "absent on disk")
    # The map Prism builds now lacks the shard: loading "only the files in the map" skipped it.
    _compiled_refuses(root, "post_quant_conv.weight", str(_shard(root)), "absent on disk",
                      shard_map=_prism_map(root))
    shutil.move(str(held), str(_shard(root)))
    _compiled_complete(root)
    _compiled_complete(root, shard_map=_prism_map(root))


def test_compiled_a_key_the_index_places_in_a_shard_whose_header_lacks_it_is_refused(tmp_path):
    root = _container(tmp_path)
    _compiled_complete(root)
    _container(root, drop_from_shard={"post_quant_conv.bias"})
    _compiled_refuses(root, "post_quant_conv.bias", str(_shard(root)), "header holds no such key")
    _container(root)
    _compiled_complete(root)


def test_compiled_a_key_asked_for_that_the_index_lacks_is_refused(tmp_path):
    root = _container(tmp_path)
    _compiled_complete(root, only={"post_quant_conv.weight"})
    _compiled_refuses(root, "post_quant_conv.scale", "index lists no such key",
                      only={"post_quant_conv.weight", "post_quant_conv.scale"})
    _compiled_complete(root, only={"post_quant_conv.weight"})


def test_compiled_a_key_the_load_did_not_deliver_is_refused(tmp_path, monkeypatch):
    # The post-load door: the pre-flight read the headers, the load reads the files again — a
    # file replaced in between delivers less than the index says (the Allegro race).
    from neurobrix.core.io.weight_loader import WeightLoader
    root = _container(tmp_path)
    _compiled_complete(root)
    real = WeightLoader._load_weight_file

    def _short(self, zip_path, device, dtype):
        out = real(self, zip_path, device, dtype)
        out.pop("post_quant_conv.bias", None)
        return out
    monkeypatch.setattr(WeightLoader, "_load_weight_file", _short)
    _compiled_refuses(root, "post_quant_conv.bias", str(_shard(root)), "did not deliver")
    monkeypatch.setattr(WeightLoader, "_load_weight_file", real)
    _compiled_complete(root)


# ---------------------------------------------------------------------------------------------
# The weight bind of both sequences: a weight the graph binds that no loaded tensor fills.
# ---------------------------------------------------------------------------------------------

def _dag():
    t = {
        "input::x": {"shape": [1, 4, 5, 5], "dtype": "float32", "is_input": True, "input_name": "x"},
        "param::post_quant_conv.weight": {"shape": [4, 4, 1, 1], "dtype": "float32",
                                          "is_parameter": True, "weight_name": "post_quant_conv.weight"},
        "param::post_quant_conv.bias": {"shape": [4], "dtype": "float32",
                                        "is_parameter": True, "weight_name": "post_quant_conv.bias"},
        "aten.convolution::0::out_0": {"shape": [1, 4, 5, 5], "dtype": "float32"},
    }
    ops = {"aten.convolution::0": {
        "op_type": "aten::convolution",
        "input_tensor_ids": ["input::x", "param::post_quant_conv.weight", "param::post_quant_conv.bias"],
        "output_tensor_ids": ["aten.convolution::0::out_0"],
        "attributes": {"args": [{"type": "tensor", "tensor_id": "input::x"},
                                {"type": "tensor", "tensor_id": "param::post_quant_conv.weight"},
                                {"type": "tensor", "tensor_id": "param::post_quant_conv.bias"},
                                [1, 1], [0, 0], [1, 1], False, [0, 0], 1], "kwargs": {}}}}
    return {"component_name": COMP, "tensors": t, "ops": ops, "execution_order": ["aten.convolution::0"],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["aten.convolution::0::out_0"]}


def test_the_consumers_are_the_container_weights_an_op_reads():
    dag = _dag()
    dag["tensors"]["param::constant_T_000001"] = {"constant": True, "constant_data": "x",
                                                   "is_parameter": True, "weight_name": "constant_T_000001"}
    dag["tensors"]["buffer::pos_embed"] = {"is_computable": True, "weight_name": "pos_embed"}
    dag["tensors"]["param::unrouted.expert"] = {"is_parameter": True, "weight_name": "unrouted.expert"}
    dag["ops"]["aten.convolution::0"]["input_tensor_ids"] += ["param::constant_T_000001", "buffer::pos_embed"]
    assert W.loader_weight_consumers(dag) == {
        "param::post_quant_conv.weight": "aten.convolution::0",
        "param::post_quant_conv.bias": "aten.convolution::0"}, (
        "a graph constant, a computed buffer or a weight no op reads is not the loader's to deliver")


def test_compiled_bind_refuses_a_weight_the_graph_binds_and_nothing_fills():
    torch = _torch()
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    full = {"param::post_quant_conv.weight": torch.ones(4, 4, 1, 1),
            "param::post_quant_conv.bias": torch.zeros(4)}
    seq = CompiledSequence(_dag(), torch.device("cpu"), torch.float32)
    seq.compile()
    seq.bind_weights(full)                                                          # clean
    seq = CompiledSequence(_dag(), torch.device("cpu"), torch.float32)
    seq.compile()
    with pytest.raises(W.AbsentWeightError) as e:                                   # injected
        seq.bind_weights({"param::post_quant_conv.weight": full["param::post_quant_conv.weight"]})
    for n in (f"component '{COMP}'", "param::post_quant_conv.bias", "aten.convolution::0"):
        assert n in str(e.value), str(e.value)
    assert "post_quant_conv.weight " not in str(e.value).split("\n", 1)[1]
    seq.bind_weights(full)                                                          # restored
    slot = seq._tensor_id_to_slot["param::post_quant_conv.bias"]
    assert seq._arena[slot] is full["param::post_quant_conv.bias"]


def test_triton_bind_refuses_a_weight_the_graph_binds_and_nothing_fills(monkeypatch):
    from neurobrix.triton.sequence import TritonSequence

    class _W:          # a weight handle: the bind reads `.ndim` only for pre-transposed weights
        ndim = 4
    full = {"post_quant_conv.weight": _W(), "post_quant_conv.bias": _W()}
    seq = TritonSequence(_dag(), 0)
    seq.compile()
    monkeypatch.setattr(seq, "compute_op_devices", lambda: None)   # placement, after the door
    seq.bind_weights(full)                                                          # clean
    with pytest.raises(W.AbsentWeightError) as e:                                   # injected
        seq.bind_weights({"post_quant_conv.weight": full["post_quant_conv.weight"]})
    for n in (f"component '{COMP}'", "param::post_quant_conv.bias", "aten.convolution::0"):
        assert n in str(e.value), str(e.value)
    # A weight stored under a shorter name (Wan's text encoder) is resolved by the bind's
    # trailing-suffix rule before the door: not refused.
    seq.bind_weights({"post_quant_conv.weight": full["post_quant_conv.weight"], "bias": _W()})
    seq.bind_weights(full)                                                          # restored
    assert seq._arena[seq._tid_to_slot["param::post_quant_conv.bias"]] is full["post_quant_conv.bias"]


def test_compiled_bind_refuses_before_its_defaults_fill_a_missing_norm():
    # The compiled bind fills an empty `.norm.` slot with ones/zeros. A norm weight the container
    # lacks must be refused BEFORE that, as --triton refuses it (it has no such default) — R30.
    torch = _torch()
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    dag = _dag()
    for old, new in (("post_quant_conv.weight", "block.0.norm.weight"),
                     ("post_quant_conv.bias", "block.0.norm.bias")):
        dag["tensors"]["param::" + new] = dict(dag["tensors"].pop("param::" + old), weight_name=new)
        op = dag["ops"]["aten.convolution::0"]
        op["input_tensor_ids"] = [("param::" + new) if t == "param::" + old else t for t in op["input_tensor_ids"]]
        for a in op["attributes"]["args"]:
            if isinstance(a, dict) and a.get("tensor_id") == "param::" + old:
                a["tensor_id"] = "param::" + new
    seq = CompiledSequence(dag, torch.device("cpu"), torch.float32)
    seq.compile()
    with pytest.raises(W.AbsentWeightError) as e:
        seq.bind_weights({"param::block.0.norm.weight": torch.ones(4, 4, 1, 1)})
    assert "param::block.0.norm.bias" in str(e.value), str(e.value)
    seq.bind_weights({"param::block.0.norm.weight": torch.ones(4, 4, 1, 1),
                      "param::block.0.norm.bias": torch.zeros(4)})


def test_triton_sequential_refuses_a_weight_the_graph_binds_and_the_store_lacks():
    # The op-by-op triton engine builds its own store from the weights; its door is the same
    # function. The clean arm goes on past the door (and then needs a device, which is not this
    # test's business): what it must NOT do is refuse.
    from neurobrix.core.runtime.graph_executor import GraphExecutor

    def _run(weights):
        ex = GraphExecutor.__new__(GraphExecutor)
        ex.mode, ex.dtype, ex._dag, ex._weights = "triton_sequential", "float32", _dag(), weights
        ex.precision_contract = lambda d: (False, (), ())
        ex._resolve_config_constants = lambda: set()
        ex._run_triton_sequential({}, 0)

    full = {"post_quant_conv.weight": object(), "post_quant_conv.bias": object()}
    for weights, refused in ((full, False),
                             ({"post_quant_conv.weight": full["post_quant_conv.weight"]}, True),
                             (full, False)):
        try:
            _run(weights)
            err = None
        except Exception as e:      # the clean arm fails later, at the device
            err = e
        assert isinstance(err, W.AbsentWeightError) is refused, repr(err)
        if refused:
            for n in (f"component '{COMP}'", "param::post_quant_conv.bias", "aten.convolution::0"):
                assert n in str(err), str(err)
