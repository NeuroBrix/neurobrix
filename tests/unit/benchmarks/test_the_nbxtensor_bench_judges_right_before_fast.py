"""The nbxtensor-vs-torch benchmark's judgement, card-less (benchmarks/nbxtensor_vs_torch): a faster wrong result is a
loss, never a win; the bound is the narrowest dtype the op reads, writes or is recorded writing; two ops differing
only in which tensors they read are one instance; the inputs are deterministic and an index stays in its extent."""
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[3] / "benchmarks" / "nbxtensor_vs_torch"
sys.path.insert(0, str(BENCH))

import instances as I  # noqa: E402
import summary as S  # noqa: E402

K = pytest.importorskip("kernel_bench")


def _row(key, contender, status, median, note=""):
    return {"key": key, "op_type": "aten::mm", "shapes": "[[2, 2], [2, 2]]", "dtypes": "float16,float16", "sites": 1,
            "contender": contender, "status": status, "median_ms": median, "note": note}


def test_a_faster_wrong_result_is_a_loss():
    rows = [_row("a", "torch_eager", "ok", "1.0"), _row("a", "torch_compile[default]", "ok", "0.8"),
            _row("a", "nbxtensor", "wrong", "0.1")]
    lost, won, total = S.kernel_losses(rows)
    assert total == 1 and won == 0 and lost[0][3] == "wrong"


def test_slower_than_the_best_right_torch_is_a_loss_and_a_wrong_torch_is_no_benchmark():
    rows = [_row("a", "torch_eager", "ok", "1.0"), _row("a", "torch_compile[default]", "wrong", "0.1"),
            _row("a", "nbxtensor", "ok", "1.2"),
            _row("b", "torch_eager", "ok", "1.0"), _row("b", "nbxtensor", "ok", "0.9")]
    lost, won, total = S.kernel_losses(rows)
    assert total == 2 and won == 1
    assert len(lost) == 1 and lost[0][1]["key"] == "a" and lost[0][0] == pytest.approx(1.2)


def test_the_bound_is_the_narrowest_dtype_read_written_or_recorded():
    assert K._bound(["float32", "float32", "float32"]) == K.TOLERANCE["float32"]
    assert K._bound(["float32", "float32", "bfloat16"]) == K.TOLERANCE["bfloat16"]
    assert K._bound(["int64", "bool"]) is None


def test_two_ops_reading_other_tensors_are_one_instance():
    a = [{"type": "tensor", "tensor_id": "x::0"}, {"type": "scalar", "value": 2}]
    b = [{"type": "tensor", "tensor_id": "y::7"}, {"type": "scalar", "value": 2}]
    c = [{"type": "tensor", "tensor_id": "y::7"}, {"type": "scalar", "value": 3}]
    assert I._literal(a) == I._literal(b) != I._literal(c)


def test_the_inputs_are_deterministic_and_an_index_stays_in_its_extent():
    inst = {"shapes": [[8, 16], [4], []], "dtypes": ["float16", "int64", "float32"]}
    x, y = K.host_inputs(inst), K.host_inputs(inst)
    assert all(np.array_equal(p, q) for p, q in zip(x, y))
    assert x[1].dtype == np.int64 and x[1].max() < 8 and x[2].shape == ()
    # an embedding's indices stay below its table's rows (02/11: [10, 32] x [1, 36000] drew up to 32)
    emb = K.host_inputs({"shapes": [[10, 32], [1, 36000]], "dtypes": ["float32", "int64"]})
    assert emb[1].max() < 10


def test_the_nbxtensor_side_runs_the_dtype_the_engine_runs_never_the_trace_record():
    # Allegro's patch-embedding conv as the Apple census derived it: a bf16 component whose graph recorded the op's
    # output as float16 (its trace dtype). The engine dispatches it at bf16 (census key ... 'bf16', 'bf16', 'bf16');
    # the bench once took the recorded float16 and timed an fp16 key no certificate holds, swept at runtime (03/11)
    inst = {"key": "ce761272518bca25", "op_type": "aten::convolution", "dtypes": ["bfloat16"] * 3,
            "output_dtypes": ["float16"], "op_uid": "aten.convolution::0", "compute_dtype": "bfloat16",
            "graph_dtype": "float16", "contract": {"safe": False, "island": None}}
    disp, facts = K.engine_dispatcher(inst, stores_fp64=False)
    assert disp.compute_dtype.name == "bfloat16" and facts["uid"] == "aten.convolution::0"
    # a calibration island is the engine's too: the op's uid reaches the contract
    _, isl = K.engine_dispatcher({**inst, "compute_dtype": "float16", "contract": {"safe": True, "island": "fp32"}},
                                 stores_fp64=False)
    assert isl["contract"] == (True, {"aten.convolution::0"}, set())
    # an instances file without the facts is refused, never guessed
    with pytest.raises(KeyError, match="compute_dtype"):
        K.engine_dispatcher({k: v for k, v in inst.items() if k != "compute_dtype"}, stores_fp64=False)


def test_a_video_instance_too_large_for_the_device_is_out_of_class_never_run():
    # a video VAE convolution at its whole request shape (Allegro, [1, 512, 22, 180, 320]): its operands and float64
    # reference exceed the op budget an 18 GB device grants — the row says so with its numbers, nothing runs
    inst = {"key": "v1", "op_type": "aten::convolution", "shapes": [[1, 512, 22, 180, 320], [512, 512, 3, 3, 3], [512]],
            "dtypes": ["bfloat16", "bfloat16", "bfloat16"], "sites": ["Allegro/vae/aten.convolution::3"],
            "output_shapes": [[1, 512, 20, 178, 318]]}
    profile = {"devices": [{"memory_mb": 18186}]}
    budget = K.device_budget_bytes(profile)
    assert 0 < budget < 18186 * 2**20                     # a share of the device, from the library's op budget fraction
    need = K.instance_bytes(inst)
    assert need > budget
    census = [{"model": "Allegro", "ops": ["vae/aten.convolution::3"], "kernel": "nbxtensor.ops.conv3d.conv3d_kernel",
               "key": "(1, 512, 6, 180, 320)"},
              {"model": "Allegro", "ops": [None], "kernel": "k", "key": "()"},     # the table holds such rows (02/11 21:47)
              {"model": None, "ops": ["x"], "kernel": "k", "key": "()"}]
    row = K.out_of_class_row(inst, need, budget, K.census_cut_keys(inst, census))
    assert row["status"] == "out_of_class" and "op budget" in row["note"] and "conv3d_kernel" in row["note"]
    small = {**inst, "shapes": [[1, 64, 2, 16, 16], [64, 64, 3, 3, 3], [64]], "output_shapes": [[1, 64, 1, 14, 14]]}
    assert K.instance_bytes(small) < budget             # an instance that fits is run whole


def test_the_reference_is_compared_tile_by_tile_never_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "TILE", 7)                    # tile edges inside the arrays
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(100)
    got = (ref + 1e-3 * rng.standard_normal(100)).astype(np.float32)
    np.save(tmp_path / "g.npy", got)
    mx, mean = K.compare_tiles(np.load(tmp_path / "g.npy", mmap_mode="r"), ref)
    whole = np.abs(got.astype(np.float64) - ref)
    assert mx == pytest.approx(whole.max() / np.abs(ref).max()) and mean == pytest.approx(whole.mean() / np.abs(ref).max())
    assert K.compare_tiles(got[:99], ref) == (float("inf"), float("inf"))


def test_the_torch_side_keeps_nothing_but_the_rows(tmp_path, monkeypatch):
    inst = {"key": "k1", "op_type": "aten::mm", "shapes": [[2, 2], [2, 2]], "dtypes": ["float32", "float32"],
            "sites": ["m/c/op"], "output_dtypes": ["float32"], "inputs": ["a", "b"], "args": [], "attributes": {},
            "symbols": {}}
    out = np.ones((2, 2), dtype=np.float32)
    nbx_file = tmp_path / "k1.npy"
    np.save(nbx_file, np.zeros(4, dtype=np.float32))      # the library's output: wrong
    monkeypatch.setattr(K, "reference", lambda *a, **k: np.ones((2, 2)))
    monkeypatch.setattr(K, "_to_numpy", lambda x: np.asarray(x, dtype=np.float64))
    monkeypatch.setattr(K, "run_torch", lambda *a, **k: (_Host(out), 3.0, [0.5, 0.6], 0.0, "test"))
    monkeypatch.setattr(K, "release_compile_caches", lambda dev: None)
    nbx = {"first": 1.0, "times": [0.4, 0.5], "peak": 0.0, "how": "test", "od": "float32", "file": str(nbx_file)}
    msgs = list(K.torch_work(inst, None, None, 0, 2, ["default"], nbx))
    rows = [m["row"] for m in msgs if "row" in m]
    # the nbxtensor row comes FIRST: a worker that dies later in a contender has already handed it
    assert rows[0]["contender"] == "nbxtensor" and msgs[1] == {"start": "torch_eager"}
    by = {r["contender"]: r for r in rows}
    assert by["torch_eager"]["status"] == "ok" and by["torch_eager"]["compile_ms"] == ""
    assert by["torch_compile[default]"]["compile_ms"] != ""          # the compile time, apart from the execution
    assert by["nbxtensor"]["status"] == "wrong" and by["nbxtensor"]["mean_err"] != ""
    assert list(tmp_path.iterdir()) == []                  # the library's output removed; no reference written


def test_every_row_reaches_the_dated_registry(tmp_path):
    import registry as R
    reg = R.Registry(tmp_path, "kernel", K.FIELDS, "default-test", BENCH.parents[1])
    reg.append({"key": "k1", "contender": "nbxtensor", "status": "ok", "median_ms": "0.5"})
    reg.append({"key": "k1", "contender": "torch_eager", "status": "ok", "median_ms": "0.4"})
    lines = (tmp_path / "nbxtensor_vs_torch.kernel.tsv").read_text().splitlines()
    assert lines[0].split("\t")[:3] == ["date", "device", "commit"] and len(lines) == 3
    assert all(l.split("\t")[1] == "default-test" and l.split("\t")[2] for l in lines[1:])
    assert R.Registry(None, "kernel", K.FIELDS, "x", BENCH.parents[1]).path is None
    # new columns: the file written under the old header is never rewritten; the rows continue in a .v2 file
    reg2 = R.Registry(tmp_path, "kernel", K.FIELDS + ["new_column"], "default-test", BENCH.parents[1])
    assert reg2.path.name == "nbxtensor_vs_torch.kernel.v2.tsv"
    reg2.append({"key": "k2", "new_column": "x"})
    assert (tmp_path / "nbxtensor_vs_torch.kernel.tsv").read_text().splitlines() == lines


class _Host:
    """A contender's output as the nbxtensor side holds it (numpy behind the NBXTensor surface _to_numpy reads)."""

    def __init__(self, a):
        self.a, self.dtype = a, "float32"

    def is_floating_point(self):
        return True

    def float(self):
        return self

    def numpy(self):
        return self.a


class _NoDevice:
    def release(self):
        pass


def test_an_instance_computes_at_the_plan_s_dtype_never_the_stored_one(monkeypatch):
    """Each component's instances are taken at the compute dtype the engine's plan gives it on the profile
    (`instances.plan_dtypes`, `derived_census.plan_record`: the census's own source), never the container's stored
    dtype: TinyLlama stores bfloat16, a V100 plan computes it in float16 (no bf16 there). What this test would do if
    the instances read the manifest: bfloat16 instances, keys the engine never forms on that card — RED. Card-less."""
    from tests.conftest import installed_container
    import derived_census as DC
    root = installed_container("TinyLlama-1.1B-Chat-v1.0")
    import json
    assert json.loads((root / "manifest.json").read_text())["dtype"] == "bfloat16"
    comps = sorted(p.name for p in (root / "components").iterdir() if (p / "graph.json").exists())
    asked = []
    monkeypatch.setattr(DC, "plan_record", lambda model, request, mode, hardware, rung: asked.append(
        (model, mode, hardware, rung)) or {"components": [{"name": c, "dtype": "float16"} for c in comps]})
    bound, _skipped = I._component_bindings("TinyLlama-1.1B-Chat-v1.0", "scenario-v100")
    assert asked == [("TinyLlama-1.1B-Chat-v1.0", "triton", "scenario-v100", None)]
    assert bound and {d for _c, _g, _s, d in bound} == {"float16"}


class _Stream:
    """A torch worker as the coordinator reads it: the messages it sends, then `end` (an exception) instead of more."""

    def __init__(self, msgs, end):
        self.msgs, self.end, self.sent = list(msgs), end, []

    def send(self, req):
        self.sent.append(req)

    def read(self, until=None):
        if self.msgs:
            return self.msgs.pop(0)
        raise self.end


_INST = {"key": "d1", "op_type": "aten::convolution", "shapes": [[2, 11200, 24, 40], [11200, 1, 3, 3]],
         "dtypes": ["float16", "float16"], "sites": ["m/c/op"]}


def test_a_worker_that_dies_in_a_contender_keeps_every_row_before_it():
    # the V100 run, 2026-11-03 04:23: a torch worker ended in max-autotune and the instance kept only
    # "torch_eager refused, the worker ended" — the nbxtensor row measured before it was lost. RED on 5e1301ad
    # (one answer per instance, none of it written when the worker died).
    nbx = {"row": {**K._row(_INST, "nbxtensor"), "status": "ok", "median_ms": "1.0"}}
    eager = {"row": {**K._row(_INST, "torch_eager"), "status": "ok", "median_ms": "2.0"}}
    wk = _Stream([nbx, {"start": "torch_eager"}, eager, {"start": "torch_compile[max-autotune]"}],
                 RuntimeError("the worker ended (rc -9)"))
    rows, ended, err = K.torch_rows(wk, _INST, {"file": "x"}, None)
    by = {r["contender"]: r for r in rows}
    assert not ended and err is not None
    assert by["nbxtensor"]["status"] == "ok" and by["torch_eager"]["median_ms"] == "2.0"
    assert by["torch_compile[max-autotune]"]["status"] == "refused"
    assert "worker ended" in by["torch_compile[max-autotune]"]["note"]


def test_a_compilation_the_run_s_end_stops_is_not_finished_never_a_loss():
    # the supervision, 2026-11-03 07:35: no hand-written bound — the run's own end; the contender it stops is "not
    # finished in the run" with its elapsed time, neither a loss nor a win, and the rows before it stand
    eager = {"row": {**K._row(_INST, "torch_eager"), "status": "ok", "median_ms": "2.0"}}
    wk = _Stream([{"start": "torch_eager"}, eager, {"start": "torch_compile[max-autotune]"}], K.RunEnded())
    rows, ended, err = K.torch_rows(wk, _INST, None, 0.0)
    by = {r["contender"]: r for r in rows}
    assert ended and err is None
    assert by["torch_eager"]["status"] == "ok"
    assert by["torch_compile[max-autotune]"]["status"] == "not_finished"
    assert "not finished in the run" in by["torch_compile[max-autotune]"]["note"]
    nbx = {**K._row(_INST, "nbxtensor"), "status": "ok", "median_ms": "1.0"}
    lost, won, total = S.kernel_losses([nbx, *rows])
    assert (len(lost), won, total) == (0, 1, 1)            # judged against the contenders that finished


def test_the_summary_counts_refusals_apart_from_losses():
    # the supervision, 2026-11-03 07:35: lost / refused (key outside the census) / refused (memory), apart, before
    # any "lost on N". RED on 5e1301ad: a KeyNotCertified refusal counted as a loss.
    def inst(key):
        return {"key": key, "op_type": "aten::mm", "shapes": "[[2, 2]]", "dtypes": "float16", "sites": 1}
    rows = [
        {**inst("a"), "contender": "nbxtensor", "status": "refused",
         "note": "KeyNotCertified: CERTIFIED-ONLY: no certified setting for conv2d_forward_kernel fp32"},
        {**inst("a"), "contender": "torch_eager", "status": "ok", "median_ms": "1.0"},
        {**inst("b"), "contender": "*", "status": "out_of_class", "note": "operands > the op budget"},
        {**inst("c"), "contender": "nbxtensor", "status": "ok", "median_ms": "3.0"},
        {**inst("c"), "contender": "torch_eager", "status": "ok", "median_ms": "1.0"},
        {**inst("d"), "contender": "nbxtensor", "status": "wrong", "median_ms": "0.1"},
        {**inst("d"), "contender": "torch_eager", "status": "ok", "median_ms": "1.0"},
    ]
    n = S.kernel_counts(rows)
    assert (n["lost"], n["won"], n["refused_key"], n["refused_memory"]) == (2, 0, 1, 1)


def test_an_input_the_graph_forms_as_a_view_reaches_the_op_as_that_view():
    # the Dell's V100 reading, 2026-11-03 01:58, verified on Allegro's graph: a linear's weight is aten.t(W), a
    # transposed VIEW; the bench gave a contiguous [K, N] array and addmv paid a copy per row the engine never makes.
    # The instance names the root and the view; both contenders replay it before the op. RED on 5e1301ad (no prelude).
    torch = pytest.importorskip("torch")
    from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher
    inst = {"key": "v1", "op_type": "aten::addmm", "inputs": ["bias", "x", "aten.t::1::out_0"],
            "shapes": [[3], [2, 4], [4, 3]], "dtypes": ["float32"] * 3, "symbols": {}, "attributes": {},
            "args": [{"type": "tensor", "tensor_id": "bias"}, {"type": "tensor", "tensor_id": "x"},
                     {"type": "tensor", "tensor_id": "aten.t::1::out_0"}],
            "roots": [{"id": "bias", "shape": [3], "dtype": "float32"}, {"id": "x", "shape": [2, 4], "dtype": "float32"},
                      {"id": "W", "shape": [3, 4], "dtype": "float32"}],
            "prelude": [{"op_uid": "aten.t::1", "op_type": "aten::t", "args": [{"type": "tensor", "tensor_id": "W"}],
                         "attributes": {}, "inputs": ["W"], "output": "aten.t::1::out_0"}]}
    assert [s for _t, s, _d in K.feeds(inst)] == [[3], [2, 4], [3, 4]]
    arrays = K.host_inputs(inst)
    disp = NativeATenDispatcher(device="cpu", compute_dtype=torch.float64)
    res = K._ArgResolver()
    store = {tid: torch.from_numpy(a).double() for (tid, _s, _d), a in zip(K.feeds(inst), arrays)}
    attrs = K._NativeAttrs(disp)
    K.apply_prelude(inst, store, lambda raw, st: res.args(raw, st, K._symbols({}), attrs),
                    lambda p, args, at: disp.dispatch(p["op_type"], args, at))
    b = store["aten.t::1::out_0"]
    assert tuple(b.shape) == (4, 3) and not b.is_contiguous()      # the view the engine hands the op
    got = K.reference(inst, arrays, res).numpy()
    want = arrays[0] + arrays[1].astype(np.float64) @ arrays[2].astype(np.float64).T
    assert np.allclose(got, want)


def test_an_op_the_plan_cuts_at_op_level_is_named_never_instanced_whole(monkeypatch):
    # the Dell's V100 run, 2026-11-03 01:23: whole-clip video VAE keys refused KeyNotCertified — the plan runs those
    # components in temporal chunks and their ops in tiles. The instances follow the plan's cuts
    # (`derived_census.cut_bindings`) and an op the plan tiles is skipped by name. RED on 5e1301ad (every op of the
    # component instanced at the request's extent).
    from tests.conftest import installed_container
    import derived_census as DC
    root = installed_container("TinyLlama-1.1B-Chat-v1.0")
    comps = sorted(p.name for p in (root / "components").iterdir() if (p / "graph.json").exists())
    monkeypatch.setattr(DC, "plan_record", lambda model, request, mode, hardware, rung: {
        "components": [{"name": c, "dtype": "float16"} for c in comps]})
    rg = DC.runtime_graph("TinyLlama-1.1B-Chat-v1.0", "model")
    mms = [u for u, o in rg["ops"].items() if o["op_type"] in ("aten::mm", "aten::addmm")][:2]
    assert len(mms) == 2
    real_tiling = DC.plan_tiling
    monkeypatch.setattr(DC, "plan_tiling", lambda plan, comp: (real_tiling({}, comp)[0], {mms[0]: 2})
                        if comp == "model" else real_tiling({}, comp))
    I.bind_profile("v100-16g")                     # a committed profile: the derivation reads the device's facts
    found, skipped = I.instances_of("TinyLlama-1.1B-Chat-v1.0", set(mms), False, "v100-16g")
    sites = {s for i in found.values() for s in i["sites"]}
    assert f"TinyLlama-1.1B-Chat-v1.0/model/{mms[1]}" in sites
    assert f"TinyLlama-1.1B-Chat-v1.0/model/{mms[0]}" not in sites
    assert "tiles" in skipped[f"model/{mms[0]}"]


def test_an_op_whose_output_outgrows_its_operands_is_counted_by_its_output():
    # a 1x1x1 convolution widening 16 -> 128 channels on a video clip (03/11: the guard counted the largest OPERAND
    # for the output, ~15 GB of a ~37 GB footprint, and the Mac swapped until its disk was full)
    inst = {"key": "w1", "op_type": "aten::convolution", "shapes": [[1, 16, 49, 480, 720], [128, 16, 1, 1, 1], [128]],
            "dtypes": ["bfloat16", "bfloat16", "bfloat16"], "output_shapes": [[1, 128, 49, 480, 720]]}
    out_elems = 128 * 49 * 480 * 720
    assert K.instance_bytes(inst) >= out_elems * (8 + 2 + 4)      # the output's float64, device and host copies
    assert K.instance_bytes(inst) > K.device_budget_bytes({"devices": [{"memory_mb": 18186}]})   # out of class


def test_an_instance_without_output_shapes_is_refused_by_name():
    with pytest.raises(KeyError, match="output_shapes"):
        K.instance_bytes({"key": "x", "shapes": [[2, 2]], "dtypes": ["float32"]})


@pytest.mark.parametrize("nd, stride, padding, dilation", [(2, 1, 1, 1), (2, 2, 1, 1), (3, 1, 1, 1), (3, 1, 2, 2)])
def test_a_convolution_s_reference_in_bands_is_the_whole_reference(nd, stride, padding, dilation):
    # the float64 CPU reference unfolds its input (34 GB for a video VAE conv3d, 03/11 12:10): computed in bands of its
    # first spatial output axis it is the same reference
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    sp = [9] * nd
    x = torch.randn(1, 4, *sp, dtype=torch.float64, generator=g)
    w = torch.randn(6, 4, *([3] * nd), dtype=torch.float64, generator=g)
    b = torch.randn(6, dtype=torch.float64, generator=g)
    whole = torch.ops.aten.convolution(x, w, b, [stride] * nd, [padding] * nd, [dilation] * nd, False, [0] * nd, 1)
    inst = {"op_type": "aten::convolution", "shapes": [list(x.shape), list(w.shape), list(b.shape)],
            "output_shapes": [list(whole.shape)]}
    row = K.conv_reference_row_bytes(inst)
    assert row == 4 * 3 ** nd * K._elems(whole.shape[3:]) * 8
    inst["ref_band_bytes"] = 2 * row                       # two output rows per band
    got = K._banded_convolution(inst, [x, w, b, stride, padding, dilation, False, 0, 1])
    # the same sums, reordered by the band's shape (torch picks its float64 conv2d algorithm per shape): within a few
    # float64 epsilons of the whole — measured 1-2 ulps (2.4e-16, 2.9e-16) — far below any judged tolerance (2e-3)
    assert got is not None and float((got - whole).abs().max() / whole.abs().max()) <= 16 * torch.finfo(torch.float64).eps
    inst["ref_band_bytes"] = row * whole.shape[2]          # fits whole: no banding
    assert K._banded_convolution(inst, [x, w, b, stride, padding, dilation, False, 0, 1]) is None


def test_the_guard_counts_a_convolution_s_reference_unfold_row():
    inst = {"key": "c", "op_type": "aten::convolution", "shapes": [[1, 256, 26, 160, 160], [256, 256, 3, 3, 3], [256]],
            "dtypes": ["float32", "bfloat16", "bfloat16"], "output_shapes": [[1, 256, 24, 160, 160]]}
    row = 256 * 27 * 160 * 160 * 8
    assert K.conv_reference_row_bytes(inst) == row
    base = {**inst, "op_type": "aten::other"}
    assert K.instance_bytes(inst) - K.instance_bytes(base) == row + 1 * 256 * 26 * 160 * 160 * 8
    # this instance swapped the Mac whole (~34 GB of unfold); banded it fits the 18 GB device's op budget
    assert K.instance_bytes(inst) < K.device_budget_bytes({"devices": [{"memory_mb": 18186}]})


def test_an_overflow_the_output_dtype_makes_is_judged_against_the_rounded_reference():
    # granite-speech's projector (the Dell's V100 run, 03/11): an fp16 product by -3.39e38 is -inf in every fp16
    # implementation; its float64 reference is finite — the judge compares with the reference rounded to fp16
    pytest.importorskip("torch")
    ref = np.array([-3.39e38 * 2.0, 1.5, -2.25], dtype=np.float64)
    got = np.array([-np.inf, 1.5, -2.25], dtype=np.float64)
    err, mean = K.compare_tiles(got, ref, "float16")
    assert err == 0.0 and mean == 0.0
    err, _ = K.compare_tiles(np.array([-np.inf, 1.5, -2.0]), ref, "float16")
    assert 0 < err < float("inf")                          # a real difference beside it is still measured
    assert K.compare_tiles(got, ref)[0] != 0.0              # unrounded (no output dtype), the overflow stays a difference


def test_an_op_the_contract_narrows_is_bounded_by_the_dtype_it_is_computed_in():
    # an fp32 SDPA the precision contract narrows to the fp16 compute dtype (GLM-4.1V, MiniCPM-o: 1.4e-2 vs 2e-3)
    inst = {"dtypes": ["float32", "float32", "float32", "float32"], "output_dtypes": ["float32"],
            "compute_dtype": "float16", "contract": {"safe": True, "island": "narrow"}}
    assert K.computed_dtypes(inst) == ["float16"]
    row = K.judge_row({}, inst, "float32", 1.0, [1.0], 0.0, "x", (1.4e-2, 1e-3), "")
    assert row["status"] == "ok" and row["tolerance"] == K.TOLERANCE["float16"]
    kept = {**inst, "contract": {"safe": True, "island": "fp32"}}        # an fp32 island stays judged at fp32
    assert K.computed_dtypes(kept) == []
    assert K.judge_row({}, kept, "float32", 1.0, [1.0], 0.0, "x", (1.4e-2, 1e-3), "")["status"] == "wrong"
    unsafe = {**inst, "contract": {"safe": False, "island": None}}       # no narrowing: the op's own dtypes
    assert K.computed_dtypes(unsafe) == []
