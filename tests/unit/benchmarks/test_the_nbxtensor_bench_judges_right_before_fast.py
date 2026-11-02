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
    assert x[1].dtype == np.int64 and x[1].max() < 16 and x[2].shape == ()


def test_the_compute_dtype_of_a_tie_never_depends_on_the_process():
    # one fp32 island beside a half weight: the op's recorded output decides; without one, the name breaks the tie
    # (a set's order is salted per process — the two sides of the bench must agree on it run after run)
    assert K._compute_dtype({"dtypes": ["float32", "bfloat16"], "output_dtypes": ["float32"]}) == "float32"
    assert K._compute_dtype({"dtypes": ["float32", "bfloat16"], "output_dtypes": None}) == "bfloat16"
    assert K._compute_dtype({"dtypes": ["float16", "float16", "float32"]}) == "float16"
    assert K._compute_dtype({"dtypes": ["int64"]}) == "float32"


def test_a_video_instance_too_large_for_the_device_is_out_of_class_never_run():
    # a video VAE convolution at its whole request shape (Allegro, [1, 512, 22, 180, 320]): its operands and float64
    # reference exceed the op budget an 18 GB device grants — the row says so with its numbers, nothing runs
    inst = {"key": "v1", "op_type": "aten::convolution", "shapes": [[1, 512, 22, 180, 320], [512, 512, 3, 3, 3], [512]],
            "dtypes": ["bfloat16", "bfloat16", "bfloat16"], "sites": ["Allegro/vae/aten.convolution::3"]}
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
    small = {**inst, "shapes": [[1, 64, 2, 16, 16], [64, 64, 3, 3, 3], [64]]}
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
    rows = K.torch_work(inst, None, None, 0, 2, ["default"], nbx)
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
