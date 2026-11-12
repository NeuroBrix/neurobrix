"""The certification harness (`ops.autotune_harness`) and its references (`ops.oracles.fp64`), moved from the
engine's certifier on 2026-11-12: what each piece promises, on its own.

- `Room` draws a padded-row key at the shortest row of its class where the top does not fit, an element-wise class at
  its floor, and refuses what nothing fits, before a value is drawn.
- `key_operands` names exactly the operands `synthesize` builds: the price a caller reads cannot drift from the draw.
- `LaunchUnderTest.sweep` poisons the output before every candidate: a configuration that writes nothing reads as NaN
  and is excluded, never credited with its predecessor's correct values; a configuration that raises is counted unrun.
- `config_to_dict` / `config_from_dict` round-trip a Triton config; `matmul_oracle` windows rows above the cap.

Injections, each seen red then restored (2026-11-12): `Room.row_shape` returning the top unchanged; `key_operands`
drawing a GEMM at its bucket's top instead of `gemm_rows`; `LaunchUnderTest.sweep` without its `poison()`;
`row_windows` returning the whole product above the cap; `flash_route_heads` stepping its heads by four.
"""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.ops import autotune_harness as H
from nbxtensor.ops.oracles import fp64 as F


def test_a_padded_row_key_is_drawn_at_its_shortest_row_where_its_top_does_not_fit():
    room = H.Room(card=1 << 30, usable=(1 << 30) // 2)
    assert room.row_shape(1024, 256, "fp32", "fp32", "softmax") == (1024, 256)          # fits whole
    rows, feat = room.row_shape(100000, 1024, "fp32", "fp32", "softmax")                # 819 MB at the top
    assert feat == H.shortest_row(1024) == 513 and rows == 100000
    with pytest.raises(H.KeyTooLargeForClass) as err:
        room.row_shape(1 << 20, 1024, "fp32", "fp32", "softmax")                        # not even the shortest
    assert err.value.card == 1 << 30
    assert H.Room().row_shape(1 << 30, 1024, "fp32", "fp32", "softmax") == (1 << 30, 1024)   # unbounded


def test_an_elementwise_class_is_drawn_at_its_floor_and_an_operand_set_over_the_share_is_refused():
    from nbxtensor.ops.autotune_bucket import class_floor
    room = H.Room(card=1 << 30, usable=1 << 29)                       # 512 MiB: 64 Mi elements in and out
    assert room.ew_count(1 << 24, "fp32", "ew") == 1 << 24
    top = 68719476736                                                     # the open class's label
    assert room.ew_count(top, "fp32", "ew") == class_floor("elements", top) < 1 << 26
    with pytest.raises(H.KeyTooLargeForClass):
        H.Room(card=1 << 20, usable=1 << 19).ew_count(top, "fp32", "ew")   # not even its floor
    room = H.Room(card=1 << 20, usable=1 << 19)
    room.refuse([((256, 256), "fp32")], "mm")                                            # 256 KiB of a 512 KiB share
    with pytest.raises(H.KeyTooLargeForClass) as err:
        room.refuse([((256, 256), "fp32"), ((256, 256), "fp32"), ((1,), "fp32")], "mm")
    assert err.value.usable == 1 << 19 and err.value.asked == 2 * 256 * 256 * 4 + 4


class _Tuner:
    arg_names = ["a_ptr", "b_ptr", "c_ptr", "M", "N", "K"]


def test_the_operands_priced_are_the_operands_drawn(monkeypatch):
    from nbxtensor.ops import autotune_certified as C
    from nbxtensor.ops import launch_keys as LK
    monkeypatch.setattr(C, "output_dtype", lambda tuner, key: "fp32")
    # the representative row count, not the bucket's top (the matrix unit may take the top on some cards)
    monkeypatch.setattr(LK, "gemm_row_representative", lambda bucket, a, b, unit_keyed: bucket - 1)
    key = (64, 48, 40, False, False, "fp32", "fp32", "fp32")
    ops = H.key_operands("nbxtensor.ops.matmul.matmul_kernel", _Tuner(), key)
    M = H.gemm_rows(key, ["fp32"] * 3, unit_keyed=True)
    assert M == 63 and ops["family"] == "mm" and ops["inputs"] == [("a", (M, 40), "fp32"), ("b", (40, 48), "fp32")]
    assert ops["out"] == ((M, 48), "fp32")
    assert H.key_operands("nbxtensor.ops.nothing.unknown_kernel", _Tuner(), key) is None


class _Cfg:
    def __init__(self, mode):
        self.kwargs, self.num_warps, self.num_stages, self.num_ctas, self.maxnreg = {"MODE": mode}, 4, 1, 1, None

    def all_kwargs(self):
        return {"MODE": self.kwargs["MODE"]}


class _Fn:
    def __init__(self, good):
        self.good = good

    def run(self, a, out, MODE):
        if MODE == "raise":
            raise RuntimeError("the backend refuses this configuration")
        if MODE == "good":
            out.copy_(self.good)
        elif MODE == "wrong":
            out.copy_(self.good * 2.0)
        # "lazy": writes nothing at all


def test_every_candidate_runs_on_a_poisoned_output():
    from nbxtensor.tensor import NBXTensor
    want = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    good = NBXTensor.from_numpy(want)
    out = NBXTensor.from_numpy(np.zeros((8, 8), dtype=np.float32))
    tuner = type("T", (), {"arg_names": ["a", "out"], "fn": _Fn(good)})()
    launch = H.LaunchUnderTest(tuner, (good, out), {}, "out", "test")
    cfgs = [_Cfg("good"), _Cfg("lazy"), _Cfg("wrong"), _Cfg("raise")]
    accepted, excluded, unrun = launch.sweep(cfgs, want.astype(np.float64), 1e-6)
    assert [c.kwargs["MODE"] for c, _dev, _s in accepted] == ["good"]
    assert [e["config"]["kwargs"]["MODE"] for e in excluded] == ["lazy", "wrong"]
    assert excluded[0]["deviation"] == float("inf")                # NaN where the lazy one wrote nothing
    assert [u["config"]["kwargs"]["MODE"] for u in unrun] == ["raise"]
    launch.final(cfgs[0])
    assert np.array_equal(F.host_values(out), want)


def test_a_configuration_round_trips_through_its_record():
    pytest.importorskip("triton")
    import triton
    cfg = triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=2)
    back = H.stored_config(H.config_dict(cfg))
    assert H.config_dict(back) == H.config_dict(cfg) and back.kwargs == {"BLOCK_M": 64}


def test_a_product_above_the_cap_is_referenced_on_its_first_middle_and_last_rows(monkeypatch):
    rng = np.random.default_rng(3)
    a, b = H.draw(rng, (90, 16), "fp32"), H.draw(rng, (16, 8), "fp32")
    whole = F.f64(a) @ F.f64(b)
    monkeypatch.setattr(F, "ORACLE_MAX_MACS", 3 * 10 * 16 * 8)
    ref = F.matmul_oracle(a, b)()
    assert isinstance(ref, F.RowWindowedOracle)
    assert [w for w, _ in ref.blocks] == [(0, 10), (40, 50), (80, 90)]
    assert F.oracle_deviation(whole, ref) <= 1e-12                   # a slice's BLAS path: round-off only
    assert F.oracle_deviation(whole + 1.0, ref) > 1e-3


def test_a_flash_key_is_drawn_at_the_fewest_heads_that_take_the_flash(monkeypatch):
    from nbxtensor.ops import attention_wrappers as AW
    from nbxtensor.ops import launch_keys as LK
    calls = []

    def route(batch, heads, *a, **k):
        calls.append(heads)
        return ("flash" if heads >= 16 else "math", None)
    monkeypatch.setattr(LK, "sdpa_route", route)
    assert AW.flash_route_heads(4096, 4096, 64, "fp16", "fp16", "fp16") == 16
    assert calls == [1, 2, 4, 8, 16]
