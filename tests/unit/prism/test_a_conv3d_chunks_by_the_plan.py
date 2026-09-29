"""A 3-D convolution streams its temporal output in chunks when the PLAN says so, never because of
the memory free at an instant (the supervisor's decision 2, 2026-09-29).

The triton wrapper read the driver's free bytes to decide; a key a live run formed then depended on
that instant and no census or derivation could form it. Prism now lists the rank-5 convolutions
whose one-shot peak exceeds the op budget (`OpLevelTilingPlan.conv3d_chunks`) and registers the
chunked path as their interceptor.

What each test would do if the fix were wrong: the wrapper test fails while `_conv3d_via_conv2d`
still reads `device_free_bytes` (red on fc69b543); the selection test fails if a conv under budget,
a rank-4 conv or a transposed conv is listed, or the over-budget one is not; the torch equivalence
test fails if the chunks miss or overlap frames (red with the receptive-field slice shortened).
"""
from __future__ import annotations

import inspect

import pytest

from neurobrix.core.prism import conv3d_chunk as C3


def test_the_wrapper_no_longer_reads_the_drivers_free_bytes():
    from neurobrix.kernels import wrappers as W
    assert "device_free_bytes" not in inspect.getsource(W._conv3d_via_conv2d)


def test_the_plan_lists_exactly_the_conv3d_over_its_budget():
    from neurobrix.core.prism.solver import PrismSolver

    class _Prof:
        def build_symbol_map(self, ic):
            return {}

        def _resolve_shape(self, meta, symbols):
            return meta["shape"]

    def conv(uid, x, w, **at):
        return uid, {"op_type": "aten::convolution", "input_tensor_ids": [uid + "x", uid + "w"],
                     "attributes": at}, {uid + "x": {"shape": x}, uid + "w": {"shape": w}}

    big = conv("big", [1, 128, 81, 480, 832], [128, 128, 3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])
    small = conv("small", [1, 16, 5, 30, 52], [16, 16, 3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])
    flat = conv("flat", [1, 128, 480, 832], [128, 128, 3, 3], stride=[1, 1], padding=[1, 1])
    trans = conv("trans", [1, 128, 81, 480, 832], [128, 128, 3, 3, 3], transposed=True)
    ops, tensors = {}, {}
    for uid, op, ts in (big, small, flat, trans):
        ops[uid] = op; tensors.update(ts)

    class _Comp:
        graph = {"ops": ops, "tensors": tensors, "execution_order": list(ops)}
    need_big, _ = C3.conv3d_need([1, 128, 81, 480, 832], [128, 128, 3, 3, 3], 1, 1, 1, 2, 2)
    assert need_big > 0
    got = PrismSolver.__new__(PrismSolver)._conv3d_chunk_uids(_Comp(), _Prof(), None, 2, need_big - 1)
    assert got == ["big"], got
    assert PrismSolver.__new__(PrismSolver)._conv3d_chunk_uids(_Comp(), _Prof(), None, 2, need_big) == []


def test_the_torch_chunked_path_equals_one_shot_conv3d(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from neurobrix.core.module.tiling_engine import conv3d_chunked
    monkeypatch.setattr(C3, "CHUNK_BYTES", 1)          # one output frame per chunk: every seam tested
    g = torch.Generator().manual_seed(0)
    for (st, pt, dt, kt) in [(1, 1, 1, 3), (2, 1, 1, 3), (1, 0, 2, 3), (1, 2, 1, 5)]:
        x = torch.randn(1, 4, 11, 6, 7, generator=g, dtype=torch.float64)
        w = torch.randn(5, 4, kt, 3, 3, generator=g, dtype=torch.float64)
        b = torch.randn(5, generator=g, dtype=torch.float64)
        ref = F.conv3d(x, w, b, (st, 1, 1), (pt, 1, 1), (dt, 1, 1))
        got = conv3d_chunked(x, w, b, [st, 1, 1], [pt, 1, 1], [dt, 1, 1], False, [0, 0, 0], 1)
        assert got.shape == ref.shape and torch.allclose(got, ref, atol=1e-12), (st, pt, dt, kt)
