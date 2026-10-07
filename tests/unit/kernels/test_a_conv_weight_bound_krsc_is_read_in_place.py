"""A convolution weight stored (K, R, S, C) by the loader is read where it lies.

Measured on the M4 Pro (2026-10-07, served adbf6e84): with the weight's channels innermost every
3x3 key the census holds runs 46-63 % faster (1024 ch at 256^2: 3.31 -> 5.38 TFLOP/s), 1x1 is
the same memory either way. Supervisor decision 2026-10-08 00:05: a physical relayout ONCE,
where the weight is written to the device, from the hardware profile's `conv.weight_layout`;
the original order is not kept; an arch that has not measured it states KCRS.

The weight stays a logical (K, C, R, S) tensor, so every reader of its shape (the key, Prism's
bytes, the census) is unchanged; only its strides say where the channels lie. The wrapper takes
either order, so a weight the relayout did not reach is still exact, only slower.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
VENDORS = REPO / "src" / "neurobrix" / "config" / "vendors"


def test_every_profile_states_its_conv_weight_layout_and_only_the_measured_arch_says_krsc():
    layouts = {str(y.relative_to(VENDORS)): (yaml.safe_load(y.read_text()).get("conv") or {}).get("weight_layout")
               for y in sorted(VENDORS.glob("*/*.yml"))}
    assert layouts and all(v in ("KCRS", "KRSC") for v in layouts.values()), layouts
    assert {k for k, v in layouts.items() if v == "KRSC"} == {"apple/apple_m4_pro.yml"}


def test_the_layout_is_read_from_the_profile_and_a_profile_without_one_is_refused(monkeypatch):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.ops import _configs as K
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {"conv": {"weight_layout": "KRSC"}})
    assert LK.conv_weight_layout() == "KRSC"
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {"flash": {}})
    with pytest.raises(ValueError, match="weight_layout"):
        LK.conv_weight_layout()
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {"conv": {"weight_layout": "KSRC"}})
    with pytest.raises(ValueError, match="weight_layout"):
        LK.conv_weight_layout()
    monkeypatch.setattr(K, "active_vendor_profile", lambda: {})
    assert LK.conv_weight_layout() == "KCRS"          # no profile: the container's own order


@pytest.mark.parametrize("w_shape, groups, transposed, expected", [
    ((64, 32, 3, 3), 1, False, True),                 # the plain 3x3 conv
    ((64, 16, 3, 3), 2, False, True),                 # grouped, more than one channel a group
    ((64, 32, 1, 1), 1, False, False),                # 1x1: the same memory in both orders
    ((64, 1, 3, 3), 64, False, False),                # depthwise: its kernel reads no channel stride
    ((32, 64, 3, 3), 1, True, False),                 # transposed: another kernel
    ((64, 32, 3), 1, False, False),                   # conv1d
    ((64, 32, 3, 3, 3), 1, False, False),             # conv3d, sliced per frame
])
def test_which_weights_the_krsc_layout_takes(w_shape, groups, transposed, expected):
    from neurobrix.kernels import launch_keys as LK
    assert LK.conv_weight_krsc(w_shape, groups, transposed, "KRSC") is expected
    assert LK.conv_weight_krsc(w_shape, groups, transposed, "KCRS") is False


def test_a_weight_is_relaid_only_when_every_reader_is_a_conv_that_takes_it():
    from neurobrix.kernels import launch_keys as LK
    tensors = {"param::a": {"shape": [8, 4, 3, 3], "weight_name": "a.weight"},
               "param::b": {"shape": [8, 4, 3, 3], "weight_name": "b.weight"},
               "param::d": {"shape": [8, 1, 3, 3], "weight_name": "d.weight"},
               "param::p": {"shape": [8, 4, 1, 1], "weight_name": "p.weight"}}
    conv = lambda w, g=1: {"op_type": "aten::convolution", "input_tensor_ids": ["x", w],
                           "attributes": {"groups": g, "transposed": False}}
    ops = {"c0": conv("param::a"), "c1": conv("param::b"), "c2": conv("param::d", 8),
           "c3": conv("param::p"), "v": {"op_type": "aten::view", "input_tensor_ids": ["param::b"]}}
    dag = {"tensors": tensors, "ops": ops, "execution_order": list(ops)}
    assert LK.krsc_conv_weights(dag, "KRSC") == {"param::a"}
    assert LK.krsc_conv_weights(dag, "KCRS") == set()


def test_the_host_relayout_keeps_the_logical_shape_and_puts_the_channels_innermost():
    from neurobrix.triton.weight_loader import host_weight_layout
    w = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)
    arr, strides = host_weight_layout(w, True)
    assert arr.flags["C_CONTIGUOUS"] and arr.shape == (2, 2, 4, 3)
    assert strides == (2 * 4 * 3, 1, 4 * 3, 3)
    view = np.lib.stride_tricks.as_strided(arr, w.shape, [s * arr.itemsize for s in strides])
    assert np.array_equal(view, w)
    same, s2 = host_weight_layout(w, False)
    assert same is w and s2 == (3 * 2 * 4, 2 * 4, 4, 1)


def _dev_ok():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        return DeviceAllocator.device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _dev_ok(), reason="no accelerator visible")
@pytest.mark.parametrize("N, C, H, W, K, kh, kw, groups, stride, pad", [
    (1, 32, 17, 19, 48, 3, 3, 1, 1, 1),
    (2, 16, 12, 12, 24, 3, 3, 2, 2, 1),
    (1, 8, 9, 9, 8, 5, 3, 1, 1, 2),
])
def test_the_wrapper_reads_a_krsc_weight_in_place_and_gives_the_same_bytes(
        monkeypatch, N, C, H, W, K, kh, kw, groups, stride, pad):
    from neurobrix.kernels import wrappers as Wr
    from neurobrix.kernels.nbx_tensor import NBXTensor
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, C, H, W)).astype(np.float32)
    w = rng.standard_normal((K, C // groups, kh, kw)).astype(np.float32)
    ref = Wr.conv2d_wrapper(NBXTensor.from_numpy(x), NBXTensor.from_numpy(w),
                            stride=stride, padding=pad, groups=groups).numpy()
    wk = NBXTensor.from_numpy(np.ascontiguousarray(w.transpose(0, 2, 3, 1))).permute(0, 3, 1, 2)
    assert tuple(wk.shape) == w.shape and not wk.is_contiguous()
    copied = []
    real = type(wk).contiguous
    monkeypatch.setattr(type(wk), "contiguous",
                        lambda self: (copied.append(tuple(self.shape)), real(self))[1])
    got = Wr.conv2d_wrapper(NBXTensor.from_numpy(x), wk, stride=stride, padding=pad,
                            groups=groups).numpy()
    assert w.shape not in copied                      # the weight was not copied back to KCRS
    assert np.array_equal(got, ref)


def test_the_executor_joins_the_relaid_weights_to_the_loader_keys(monkeypatch):
    from types import SimpleNamespace
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.kernels import launch_keys as LK
    monkeypatch.setattr(LK, "conv_weight_layout", lambda: "KRSC")
    conv = lambda w: {"op_type": "aten::convolution", "input_tensor_ids": ["x", w],
                      "attributes": {"groups": 1, "transposed": False}}
    tensors = {f"param::{n}": {"shape": [8, 4, 3, 3]} for n in ("enc.a", "b", "c")}
    ops = {"c0": conv("param::enc.a"), "c1": conv("param::b"),
           "u": {"op_type": "aten::view", "input_tensor_ids": ["param::c"]}}
    ex = SimpleNamespace(_dag={"tensors": tensors, "ops": ops, "execution_order": list(ops)},
                         _pending_weight_binding={"enc.a": "a", "b": "kb", "c": "kb"})
    krsc = GraphExecutor._krsc_loader_keys
    assert krsc(ex, {"a", "kb"}) == {"a"}             # kb is also bound to c, which a view reads
    assert krsc(ex, None) is None                     # a load of everything: file order
    ex._pending_weight_binding = None
    assert krsc(ex, {"a", "kb"}) is None
