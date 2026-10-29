"""The replication pad and the element-wise activations read a permuted input where it lies, with one allocation —
the result. mochi-1-preview's VAE at the 8 GB rung (2026-10-28): its silu copied a 3 GB permute output contiguous
first, and its replication pad held that copy plus one output-sized tensor per padded axis (narrow + expand + cat),
none priced by the plan.

What this test would do if the code were wrong: a copy beside the result (more than one allocation — red); an edge
not repeated, or a value at the wrong place for a permuted input (numpy disagrees — red)."""
import numpy as np
import pytest

from nbxtensor.tensor import DeviceAllocator, NBXTensor


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


def _counting(monkeypatch):
    sizes = []
    orig = DeviceAllocator.malloc_cuda
    monkeypatch.setattr(DeviceAllocator, "malloc_cuda",
                        staticmethod(lambda n, *a, **k: (sizes.append(n), orig(n, *a, **k))[1]))
    return sizes


@pytest.mark.parametrize("n_axes", [1, 2, 3])
@pytest.mark.parametrize("permuted", [False, True])
def test_the_replication_pad_repeats_the_edge_in_one_pass(monkeypatch, n_axes, permuted):
    from nbxtensor.ops import wrappers as W
    rng = np.random.default_rng(n_axes)
    host = rng.standard_normal((2, 3, 4, 5, 6)).astype(np.float32)
    x = NBXTensor.from_numpy(host).to_cuda(0)
    ref_in = host
    if permuted:
        x = x.permute(0, 2, 1, 3, 4)
        ref_in = host.transpose(0, 2, 1, 3, 4)
    pads = [1, 2, 0, 3, 2, 1][:2 * n_axes]
    getattr(W, f"replication_pad{n_axes}d_wrapper")(x, pads)      # a process's first launch allocates once beside
    sizes = _counting(monkeypatch)
    out = getattr(W, f"replication_pad{n_axes}d_wrapper")(x, pads)
    assert len(sizes) == 1, f"allocations beside the result: {sizes}"
    width = [(0, 0)] * (5 - n_axes) + [(pads[2 * k], pads[2 * k + 1]) for k in reversed(range(n_axes))]
    np.testing.assert_array_equal(out.numpy(), np.pad(ref_in, width, mode="edge"))


@pytest.mark.parametrize("name", ["silu", "relu", "sigmoid_wrapper", "neg"])
def test_an_activation_of_a_permuted_input_keeps_its_layout_and_copies_nothing(monkeypatch, name):
    from nbxtensor.ops import wrappers as W
    rng = np.random.default_rng(1)
    host = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x = NBXTensor.from_numpy(host).to_cuda(0).permute(0, 2, 3, 1)
    assert not x.is_contiguous()
    getattr(W, name)(x)                                             # a process's first launch allocates once beside
    sizes = _counting(monkeypatch)
    out = getattr(W, name)(x)
    assert len(sizes) == 1, f"allocations beside the result: {sizes}"
    assert tuple(out.stride()) == tuple(x.stride()) and tuple(out.shape) == tuple(x.shape)
    v = host.transpose(0, 2, 3, 1)
    ref = {"silu": v / (1 + np.exp(-v)), "relu": np.maximum(v, 0), "sigmoid_wrapper": 1 / (1 + np.exp(-v)),
           "neg": -v}[name]
    np.testing.assert_allclose(out.contiguous().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_a_conv3d_adds_each_tap_into_its_sum(monkeypatch):
    """`_conv3d_via_conv2d` adds each later tap INTO the sum: `out + tap` held the old sum, the tap and the new one."""
    from nbxtensor.ops import wrappers as W
    rng = np.random.default_rng(2)
    x = NBXTensor.from_numpy(rng.standard_normal((1, 4, 5, 8, 8)).astype(np.float32)).to_cuda(0)
    w = NBXTensor.from_numpy(rng.standard_normal((6, 4, 3, 3, 3)).astype(np.float32)).to_cuda(0)
    W._conv3d_via_conv2d(x, w, None, 1, (0, 1, 1), 1, 1)        # a process's first launch allocates once beside
    sizes = _counting(monkeypatch)
    out = W._conv3d_via_conv2d(x, w, None, 1, (0, 1, 1), 1, 1)
    full = out.numel() * 4
    assert sum(1 for n in sizes if n >= full) == 3, f"output-sized allocations for 3 taps: {sizes}"
