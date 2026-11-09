"""The gathered path (offload_gathered / restore_scattered: one kernel per K and V gathers every layer's page on the
device, then one copy per page and per run of a device's layers) moves the same bits into the same host layout as the
per-layer copies: each path restores what the other offloaded, bit-exact, into the same pool addresses; a model whose
layers sit on two cards keeps each card's run. Decided between the agents 2026-11-09 20:52 (the V100 measure: per-layer
copies 150-230 ms, one copy 7-21 ms for 125 TinyLlama pages).

What this test would do if the code were wrong: a page or row misplaced, a layer's K and V swapped, the host layout
differing from the per-layer path's (red: the cross restore differs), a pool moved, the other pages written (red)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.kv_host_pages import HostPages, offload, offload_gathered, restore, restore_scattered
from nbxtensor.tensor import NBXTensor

LAYERS, PAGES, HEADS, TOKENS, DK, DV = 4, 7, 2, 16, 64, 32


def _cards(n):
    try:
        if DeviceAllocator.device_count() < n:
            pytest.skip(f"{n} accelerator(s) needed, {DeviceAllocator.device_count()} visible")
        DeviceAllocator.set_device(0)
        NBXTensor.zeros((1,), dtype=NBXDtype.float16, device="cuda:0")
    except Exception as e:                                   # a card-less machine
        pytest.skip(f"no accelerator: {e}")


def _pools(rng, d, devices):
    out = []
    for i in range(LAYERS):
        DeviceAllocator.set_device(devices[i])
        out.append(NBXTensor.from_numpy(rng.standard_normal((PAGES, HEADS, TOKENS, d)).astype(np.float16)))
    return out


def _zero(pools):
    for p in pools:
        DeviceAllocator.set_device(p._device_idx)
        z = NBXTensor.from_numpy(np.zeros(p.shape, dtype=np.float16))
        DeviceAllocator.memcpy(p.data_ptr(), z.data_ptr(), z.nbytes() if callable(z.nbytes) else z._nbytes, 3)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("out_fn, in_fn", [(offload_gathered, restore_scattered), (offload_gathered, restore),
                                           (offload, restore_scattered)])
@pytest.mark.parametrize("cards", [1, 2])
@pytest.mark.parametrize("host_pages", [[2, 0, 1], [0, 1, 2]])      # scattered, and consecutive (one merged copy)
def test_the_paths_agree_bit_exact(pinned, out_fn, in_fn, cards, host_pages):
    _cards(cards)
    devices = [0] * LAYERS if cards == 1 else [0, 0, 1, 1]
    rng = np.random.default_rng(11)
    k, v = _pools(rng, DK, devices), _pools(rng, DV, devices)
    before_k, before_v = [p.numpy().copy() for p in k], [p.numpy().copy() for p in v]
    addrs = [p.data_ptr() for p in (*k, *v)]
    host = HostPages(capacity=3, num_layers=LAYERS, k_page_bytes=HEADS * TOKENS * DK * 2,
                     v_page_bytes=HEADS * TOKENS * DV * 2, pinned=pinned)
    try:
        out_fn(k, v, [5, 1, 3], host, host_pages)
        _zero([*k, *v])
        in_fn(k, v, host, host_pages, [0, 4, 6])
        for d in set(devices):
            DeviceAllocator.device_synchronize(d)
        for i in range(LAYERS):
            kn, vn = k[i].numpy(), v[i].numpy()
            for src, dst in ((5, 0), (1, 4), (3, 6)):
                assert np.array_equal(kn[dst], before_k[i][src]), (i, "k", src, dst)
                assert np.array_equal(vn[dst], before_v[i][src]), (i, "v", src, dst)
            assert not kn[[1, 2, 3, 5]].any() and not vn[[1, 2, 3, 5]].any()
        assert [p.data_ptr() for p in (*k, *v)] == addrs
    finally:
        host.close()


@pytest.mark.parametrize("cards", [1, 2])
def test_every_launch_goes_through_the_launch_contract(monkeypatch, cards):
    """backend/runtime.set_device is called before every Triton launch (it aligns Triton's driver with the tensor's
    device). The gathered path skipped it: on Metal the FIRST gathered launch of a fresh process failed ("Arg 0: expected
    an MPS tensor") until another op had run (the Mac, 2026-11-09 21:16). Red if a launch is not preceded by it."""
    _cards(cards)
    from nbxtensor.backend import runtime
    from nbxtensor.memory import kv_host_pages as M
    devices = [0] * LAYERS if cards == 1 else [0, 0, 1, 1]
    rng = np.random.default_rng(3)
    k, v = _pools(rng, DK, devices), _pools(rng, DV, devices)
    seen, launches = [], []
    real_set, real_kernel = runtime.set_device, M._kv_page_rows_kernel

    class _Spy:
        def __getitem__(self, grid):
            def go(*a, **kw):
                launches.append(len(seen))                   # how many contract calls preceded this launch
                return real_kernel[grid](*a, **kw)
            return go
    monkeypatch.setattr(runtime, "set_device", lambda t: (seen.append(t._device_idx), real_set(t)))
    monkeypatch.setattr(M, "_kv_page_rows_kernel", _Spy())
    host = HostPages(2, LAYERS, HEADS * TOKENS * DK * 2, HEADS * TOKENS * DV * 2, pinned=False)
    try:
        M.offload_gathered(k, v, [1, 2], host, [0, 1])
        M.restore_scattered(k, v, host, [0, 1], [3, 4])
    finally:
        host.close()
    assert launches and all(b > a for a, b in zip([0] + launches, launches)), (launches, seen)
    assert set(seen) == set(devices)


def test_importing_the_module_installs_the_launcher_seam():
    """A fresh process that imports kv_host_pages and launches its kernel first: the launch must go through the
    library's launcher, never Triton's default driver (Metal refused the NBXTensor argument there; CUDA's imports torch).
    Red if the import leaves the seam uninstalled."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "from nbxtensor.memory import kv_host_pages; "
                          "import nbxtensor.backend.launcher as L, sys; print(L._installed, 'torch' in sys.modules)"],
                         capture_output=True, text=True, env={**__import__("os").environ, "NBX_LAUNCHER": "nbx"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["True", "False"], out.stdout
