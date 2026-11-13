"""nbxtensor.profile.kernel_times: inside a session every Metal launch runs alone in its own encoder and is timed at its
boundaries; the table ranks kernels, changes no output byte, and costs nothing when no session is open.

What this test does if the code were wrong:
- the ranking cell fails when a kernel 256 times heavier does not measure at least 50 times the lighter one, or when
  the launches are not counted by their own names;
- the bytes cell fails when a chain of library ops gives other bytes inside a session than outside;
- the off cell fails when, with no session, ten launches open more than the one shared encoder;
- the refusal cells fail when a second session opens inside the first, or when a device refuses the sample buffer
  and no ZERO FALLBACK names it;
- the pool cell fails when a session past the device's live sample buffers loses or refuses a launch.
Metal only (skipped, said, elsewhere).
"""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.backend.runtime import gpu_backend

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from nbxtensor import profile  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402


@pytest.fixture(autouse=True)
def _metal():
    if gpu_backend() != "metal":
        pytest.skip(f"per-kernel GPU time is a Metal mode; this backend is {gpu_backend()!r}")
    from nbxtensor.backend import launcher
    launcher.install()                                    # the launches go through the library's launcher, as in a run


def _ours(fn):
    """A test kernel routed through nbxtensor's launcher, which takes only the library's own kernels (`_is_ours`)."""
    fn.__module__ = "nbxtensor.tests"
    return triton.jit(fn)


def _spin(x, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(x + i, mask=m)
    for _ in range(ITERS):
        v = v * 1.0000001 + 0.5
    tl.store(x + i, v, mask=m)


def _spin_light(x, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    _spin(x, n, ITERS, BLOCK)


def _spin_heavy(x, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    _spin(x, n, ITERS, BLOCK)


_spin = _ours(_spin)
_spin_light, _spin_heavy = _ours(_spin_light), _ours(_spin_heavy)


N = 1 << 20


def _launch(kernel, x, iters):
    kernel[(triton.cdiv(N, 1024),)](x, N, ITERS=iters, BLOCK=1024)


def test_a_heavier_kernel_ranks_heavier_and_each_launch_is_counted_by_its_name():
    x = NBXTensor.from_numpy(np.ones(N, dtype=np.float32)).to("cuda:0")
    for _ in range(2):                                    # compiled and the GPU's clock up before the session
        _launch(_spin_light, x, 16)
        _launch(_spin_heavy, x, 4096)
    x.to_cpu()
    with profile.kernel_times() as table:
        for _ in range(3):
            _launch(_spin_light, x, 16)
            _launch(_spin_heavy, x, 4096)
    rows = {name: (count, total) for name, count, total, _m in table.ranked()}
    light = next(v for k, v in rows.items() if "spin_light" in k)
    heavy = next(v for k, v in rows.items() if "spin_heavy" in k)
    assert light[0] == 3 and heavy[0] == 3 and table.unsampled == 0
    assert heavy[1] >= 50 * light[1], (light, heavy)
    assert table.ranked()[0][0] == next(k for k in rows if "spin_heavy" in k)
    assert "a ranking" in table.format()


def _chain(a, b):
    from nbxtensor.ops import attention_wrappers as AW, elementwise_wrappers as EW
    c = AW.softmax(EW.mul(EW.add(a, b), a), dim=-1)
    return EW.add(c, b).to_cpu().numpy().tobytes()


def test_a_chain_of_library_ops_gives_the_same_bytes_inside_and_outside_a_session():
    rng = np.random.default_rng(0)
    a = NBXTensor.from_numpy(rng.standard_normal((64, 64), dtype=np.float32)).to("cuda:0")
    b = NBXTensor.from_numpy(rng.standard_normal((64, 64), dtype=np.float32)).to("cuda:0")
    outside = _chain(a, b)
    with profile.kernel_times() as table:
        inside = _chain(a, b)
    assert inside == outside
    assert sum(c for _n, c, _t, _m in table.ranked()) >= 2


def test_with_no_session_ten_launches_share_one_encoder():
    from nbxtensor.backend.metal_device import runtime
    from nbxtensor.backend import metal_profile
    rt = runtime()
    x = NBXTensor.from_numpy(np.ones(N, dtype=np.float32)).to("cuda:0")
    _launch(_spin_light, x, 1)
    rt.flush()
    opened = []
    real = rt._new_encoder
    rt._new_encoder = lambda cb, kind: opened.append(kind) or real(cb, kind)
    try:
        for _ in range(10):
            _launch(_spin_light, x, 1)
        assert metal_profile.SESSION is None and len(opened) == 1
    finally:
        del rt._new_encoder
        rt.flush()


def test_a_second_session_inside_the_first_is_refused_by_name():
    with profile.kernel_times():
        with pytest.raises(RuntimeError, match="already open"):
            with profile.kernel_times():
                pass


def test_a_sample_buffer_the_device_refuses_is_refused_by_name():
    from nbxtensor.backend import metal_profile
    from nbxtensor.backend.metal_device import runtime

    class _Refusing:
        """The real device, except that it refuses every counter sample buffer (a size past its limit)."""

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def newCounterSampleBufferWithDescriptor_error_(self, desc, err):
            return None, "sample count past the device's limit"

    with pytest.raises(RuntimeError, match="ZERO FALLBACK: a timestamp sample buffer of"):
        metal_profile.Session(_Refusing(runtime()._device), 1 << 20)


def test_a_session_longer_than_the_device_s_sample_buffers_reuses_them():
    """More command buffers than the pool holds (an autotune sweep inside a session made 33): the oldest is read and
    its samples reused, every launch still counted."""
    from nbxtensor.backend import metal_profile
    from nbxtensor.backend.metal_device import runtime
    rt = runtime()
    x = NBXTensor.from_numpy(np.ones(N, dtype=np.float32)).to("cuda:0")
    _launch(_spin_light, x, 1)
    rt.flush()
    with profile.kernel_times() as table:
        pool = len(metal_profile.SESSION.free)
        for _ in range(pool + 8):
            _launch(_spin_light, x, 1)
            rt.close_stream(0)                            # one command buffer per launch: past the pool
    assert table.ranked()[0][1] == pool + 8 and table.unsampled == 0
