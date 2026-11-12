"""On Metal, Triton's compiled kernels run on nbxtensor's own stream: one serial compute encoder, each pointer bound as
(buffer, offset) of its allocation (`backend.metal_dispatch`), the host waiting only where it reads.

Before (the Mac, 2026-11-12): triton-ext's runtime committed a command buffer per launch on a queue of its own and
wrapped each pointer afresh; Metal ran two consecutive launches overlapped (a kernel reading its predecessor's output
was wrong 4 times in 5), so every launch waited for the whole device: 131 493 waits, 19.58 s of a 30.36 s TinyLlama
run.

What this test does if the code is wrong, a long kernel (milliseconds) writing 3.0 everywhere first:
1. the next kernel reads it (write then read): 6.0, not a mix of stale values;
2. a device-to-device copy (a blit) reads it, the host reads it, and its freed memory reused by a tensor the host
   fills with zeros is not overwritten late;
3. a tensor already on "cpu", read in place, waits for the kernels encoded before;
4. five launches make no host wait between them;
5. every argument kind binds: a view inside an allocation (offset), int64 and float scalars, more than 31
   arguments, an assert the kernel fails (raises at the next flush, never by a wait after the launch), a table of
   captured addresses read through (`pinned_gpu_address`, the MoE band's way); and an address in no allocation is
   refused by name;
6. triton-ext's launch metadata is honoured as its own launcher honours it: threadgroup memory (`shared`), the
   pipeline's thread limit, Metal's 31-entry buffer table, and a grid-barrier kernel's co-residency
   (`cross_tg_barrier`, against this GPU's core count) — each refused by name before anything is encoded.

Run: python -m pytest nbxtensor/tests/test_a_metal_kernel_runs_on_nbxtensors_own_stream.py
"""
import sys

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal only", allow_module_level=True)
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

N = 1 << 20
REPS = 4000


def _ours(fn):
    """A test kernel routed through nbxtensor's launcher, which takes only the library's own kernels (`_is_ours`)."""
    fn.__module__ = "nbxtensor.tests"
    return triton.jit(fn)


def _slow_fill(o, n, v, reps, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(reps):
        acc = acc * 0.5 + v                       # converges to exactly 2v in fp32
    tl.store(o + i, acc, mask=i < n)


def _scalars(o, a: tl.int64, b: tl.float32, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o + i, (a + i).to(tl.float32) * b, mask=i < n)


def _checked(x, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x + i, mask=i < n, other=0.0)
    tl.device_assert(v < 100.0, "a value of 100 or more")


def _read_through(table, o, n, BLOCK: tl.constexpr):
    """o[e, :] = the n floats at the address held in table[e]."""
    e = tl.program_id(0)
    src = tl.load(table + e).to(tl.pointer_type(tl.float32))
    i = tl.arange(0, BLOCK)
    tl.store(o + e * n + i, tl.load(src + i, mask=i < n), mask=i < n)


_slow_fill, _scalars, _read_through = _ours(_slow_fill), _ours(_scalars), _ours(_read_through)
_checked.__module__ = "nbxtensor.tests"
_checked = triton.jit(_checked, debug=True)


def _device():
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.backend import metal_device as M
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    import nbxtensor.ops.elementwise_wrappers  # noqa: F401  the launcher's seam, as every op module installs it
    try:
        DeviceAllocator.set_device(0)
        NBXTensor.zeros((1,), dtype=NBXDtype.float32, device="cuda:0")
        M.runtime().sync()
    except Exception as e:                                         # pragma: no cover
        pytest.skip(f"no Metal device: {e}")
    return M, NBXTensor, NBXDtype, DeviceAllocator


def _filled(T, D):
    o = T.empty((N,), dtype=D.float32, device="cuda:0")
    _slow_fill[(N // 1024,)](o, N, 1.5, REPS, BLOCK=1024)
    return o


def test_the_next_kernel_reads_what_the_last_one_wrote():
    from nbxtensor.ops import elementwise_wrappers as E
    M, T, D, A = _device()
    for _ in range(5):
        assert np.all(E.add(_filled(T, D), _filled(T, D)).numpy() == 6.0)


def test_a_blit_reads_the_kernels_output():
    M, T, D, A = _device()
    dst = T.from_numpy(np.full(N, -1.0, dtype=np.float32), device="cuda:0")
    o = _filled(T, D)
    A.memcpy(dst.data_ptr(), o.data_ptr(), N * 4, kind=3)
    assert np.all(dst.numpy() == 3.0)


def test_the_host_reads_the_kernels_output():
    M, T, D, A = _device()
    assert np.all(_filled(T, D).numpy() == 3.0)


def test_a_freed_buffer_is_not_written_after_its_reuse():
    M, T, D, A = _device()
    o = _filled(T, D)
    freed = o.data_ptr()
    del o
    fresh = T.from_numpy(np.zeros(N, dtype=np.float32), device="cuda:0")
    assert fresh.data_ptr() == freed                                # the freed memory, handed out again
    M.runtime().sync()                                              # every kernel done
    assert np.all(fresh.numpy() == 0.0)


def test_a_host_tensor_read_in_place_waits_for_the_kernels_before_it(monkeypatch):
    M, T, D, A = _device()
    host = T.from_numpy(np.ones(4, dtype=np.float32), device="cuda:0").to_cpu()
    _filled(T, D)
    flushed = []
    real = M.runtime().flush
    monkeypatch.setattr(M.runtime(), "flush", lambda *a: (flushed.append(1), real(*a))[1])
    host.numpy()
    assert flushed, "a host tensor was read in place while kernels were pending"


def test_five_launches_wait_for_nothing_between_them(monkeypatch):
    M, T, D, A = _device()
    M.runtime().sync()
    waits = []
    real = M.runtime()._await
    monkeypatch.setattr(M.runtime(), "_await", lambda cb: (waits.append(1), real(cb))[1])
    from triton_apple_backend import metal_native                   # triton-ext's device wait, the old path's
    native = metal_native.synchronize
    monkeypatch.setattr(metal_native, "synchronize", lambda: (waits.append(1), native())[1])
    outs = [_filled(T, D) for _ in range(5)]
    assert waits == [], f"{len(waits)} host waits between five launches"
    assert all(np.all(o.numpy() == 3.0) for o in outs)


def test_a_view_inside_an_allocation_binds_at_its_offset():
    M, T, D, A = _device()
    base = T.from_numpy(np.full(4096, -1.0, dtype=np.float32), device="cuda:0")
    view = base[1000:2000]
    assert view.data_ptr() == base.data_ptr() + 4000
    _slow_fill[(1,)](view, 1000, 2.0, 200, BLOCK=1024)
    got = base.numpy()
    assert np.all(got[1000:2000] == 4.0) and np.all(got[:1000] == -1.0) and np.all(got[2000:] == -1.0)


def test_int64_and_float_scalars_arrive_as_given():
    M, T, D, A = _device()
    o = T.empty((256,), dtype=D.float32, device="cuda:0")
    _scalars[(1,)](o, 3 << 33, 0.5, 256, BLOCK=256)
    want = ((np.int64(3 << 33) + np.arange(256)).astype(np.float32) * np.float32(0.5))
    assert np.array_equal(o.numpy(), want)


def test_a_failed_device_assert_raises_and_a_kept_one_does_not():
    """The status block is checked after the launch, never by a wait for it: at the next flush (or the next launch,
    if the host already sees it), as triton-ext's own launcher does."""
    M, T, D, A = _device()
    _checked[(1,)](T.from_numpy(np.ones(64, dtype=np.float32), device="cuda:0"), 64, BLOCK=64)
    M.runtime().flush()                                             # a kept assert: nothing raised
    bad = T.from_numpy(np.full(64, 200.0, dtype=np.float32), device="cuda:0")
    with pytest.raises(Exception, match="100 or more"):
        _checked[(1,)](bad, 64, BLOCK=64)
        M.runtime().flush()
    _checked[(1,)](T.from_numpy(np.ones(64, dtype=np.float32), device="cuda:0"), 64, BLOCK=64)
    M.runtime().flush()                                             # reported once, then clear


def _scalars33(o, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21,
               a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32):
    s = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19
         + a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 + a32 * 1000)
    tl.store(o, s)


_scalars33 = _ours(_scalars33)


def test_a_kernel_of_more_than_31_arguments_reads_every_one():
    """34 arguments, more than Metal's 31 buffer slots: the scalars travel packed in one block."""
    M, T, D, A = _device()
    o = T.zeros((1,), dtype=D.int32, device="cuda:0")
    _scalars33[(1,)](o, *range(1, 33), 7)
    assert int(o.numpy()[0]) == sum(range(1, 33)) + 7000


class _Encoder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a: self.calls.append((name, a))


def _spy(monkeypatch, M):
    enc = _Encoder()
    monkeypatch.setattr(M.runtime(), "encoder_for", lambda stream, kind="compute": (None, enc))
    monkeypatch.setattr(M.runtime(), "note_dispatch", lambda stream: None)
    return enc


def _pipe(max_threads=1024):
    from nbxtensor.backend.metal_dispatch import Pipeline

    class _Pso:
        def maxTotalThreadsPerThreadgroup(self):
            return max_threads
    return Pipeline(_Pso(), "k")


def test_the_threadgroup_memory_the_kernel_declares_is_set(monkeypatch):
    from nbxtensor.backend import metal_dispatch as MD
    M, T, D, A = _device()
    enc = _spy(monkeypatch, M)
    MD.dispatch(_pipe(), 0, [], None, [], (1, 1, 1), (128, 1, 1), shared=4096)
    assert ("setThreadgroupMemoryLength_atIndex_", (4096, 0)) in enc.calls
    enc.calls.clear()
    MD.dispatch(_pipe(), 0, [], None, [], (1, 1, 1), (128, 1, 1))
    assert not any(c[0] == "setThreadgroupMemoryLength_atIndex_" for c in enc.calls)


def test_a_launch_the_pipeline_or_metal_cannot_take_is_refused_by_name(monkeypatch):
    from nbxtensor.backend import metal_dispatch as MD
    M, T, D, A = _device()
    enc = _spy(monkeypatch, M)
    with pytest.raises(RuntimeError, match="asks 1024 threads a threadgroup; its pipeline accepts at most 512"):
        MD.dispatch(_pipe(512), 0, [], None, [], (1, 1, 1), (1024, 1, 1))
    with pytest.raises(RuntimeError, match="binds 32 buffers; Metal's argument table holds 31"):
        MD.dispatch(_pipe(), 0, [0x10] * 32, None, [], (1, 1, 1), (128, 1, 1))
    assert enc.calls == []                                          # refused before anything was encoded


def test_a_grid_barrier_kernel_wider_than_the_gpu_holds_is_refused(monkeypatch):
    from nbxtensor.backend import metal_dispatch as MD
    M, T, D, A = _device()
    _spy(monkeypatch, M)
    monkeypatch.setattr(MD, "_CORES", [16])
    MD.dispatch(_pipe(1024), 0, [], None, [], (128, 1, 1), (128, 1, 1), barrier=True)     # 16 x 8 = 128 fit
    with pytest.raises(RuntimeError, match="a grid of 129; this GPU holds 128 of them at once"):
        MD.dispatch(_pipe(1024), 0, [], None, [], (129, 1, 1), (128, 1, 1), barrier=True)
    MD.dispatch(_pipe(1024), 0, [], None, [], (129, 1, 1), (128, 1, 1))                   # no barrier: any grid


def test_the_gpu_core_count_is_read_from_the_registry():
    from nbxtensor.backend import metal_dispatch as MD
    assert MD.gpu_cores() > 0


def test_a_table_of_captured_addresses_is_read_through():
    from nbxtensor.backend.triton_ext_driver import pinned_addresses, pinned_gpu_address
    M, T, D, A = _device()
    rows = [T.from_numpy(np.full(64, float(e + 1), dtype=np.float32), device="cuda:0") for e in range(4)]
    with pinned_addresses(*rows):
        table = T.from_numpy(np.array([pinned_gpu_address(r.data_ptr()) for r in rows], dtype=np.int64),
                             device="cuda:0")
        o = T.empty((4 * 64,), dtype=D.float32, device="cuda:0")
        _read_through[(4,)](table, o, 64, BLOCK=64)
        got = o.numpy().reshape(4, 64)
    assert np.array_equal(got, np.repeat(np.arange(1, 5, dtype=np.float32)[:, None], 64, axis=1))


def test_an_address_in_no_allocation_is_refused_by_name():
    from nbxtensor.backend import metal_dispatch
    M, T, D, A = _device()
    stray = np.zeros(1024, dtype=np.float32)                        # numpy's memory: no allocation of this runtime
    _cb, enc = M.runtime().encoder_for(0)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: kernel 'k', pointer argument 0"):
        metal_dispatch.bind_pointer(M.runtime(), enc, 0, stray.ctypes.data, "k")
