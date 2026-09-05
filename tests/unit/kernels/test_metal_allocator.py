"""The Metal `DeviceAllocator`, on the hardware it was written for.

Step 5 of the Metal adoption plan. `test_device_backend_seam.py` describes
the seam a second implementation slots into and pins the shape of it; this
file pins the implementation itself, and it runs only where there is an
Apple GPU to answer.

Two properties are load-bearing and are pinned first, because getting either
wrong is the difference between a port and a plausible-looking lie:

* **Zero fallback.** Nothing here may quietly return host memory, quietly
  succeed on a device that does not exist, or quietly swallow a bad free.
* **Unified memory is an assumption, not a convenience.** Every address
  handed out is a shared buffer's `contents()`, which is the same address
  the GPU sees ONLY when the memory is shared. A device that says otherwise
  must be refused rather than trusted.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from neurobrix.kernels import metal_device, nbx_tensor
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


def _on_metal() -> bool:
    try:
        return nbx_tensor._detect_gpu_backend() == "metal"
    except Exception:
        return False


metal_only = pytest.mark.skipif(
    not _on_metal(), reason="needs an Apple GPU with unified memory")


# --- the seam's own contract ------------------------------------------------

def test_metal_is_not_a_row_in_the_cuda_hip_table():
    """The premise `test_device_backend_seam.py` pins, from this side.

    `_GPU_BACKENDS` holds symbol tables over one C ABI. Metal is a different
    API, so it gets its own table; adding a third row would have made that
    file's contract assertion false and, worse, would have implied Metal
    supports every key `cuda` does."""
    assert "metal" not in nbx_tensor._GPU_BACKENDS
    assert metal_device.METAL_BACKEND is not None


def test_metal_declares_everything_a_port_must_provide():
    """The same required set `test_the_contract_covers_what_a_port_must_provide`
    demands of every backend, checked against the Metal table."""
    required = {
        "rt_libs", "malloc", "free", "memcpy", "memset",
        "set_device", "get_device", "device_count", "mem_get_info", "sync",
        "malloc_host", "free_host",
    }
    missing = required - set(metal_device.METAL_BACKEND)
    assert not missing, f"Metal does not declare {sorted(missing)}"


def test_every_declared_entry_point_exists_on_the_runtime():
    """A name in the table with nothing behind it is an AttributeError at the
    first allocation on a machine nobody has."""
    if not _on_metal():
        pytest.skip("needs an Apple GPU")
    runtime = metal_device.runtime()
    for key, name in metal_device.METAL_BACKEND.items():
        if key == "rt_libs":
            continue
        assert callable(getattr(runtime, name, None)), (
            f"{key!r} names {name!r}, which the runtime does not provide")


def test_metal_device_module_imports_no_torch():
    """R33 — the rule that made this port a bounded job."""
    import inspect
    source = inspect.getsource(metal_device)
    assert "import torch" not in source
    assert "torch." not in source


# --- zero fallback ----------------------------------------------------------

def test_no_metal_means_a_raise_not_a_cpu_backend(monkeypatch):
    """The whole point of the seam refusing loudly.

    With CUDA, ROCm and Metal all unavailable, detection must RAISE. A silent
    CPU path here would be the engine computing on the wrong device under a
    device's name — the failure mode Zero Fallback exists to forbid."""
    monkeypatch.setattr(metal_device, "metal_device_available", lambda: False)
    nbx_tensor._detect_gpu_backend.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="No GPU runtime found"):
            nbx_tensor._detect_gpu_backend()
    finally:
        nbx_tensor._detect_gpu_backend.cache_clear()


def test_a_device_without_unified_memory_is_refused():
    """Every address here is a shared buffer's `contents()`. That is the same
    address the GPU sees only on unified memory; on a discrete Metal GPU the
    pointer would be silently wrong, so construction must refuse."""
    if not _on_metal():
        pytest.skip("needs an Apple GPU")

    real = metal_device.MetalRuntime.__init__
    device = metal_device.runtime()._device

    class _Discrete:
        def __getattr__(self, name):
            return getattr(device, name)

        def hasUnifiedMemory(self):
            return False

    import Metal
    original = Metal.MTLCreateSystemDefaultDevice
    Metal.MTLCreateSystemDefaultDevice = lambda: _Discrete()
    metal_device.reset_runtime_for_tests()
    try:
        with pytest.raises(metal_device.MetalUnavailableError,
                           match="unified memory"):
            metal_device.MetalRuntime()
    finally:
        Metal.MTLCreateSystemDefaultDevice = original
        metal_device.reset_runtime_for_tests()
        assert real is metal_device.MetalRuntime.__init__


@metal_only
def test_freeing_a_pointer_we_never_handed_out_is_loud():
    """CUDA answers `cudaErrorInvalidValue`; silence here would hide a
    double free until it corrupted something far away."""
    runtime = metal_device.runtime()
    assert runtime.free(ctypes.c_void_p(0xDEADBEEF)) != 0


@metal_only
def test_a_second_device_index_is_refused_not_clamped():
    """`MTLCopyAllDevices()` reports one GPU here. Accepting `set_device(1)`
    by quietly using device 0 would make a multi-GPU bug look like success."""
    runtime = metal_device.runtime()
    assert runtime.set_device(ctypes.c_int(0)) == 0
    assert runtime.set_device(ctypes.c_int(1)) != 0


@metal_only
def test_an_allocation_past_the_device_budget_refuses():
    """Metal keeps handing out buffers past `recommendedMaxWorkingSetSize`,
    backed by memory the whole machine shares. The allocator says no at the
    device's own ceiling instead of letting a fill loop take the user's RAM."""
    runtime = metal_device.runtime()
    out = ctypes.c_void_p()
    too_big = runtime._budget_bytes() + (1 << 20)
    assert runtime.malloc(ctypes.byref(out), ctypes.c_size_t(too_big)) != 0
    assert not out.value


# --- allocation, the ordinary path -----------------------------------------

@metal_only
def test_alloc_and_free_return_the_live_count_to_zero():
    runtime = metal_device.runtime()
    start = runtime._live_bytes
    out = ctypes.c_void_p()
    assert runtime.malloc(ctypes.byref(out), ctypes.c_size_t(4096)) == 0
    assert out.value and out.value % 16 == 0, "Triton assumes 16-byte alignment"
    assert runtime._live_bytes == start + 4096
    assert runtime.free(ctypes.c_void_p(out.value)) == 0
    assert runtime._live_bytes == start


@metal_only
def test_live_bytes_survive_repeated_cycles():
    """The counter must come back DOWN. `MTLDevice.currentAllocatedSize()`
    does not — measured 2026-09-05, it read 2/4/6/8 GB across four cycles of
    allocating and releasing the same 2 GiB — which is why the budget is
    checked against this counter and not against the device's."""
    runtime = metal_device.runtime()
    start = runtime._live_bytes
    for _ in range(4):
        pointers = []
        for _ in range(8):
            out = ctypes.c_void_p()
            assert runtime.malloc(ctypes.byref(out),
                                  ctypes.c_size_t(1 << 20)) == 0
            pointers.append(out.value)
        assert runtime._live_bytes == start + 8 * (1 << 20)
        for pointer in pointers:
            assert runtime.free(ctypes.c_void_p(pointer)) == 0
        assert runtime._live_bytes == start


@metal_only
@pytest.mark.parametrize("dtype", ["float32", "float16", "int32", "int64",
                                   "uint8", "bool"])
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 4), (2, 3, 5)])
def test_numpy_round_trip_is_bit_exact(dtype, shape):
    """The copy in and the copy out are the same bytes. On unified memory
    both are a `memmove`, so anything else would be a real defect."""
    rng = np.random.RandomState(20260905)
    if dtype == "bool":
        source = rng.randint(0, 2, size=shape).astype(np.bool_)
    elif dtype.startswith(("int", "uint")):
        source = rng.randint(0, 100, size=shape).astype(dtype)
    else:
        source = (rng.randn(*shape) * 100).astype(dtype)

    tensor = NBXTensor.from_numpy(source)
    got = tensor.numpy()
    assert got.dtype == source.dtype
    assert got.shape == source.shape
    assert np.array_equal(got.view(np.uint8), source.view(np.uint8))


@metal_only
def test_memcpy_is_one_operation_for_every_kind():
    """H2D, D2H, D2D and H2H are the same copy when the memory is shared.
    Pinned to say so: a future change that made only some kinds work would
    be invisible until a specific transfer path ran."""
    payload = np.arange(64, dtype=np.float32)
    host = (ctypes.c_char * payload.nbytes).from_buffer_copy(payload.tobytes())
    device = DeviceAllocator.malloc_cuda(payload.nbytes)
    other = DeviceAllocator.malloc_cuda(payload.nbytes)
    readback = (ctypes.c_char * payload.nbytes)()
    try:
        DeviceAllocator.memcpy(device, ctypes.addressof(host),
                               payload.nbytes, kind=1)        # H2D
        DeviceAllocator.memcpy(other, device, payload.nbytes, kind=3)   # D2D
        DeviceAllocator.memcpy(ctypes.addressof(readback), other,
                               payload.nbytes, kind=2)        # D2H
        got = np.frombuffer(bytes(readback), dtype=np.float32)
        assert np.array_equal(got, payload)
    finally:
        DeviceAllocator.free_cuda(device)
        DeviceAllocator.free_cuda(other)


@metal_only
def test_memset_writes_the_low_byte():
    size = 32
    pointer = DeviceAllocator.malloc_cuda(size)
    try:
        DeviceAllocator.memset_cuda(pointer, 0xAB, size)
        buffer = (ctypes.c_char * size)()
        DeviceAllocator.memcpy(ctypes.addressof(buffer), pointer, size, kind=2)
        assert bytes(buffer) == b"\xab" * size
    finally:
        DeviceAllocator.free_cuda(pointer)


# --- device queries ---------------------------------------------------------

@metal_only
def test_get_device_answers_where_it_used_to_refuse():
    """The line this whole chantier is measured against.

    Before the Metal implementation this raised
    `RuntimeError: No GPU runtime found (tried CUDA and ROCm/HIP)`."""
    assert DeviceAllocator.get_device() == 0
    assert DeviceAllocator.device_count() == 1


@metal_only
def test_memory_info_is_headroom_against_the_device_budget():
    runtime = metal_device.runtime()
    total = DeviceAllocator.device_total_bytes()
    free = DeviceAllocator.device_free_bytes()
    assert total == runtime._budget_bytes() > 0
    assert 0 <= free <= total
    pointer = DeviceAllocator.malloc_cuda(64 * 1024 * 1024)
    try:
        assert DeviceAllocator.device_free_bytes() <= free - 64 * 1024 * 1024
    finally:
        DeviceAllocator.free_cuda(pointer)
        DeviceAllocator.empty_cache_pool()


# --- ordering ---------------------------------------------------------------

@metal_only
def test_streams_are_command_queues_and_handles_start_at_one():
    """Zero is the seam's default stream; a create that returned it would
    silently alias every stream onto the default one."""
    handle = DeviceAllocator.create_stream()
    try:
        assert handle >= 1
        DeviceAllocator.stream_synchronize(handle)
    finally:
        DeviceAllocator.destroy_stream(handle)


@metal_only
def test_destroying_an_unknown_stream_is_loud():
    runtime = metal_device.runtime()
    assert runtime.stream_destroy(ctypes.c_void_p(999999)) != 0


@metal_only
def test_device_sync_is_a_real_round_trip():
    DeviceAllocator.sync_device()      # must not raise


@metal_only
def test_an_ordering_event_refuses_to_be_a_stopwatch():
    """Mirrors the CUDA pin in `test_peer_access.py`: `timing=False` is a
    promise the handle is not a clock, and reading it as one must fail rather
    than return a plausible number."""
    start = DeviceAllocator.create_event(timing=False)
    end = DeviceAllocator.create_event(timing=False)
    try:
        DeviceAllocator.record_event(start, 0)
        DeviceAllocator.record_event(end, 0)
        DeviceAllocator.event_synchronize(end)
        with pytest.raises(RuntimeError):
            DeviceAllocator.event_elapsed_ms(start, end)
    finally:
        DeviceAllocator.destroy_event(start)
        DeviceAllocator.destroy_event(end)


@metal_only
def test_a_timing_event_measures_on_the_gpu_timeline():
    """The interval comes from `MTLCommandBuffer.GPUEndTime`, a GPU-timeline
    timestamp, not a host clock read around the call."""
    start = DeviceAllocator.create_event(timing=True)
    end = DeviceAllocator.create_event(timing=True)
    try:
        DeviceAllocator.record_event(start, 0)
        DeviceAllocator.record_event(end, 0)
        DeviceAllocator.event_synchronize(end)
        assert DeviceAllocator.event_elapsed_ms(start, end) >= 0.0
    finally:
        DeviceAllocator.destroy_event(start)
        DeviceAllocator.destroy_event(end)


@metal_only
def test_a_stream_can_wait_on_an_event_without_blocking_the_host():
    """`encodeWaitForEvent:` is the Metal counterpart of
    `cudaStreamWaitEvent`, and it is what makes cross-queue ordering real
    rather than a host sync in disguise."""
    event = DeviceAllocator.create_event(timing=False)
    stream = DeviceAllocator.create_stream()
    try:
        DeviceAllocator.record_event(event, 0)
        DeviceAllocator.stream_wait_event(stream, event)
        DeviceAllocator.stream_synchronize(stream)
    finally:
        DeviceAllocator.destroy_event(event)
        DeviceAllocator.destroy_stream(stream)


# --- what Metal does not have ----------------------------------------------

def test_absent_capabilities_are_omitted_rather_than_stubbed():
    """Keys with no honest Metal counterpart are ABSENT, and the seam turns
    absence into a refusal naming the backend. A no-op stub would have been
    the silent wrong this engine exists to avoid."""
    table = metal_device.METAL_BACKEND
    for key in ("malloc_async", "free_async", "memcpy_async",
                "device_can_access_peer", "device_enable_peer_access"):
        assert key not in table, f"{key!r} has no Metal counterpart"


@metal_only
def test_peer_access_answers_false_rather_than_raising():
    """One GPU: there is no peer, and False is the truth rather than a
    degraded answer."""
    assert DeviceAllocator.can_access_peer(0, 0) is True
    assert DeviceAllocator.can_access_peer(0, 1) is False
