"""A kernel's emitter-declared trailing buffers are bound, or the launch refuses.

triton-ext's compiler appends a buffer parameter to a kernel that prints or
asserts on the device, and records it in the compile metadata. `metal_native`
binds the tuple it is given at indices 0..n-1 and never compares that count with
what the kernel declares, so a driver that ignores the metadata leaves the slot
unbound and whatever the kernel writes there is LOST.

Measured 2026-09-17 before the fix: a `tl.device_assert` failing on every lane
ran to completion and raised nothing, while the kernel's ordinary output stayed
correct — a guard paid for and not delivered. Four NeuroBrix kernels are compiled
`@triton.jit(debug=True)` to keep exactly such an assert.

This file pins the contract on BOTH sides, so neither half can rot alone:

  * the seam exists and every driver answers it (no backend, no GPU needed);
  * a driver that cannot bind what a kernel declares REFUSES by name rather
    than launching short;
  * on triton-ext, a failing device assert actually raises.
"""
import pytest

from neurobrix.kernels import launcher as L


class _Meta:
    def __init__(self, **kw):
        self.num_warps = 4
        self.shared = 0
        self.name = "k"
        for k, v in kw.items():
            setattr(self, k, v)


def test_the_base_driver_declares_no_trailing_buffers():
    """CUDA's kernels carry nothing after the caller's arguments, and the
    default must say so rather than raise — a driver that says nothing has to
    behave exactly as it did before the seam existed."""
    assert L.Driver().trailing_buffers(_Meta()) is None


def _selectable_drivers():
    """Every driver the engine can select, base included.

    Listed rather than walked from `Driver.__subclasses__()`, and that is the
    point: the archived fork's `MetalDriver` was DUCK-TYPED against `Driver`
    rather than subclassing it, so a subclass walk would have silently omitted
    the one driver that was actually broken when this seam landed. The next
    duck-typed driver must be added to this list by hand, which is a visible
    act."""
    from neurobrix.triton import triton_ext_driver
    return (L.Driver, L.CudaDriver, triton_ext_driver.TritonExtDriver)


def test_every_driver_answers_the_whole_launcher_protocol():
    """Not just `launch(..., trailing=)` — every method the launcher calls.

    This test exists in this shape because its first version checked only the
    `launch` signature and passed while `MetalDriver` had no `trailing_buffers`
    at all. The launcher calls that in `prepare`, so every kernel on that
    backend raised AttributeError and only `test_launcher_contract` caught it.
    Checking one method of a protocol proves nothing about the protocol.
    """
    import inspect
    protocol = [name for name, _ in inspect.getmembers(L.Driver, inspect.isfunction)
                if not name.startswith("_")]
    assert "trailing_buffers" in protocol and "launch" in protocol, protocol
    for cls in _selectable_drivers():
        for name in protocol:
            assert hasattr(cls, name), (
                f"{cls.__name__} does not answer {name}(), which the launcher "
                f"calls on whatever driver is selected. "
                f"{cls.__name__} is duck-typed against Driver, so inheritance "
                f"does not cover for it.")


def test_every_driver_launch_accepts_the_seam():
    """Whatever `prepare` computes, `launch` is handed back. A driver whose
    signature does not take it would fail at the first launch of any kernel,
    not just one that declares a buffer."""
    import inspect
    for cls in _selectable_drivers():
        params = inspect.signature(cls.launch).parameters
        assert "trailing" in params, f"{cls.__name__}.launch drops the seam"
        assert params["trailing"].default is None, (
            f"{cls.__name__}.launch must default it to None: a caller that "
            f"knows nothing about trailing buffers still launches correctly")


def test_every_driver_declares_nothing_by_default():
    """`trailing_buffers` must be answerable for a kernel that declares no
    extra buffer — the overwhelmingly common case — without touching a GPU."""
    for cls in _selectable_drivers():
        if cls is L.Driver:
            inst = cls()
        else:
            inst = cls.__new__(cls)            # no device, no libcuda, no Metal
        try:
            got = inst.trailing_buffers(_Meta())
        except Exception as exc:                # noqa: BLE001
            raise AssertionError(
                f"{cls.__name__}.trailing_buffers raised on metadata that "
                f"declares nothing: {type(exc).__name__}: {exc}") from exc
        assert got is None, f"{cls.__name__} invented a buffer out of nothing: {got!r}"


def test_a_driver_that_cannot_bind_a_declared_buffer_refuses():
    """The failure that must never be silent. A driver that cannot bind a
    trailing buffer is asked to launch one, and must refuse BY NAME."""

    class _Declared:
        def __repr__(self):
            return "<assert status buffer>"

    for drv in (L.CudaDriver.__new__(L.CudaDriver),):
        with pytest.raises(RuntimeError) as e:
            drv.launch(object(), (1, 1, 1), (32, 1, 1), 0, 0, [],
                       trailing=_Declared())
        assert "trailing" in str(e.value).lower()
        assert "<assert status buffer>" in str(e.value), (
            "the refusal must name WHAT it could not bind, not just that it "
            "could not")


# --------------------------------------------------------------------------
# The behavioural half. It needs the backend whose emitter declares the buffer.
# --------------------------------------------------------------------------

def _ext_or_skip():
    from neurobrix.triton.metal_backend import nbx_driver_module
    mod = nbx_driver_module()
    if "triton_ext" not in mod:
        pytest.skip(f"the declared-buffer ABI is triton-ext's; driver here is {mod}")
    pytest.importorskip("triton_apple_backend")


def test_triton_ext_reports_a_failing_device_assert():
    """The measurement that named the defect, as a test.

    `debug=True` is what keeps `tl.device_assert` in the IR at all
    (`triton/language/semantic.py` drops it otherwise), so an assert here is one
    the author demanded. A backend that declares the status buffer and a driver
    that binds it must turn a failed predicate into a raised error.
    """
    _ext_or_skip()
    import numpy as np
    import triton
    import triton.language as tl
    from neurobrix.kernels.nbx_tensor import NBXTensor

    @triton.jit(debug=True)
    def _guarded(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        m = off < n
        v = tl.load(x_ptr + off, mask=m, other=0.0)
        tl.device_assert(v < 1000.0, "v must stay under 1000")
        tl.store(out_ptr + off, v * 2.0, mask=m)

    n = BLOCK = 64

    # The kernel declares the buffer: the seam must see it.
    prep, _ = L.prepare(_guarded,
                        (NBXTensor.from_numpy(np.arange(n, dtype=np.float32)),
                         NBXTensor.zeros((n,), dtype="float32"), n),
                        {"BLOCK": BLOCK})
    assert prep.trailing is not None, (
        "a debug=True kernel carrying tl.device_assert declares an assert "
        "status buffer; the seam returned None, so nothing would be bound")
    assert prep.trailing.assert_layout is not None

    # Predicate holds: ordinary result, no raise.
    ok_in = NBXTensor.from_numpy(np.arange(n, dtype=np.float32))
    ok_out = NBXTensor.zeros((n,), dtype="float32")
    L.launch(_guarded, (1,), ok_in, ok_out, n, BLOCK=BLOCK)
    np.testing.assert_allclose(ok_out.to_cpu().numpy(),
                               2.0 * np.arange(n, dtype=np.float32))

    # Predicate fails on every lane: this MUST raise. Before 2026-09-17 it
    # returned normally and the failure was lost.
    bad_in = NBXTensor.from_numpy(np.full(n, 5000.0, dtype=np.float32))
    bad_out = NBXTensor.zeros((n,), dtype="float32")
    with pytest.raises(Exception) as e:
        L.launch(_guarded, (1,), bad_in, bad_out, n, BLOCK=BLOCK)
    assert "assert" in str(e.value).lower()
    assert "v must stay under 1000" in str(e.value), (
        "the raise must carry the author's own message, which is the only "
        "thing that says WHICH contract broke")


def test_a_kernel_without_an_assert_declares_nothing():
    """The control. Without it the test above would pass on a seam that
    returned a buffer for every kernel, which would be its own defect."""
    _ext_or_skip()
    import numpy as np
    import triton
    import triton.language as tl
    from neurobrix.kernels.nbx_tensor import NBXTensor

    @triton.jit
    def _plain(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        m = off < n
        tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m, other=0.0) * 2.0,
                 mask=m)

    n = BLOCK = 64
    prep, _ = L.prepare(_plain,
                        (NBXTensor.from_numpy(np.arange(n, dtype=np.float32)),
                         NBXTensor.zeros((n,), dtype="float32"), n),
                        {"BLOCK": BLOCK})
    assert prep.trailing is None


def test_the_shared_memory_ceiling_is_asked_for_never_invented():
    """The ceiling the launcher prunes configurations against.

    `TritonExtDriver` read `getattr(hw_constants, "MAX_THREADGROUP_MEMORY",
    32768)` until 2026-09-17 — a name triton-ext does not declare — so the
    literal was taken on EVERY call and the launcher pruned against a number no
    backend had stated. It was right by luck (both the backend's own
    `TG_BUDGET_BYTES` and `MTLDevice.maxThreadgroupMemoryLength` are 32768 on an
    M4 Pro), which is exactly why nothing caught it.

    Pinned here as a property, not as 32768: the number is the device's to say
    and this test must not become another place that states it.
    """
    _ext_or_skip()
    import triton_apple_backend.hw_constants as hw
    from neurobrix.triton.triton_ext_driver import TritonExtDriver

    declared = getattr(hw, "TG_BUDGET_BYTES", None)
    assert declared is not None, (
        "triton-ext no longer declares TG_BUDGET_BYTES; the driver must refuse "
        "rather than fall back, and this test must be re-pointed at whatever "
        "name replaced it")
    got = TritonExtDriver.__new__(TritonExtDriver).max_shared_memory_per_block()
    assert got == int(declared), (
        f"the driver answered {got}, the backend declares {declared}: the "
        f"ceiling must come from the backend, not from this driver")

    # And the name that was being read must stay absent, or the bug's shape
    # comes back silently.
    assert not hasattr(hw, "MAX_THREADGROUP_MEMORY"), (
        "MAX_THREADGROUP_MEMORY now exists upstream; decide deliberately which "
        "of the two is the ceiling instead of letting a getattr default decide")
