"""`nbxtensor.memory.shadow.install_shadow` turns a process into a device that is not there: allocations hand out addresses
that address nothing, streams and events are handles, the launcher launches nothing, `shadow_launch` (what stands behind an
autotuned kernel's run) answers None, a value born on the host answers its real value — whatever the arguments
`from_numpy` was given (a stand-in that dropped `device` killed every walked census, 2026-11-12) — a value nobody
knows answers its benign read, and the shadow refuses to install while a device is visible.

`install_shadow` rewires the allocator and the tensor class for the whole process, so each case runs in its own interpreter
with no device in sight (`CUDA_VISIBLE_DEVICES=`; on an Apple host `NBX_CENSUS=1`, the Metal backend's own door).

What these tests would do if the code were wrong: a launch or a device operation reaches the driver (no device: an
error); `from_numpy(arr, dtype, device)` raises a TypeError; a host-born count reads 1 instead of its value; the
device `set_device` named is forgotten; a visible device is accepted — RED."""
import os
import subprocess
import sys
import textwrap


def _shadow(body: str, **env) -> subprocess.CompletedProcess:
    code = "from nbxtensor.memory import shadow\n" + textwrap.dedent(body)
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", NBX_CENSUS="1", **env)
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)


def test_the_driver_surface_answers_with_no_driver():
    r = _shadow("""
        shadow.install_shadow(2)
        from nbxtensor.memory.allocator import DeviceAllocator as DA
        from nbxtensor.backend import launcher
        assert DA.device_count() == 2
        DA.set_device(1)
        assert DA.get_device() == 1 and shadow.shadow_device() == 1
        s, ev = DA.create_stream(), DA.create_event(timing=False)
        assert s and ev and s != ev
        DA.record_event(ev, stream=s); DA.stream_wait_event(0, ev, flags=0); DA.destroy_event(ev); DA.destroy_stream(s)
        DA.sync_direct(); DA.device_synchronize()
        p = DA.malloc_direct(1000)
        assert shadow.shadow_range(p + 10) == (p, 1000)
        assert launcher.launch(object(), (1,)) is None
        assert shadow.shadow_launch(1, 2, grid=3) is None
        shadow.install_shadow(5)                       # once per process: a second install changes nothing
        assert DA.device_count() == 2
        print("SHADOW")
    """)
    assert r.returncode == 0 and "SHADOW" in r.stdout, r.stderr[-1500:]


def test_a_host_born_value_answers_its_value_and_an_unknown_one_its_benign_read():
    r = _shadow("""
        shadow.install_shadow()
        import numpy as np
        from nbxtensor.tensor import NBXTensor
        from nbxtensor.dtype.nbx_dtype import NBXDtype
        frames = NBXTensor.from_numpy(np.array([7], dtype=np.int64), None, 0)     # every argument from_numpy takes
        assert frames.item() == 7 and isinstance(frames.item(), int)
        grid = NBXTensor.from_numpy(np.array([[1, 4, 6]], dtype=np.int64))
        assert grid.numpy().tolist() == [[1, 4, 6]]
        unknown = NBXTensor.empty((2, 3), dtype=NBXDtype.int32)
        assert unknown.numpy().tolist() == [[1, 1, 1], [1, 1, 1]]
        assert NBXTensor.empty((2,), dtype=NBXDtype.float16).numpy().tolist() == [0.0, 0.0]
        assert NBXTensor.empty((1,), dtype=NBXDtype.bool_).item() is True
        assert NBXTensor.empty((1,), dtype=NBXDtype.float32).item() == 0.0
        print("VALUES")
    """)
    assert r.returncode == 0 and "VALUES" in r.stdout, r.stderr[-1500:]


def test_a_visible_device_is_refused():
    r = _shadow("""
        from nbxtensor.memory.allocator import DeviceAllocator as DA
        DA.device_count = staticmethod(lambda: 1)              # a stand-in for a card in sight
        try:
            shadow.install_shadow()
        except shadow.DevicesVisible as e:
            assert "1 device(s) visible" in str(e), e
            assert shadow.shadow_device() == 0
            print("REFUSED")
    """)
    assert r.returncode == 0 and "REFUSED" in r.stdout, r.stderr[-1500:]
