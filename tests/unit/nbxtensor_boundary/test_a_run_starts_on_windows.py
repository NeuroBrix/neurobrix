"""The four places that stopped every `neurobrix run` on Windows (the audit of 2026-11-07, windows_audit.md 1-4 and 9),
each run here under a simulated Windows — the system's absence of a call (os.preadv, /proc) or its own API (ReadFile,
GetProcessMemoryInfo) stood in by a stub — and the same call unchanged on Linux (the owner's rule via the supervisor
04:50: Windows code is written without a Windows machine, proven by tests that simulate it). No card.

What this test would do if the code were wrong: a weight read needing os.preadv (red), a first-launch measure that
crashes instead of being declared unmeasured (red), a plan needing /proc or libproc for its resident term (red), a BLAS
found only through /proc/self/maps (red), a GPU library named as Linux names it (red)."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _Kernel32:
    """ReadFile with an OVERLAPPED offset, served from the real file behind a fake handle."""

    def __init__(self, files):
        self.files = files

    def ReadFile(self, handle, buf, n, got, ov):
        f = self.files[handle.value]
        ovs = ov._obj
        off = ovs.Offset | (ovs.OffsetHigh << 32)
        f.seek(off)
        data = f.read(n)
        import ctypes
        ctypes.memmove(buf, data, len(data))
        got._obj.value = len(data)
        return 1

    def GetLastError(self):
        return 0

    def GetCurrentProcess(self):
        return -1


def test_the_weights_are_read_at_their_offset_without_preadv(monkeypatch, tmp_path):
    import ctypes
    from nbxtensor.io import safetensors as S
    data = bytes(range(256)) * 4096
    p = tmp_path / "w.bin"
    p.write_bytes(data)
    with open(p, "rb") as f:
        monkeypatch.delattr(os, "preadv", raising=False)
        monkeypatch.delattr(os, "pread", raising=False)
        monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: 777))
        monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32({777: f})), raising=False)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as pool:
            got = S.read_bytes(f.fileno(), 1000, 300000, (4, pool))
    assert got.tobytes() == data[1000:301000]


def test_a_measure_windows_cannot_make_is_declared_unmeasured(monkeypatch, engine_log):
    from nbxtensor.io import read_rate as R
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    with pytest.raises(RuntimeError, match="no uncached read-rate measure on this system"):
        R.workers_for("/nonexistent", witness_dir="/nonexistent")
    from neurobrix.core import workspace as W
    from neurobrix.core.prism import autodetect as A
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    monkeypatch.setattr(W, "_IO_UNMEASURED", None)
    assert A._measure_io_workers() is None
    assert "[Engine] io workers: 1 for this run — no uncached read-rate measure on this system" in engine_log()


def test_the_resident_term_is_the_working_set_on_windows(monkeypatch):
    import ctypes
    from neurobrix.core.prism import host_footprint as H

    class Psapi:
        def GetProcessMemoryInfo(self, process, counters, cb):
            counters._obj.WorkingSetSize = 123456789
            return 1
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32({}), psapi=Psapi()), raising=False)
    assert H.process_footprint_now() == 123456789


def test_the_blas_is_found_beside_numpy_on_windows(monkeypatch, tmp_path):
    from nbxtensor.backend.cpu import blas as B
    root = tmp_path / "site" / "numpy"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    libs = tmp_path / "site" / "numpy.libs"
    libs.mkdir()
    dll = libs / "libscipy_openblas64_-abc.dll"
    dll.write_bytes(b"")
    fake_numpy = types.SimpleNamespace(__file__=str(root / "__init__.py"))
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setattr(B.sys, "platform", "win32")
    assert B._loaded_openblas() == str(dll)


def test_the_gpu_libraries_are_named_as_windows_names_them(monkeypatch):
    import nbxtensor.backend.libnames as L
    monkeypatch.setattr(sys, "platform", "win32")
    win = importlib.reload(L)
    try:
        assert win.CUDA_DRIVER == ["nvcuda.dll"] and win.CUDA_RUNTIME[0] == "cudart64_12.dll"
        assert win.CUDA_RUNTIME_IN_WHEEL == ("bin", "cudart64_12.dll") and win.NVML == ["nvml.dll"]
        assert win.HIP_RUNTIME[0].startswith("amdhip64") and win.LEVEL_ZERO == ["ze_loader.dll"]
    finally:
        monkeypatch.setattr(sys, "platform", "linux")
        lin = importlib.reload(L)
    assert lin.CUDA_DRIVER == ["libcuda.so.1", "libcuda.so"]
    assert lin.CUDA_RUNTIME == ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
    from nbxtensor.backend import runtime
    assert runtime._GPU_BACKENDS["cuda"]["driver_libs"] == ["libcuda.so.1", "libcuda.so"]
