"""The Triton runtime's growth on a UNIFIED device (Apple) — the Dell's 2026-10-28 21:02 question: `runtime_growth`
skipped unified devices, so the M4 Pro never recorded one and its plans kept the 0.92 fraction.

On Metal the driver reading is the allocator's own count (`metal_device.mem_get_info`: budget less OUR live bytes;
`currentAllocatedSize` is cumulative), so `used - live` reads 0 whatever the runtime grew. The runtime's growth (the
kernels' pipeline states, libraries, command buffers) lands in the process's physical footprint: the unified reading
is that footprint less the allocator's live bytes, one figure for the one pool."""
import pytest

from neurobrix.core.runtime import runtime_growth as RG
from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA

MB = 1024 * 1024


def test_a_unified_device_reads_the_footprint_less_the_live_bytes(monkeypatch):
    import neurobrix.core.prism.host_footprint as HF
    monkeypatch.setattr(DA, "_cuda_live_bytes", {0: 3000 * MB})
    monkeypatch.setattr(HF, "process_footprint_now", lambda: 3500 * MB)
    monkeypatch.setattr(DA, "mem_get_info", staticmethod(lambda d=None: (15000 * MB, 18000 * MB)))
    before = RG.untracked_mb([0], unified=True)
    assert before == {0: pytest.approx(500.0)}
    # the run loads its kernels: the footprint grows 650 MB beside 1 000 MB more live tensors
    monkeypatch.setattr(DA, "_cuda_live_bytes", {0: 4000 * MB})
    monkeypatch.setattr(HF, "process_footprint_now", lambda: 5150 * MB)
    after = RG.untracked_mb([0], unified=True)
    assert after[0] - before[0] == pytest.approx(650.0)


def test_the_driver_reading_on_metal_says_nothing_of_the_growth(monkeypatch):
    """What the discrete-card reading gives on Metal, whose mem_get_info answers the allocator's own count: 0."""
    monkeypatch.setattr(DA, "_cuda_live_bytes", {0: 4000 * MB})
    monkeypatch.setattr(DA, "mem_get_info", staticmethod(lambda d=None: (14000 * MB, 18000 * MB)))
    assert RG.untracked_mb([0]) == {0: pytest.approx(0.0)}
