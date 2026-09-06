"""Direct card-to-card access, and the events that order it.

Both of these were found on 2026-09-03 while measuring what a cross-GPU
collective costs, and neither was a tensor-parallelism problem:

* `cudaDeviceEnablePeerAccess` appeared NOWHERE in the engine. Every GPU pair
  on this rig reports NV2 — two bonded NVLinks — and not one cross-device copy
  had ever used them, because the driver silently stages a peer copy through
  host memory when access has not been granted. A 34 MB stage-boundary
  hand-off took 6.44 ms instead of 0.74 ms.

* The triton branch recorded timing-ENABLED events (9.5 us) where the compiled
  branch three lines below used `torch.cuda.Event`, whose default is
  `enable_timing=False` (1.6 us). Both worked, so the asymmetry was invisible.

The tests that need two GPUs skip cleanly on a machine that has one, because
the contract they check is about routing, not about this rig.

Full measurement: validation_outputs/tp_collective_latency_2026_09_03/VERDICT.md
"""

from __future__ import annotations

import pytest

from neurobrix.kernels import nbx_tensor
from neurobrix.kernels.nbx_tensor import DeviceAllocator


def _gpu_count() -> int:
    try:
        return DeviceAllocator.device_count()
    except Exception:                                  # pragma: no cover
        return 0


needs_two_gpus = pytest.mark.skipif(
    _gpu_count() < 2, reason="peer access needs two devices to mean anything")


# --- the contract is data, so a port inherits it ----------------------------

@pytest.mark.parametrize("key", [
    "device_can_access_peer", "device_enable_peer_access",
    "event_create_flags", "event_sync", "event_elapsed",
])
def test_both_backends_declare_the_new_capabilities(key):
    """A key in `cuda` but not in `hip` is a crash on AMD alone. Peer access
    is xGMI there rather than NVLink, but the call and its meaning are the
    same, so it belongs in the table rather than in a branch."""
    for vendor, table in nbx_tensor._GPU_BACKENDS.items():
        assert key in table, f"backend {vendor!r} does not declare {key!r}"


def test_peer_access_is_not_queried_by_hand_anywhere_else():
    """One place decides this. A second call site would drift out of the memo
    and start asking the driver on a hot transfer path."""
    import pathlib
    root = pathlib.Path(nbx_tensor.__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "nbx_tensor.py":
            continue
        text = path.read_text()
        if "EnablePeerAccess" in text or "CanAccessPeer" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        "peer access queried outside DeviceAllocator: " + ", ".join(offenders))


# --- memoisation, in both directions ----------------------------------------

def test_same_device_is_always_reachable():
    assert DeviceAllocator.ensure_peer_access(0, 0) is True


@needs_two_gpus
def test_the_answer_is_remembered_so_the_hot_path_pays_once():
    DeviceAllocator._peer_enabled.discard((0, 1))
    DeviceAllocator._peer_refused.discard((0, 1))
    first = DeviceAllocator.ensure_peer_access(0, 1)
    assert DeviceAllocator.ensure_peer_access(0, 1) is first
    memo = (DeviceAllocator._peer_enabled if first
            else DeviceAllocator._peer_refused)
    assert (0, 1) in memo, "the outcome must be memoised, not re-queried"


@needs_two_gpus
def test_a_refusal_is_remembered_too(monkeypatch):
    """A rig whose cards have no direct link must not re-ask the driver on
    every hand-off — that would make the slow path slower still."""
    DeviceAllocator._peer_enabled.discard((0, 1))
    DeviceAllocator._peer_refused.discard((0, 1))
    monkeypatch.setenv("NBX_DISABLE_PEER_ACCESS", "1")
    assert DeviceAllocator.ensure_peer_access(0, 1) is False
    assert (0, 1) in DeviceAllocator._peer_refused
    DeviceAllocator._peer_refused.discard((0, 1))


# --- routing must not change the bytes --------------------------------------

@needs_two_gpus
def test_a_cross_device_transfer_is_byte_identical_either_way():
    """Peer access changes the ROUTE the driver takes, never the payload.

    The interior-narrow view is the case that used to fault at the
    DeepSeek-Coder-V2-Lite boundary, so it is the one worth re-checking
    whenever the transfer path is touched."""
    import numpy as np

    from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor
    from neurobrix.triton.device_transfer import transfer_tensor

    rs = np.random.RandomState(1234)
    source = rs.randn(1, 16, 64, 128).astype(np.float16)

    def move(peer: bool):
        if peer:
            DeviceAllocator._peer_refused.discard((0, 1))
            DeviceAllocator.ensure_peer_access(0, 1)
        else:
            DeviceAllocator._peer_enabled.discard((0, 1))
            DeviceAllocator._peer_refused.add((0, 1))
        DeviceAllocator.set_device(0)
        t = NBXTensor.zeros(source.shape, dtype=NBXDtype.float16, device="cuda:0")
        DeviceAllocator.memcpy(t.data_ptr(), source.ctypes.data,
                               source.nbytes, kind=1)
        moved = transfer_tensor(t.narrow(2, 7, 23), 1)
        DeviceAllocator.set_device(1)
        DeviceAllocator.sync_device()
        out = np.empty((1, 16, 23, 128), dtype=np.float16)
        DeviceAllocator.memcpy(out.ctypes.data, moved.data_ptr(),
                               out.nbytes, kind=2)
        return out

    expected = np.ascontiguousarray(source[:, :, 7:30, :])
    staged, direct = move(False), move(True)
    assert staged.tobytes() == expected.tobytes()
    assert direct.tobytes() == staged.tobytes(), (
        "enabling peer access changed the transferred bytes — it must only "
        "change the route")


# --- events order streams; they are not stopwatches -------------------------

@pytest.mark.skipif(_gpu_count() < 1, reason="needs a GPU")
def test_an_ordering_event_cannot_be_used_as_a_stopwatch():
    """The point of `timing=False` is that the runtime skips the clock read.
    Asking such an event for an interval must fail loudly rather than return
    a plausible wrong number."""
    DeviceAllocator.set_device(0)
    a = DeviceAllocator.create_event(timing=False)
    b = DeviceAllocator.create_event(timing=False)
    DeviceAllocator.record_event(a, 0)
    DeviceAllocator.record_event(b, 0)
    DeviceAllocator.event_synchronize(b)
    with pytest.raises(RuntimeError):
        DeviceAllocator.event_elapsed_ms(a, b)


@pytest.mark.skipif(_gpu_count() < 1, reason="needs a GPU")
def test_a_timing_event_still_measures():
    DeviceAllocator.set_device(0)
    a = DeviceAllocator.create_event(timing=True)
    b = DeviceAllocator.create_event(timing=True)
    DeviceAllocator.record_event(a, 0)
    DeviceAllocator.record_event(b, 0)
    DeviceAllocator.event_synchronize(b)
    assert DeviceAllocator.event_elapsed_ms(a, b) >= 0.0


def test_the_engine_creates_no_timing_events_for_ordering():
    """Every `create_event` in the engine is an ordering handle; none of them
    reaches `event_elapsed_ms`. A new one that omits `timing=False` is paying
    6x for a clock read nobody reads."""
    import pathlib
    import re

    root = pathlib.Path(nbx_tensor.__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "nbx_tensor.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"create_event\(\s*\)", line):
                offenders.append(f"{path.relative_to(root)}:{number}")
    assert not offenders, (
        "create_event() with default timing, used for ordering only — pass "
        "timing=False (9.5 us -> 1.6 us per record):\n  " + "\n  ".join(offenders))
