"""zero3's Triton half moves a block through nbxtensor (`nbxtensor.memory.block_upload`): the strategy decides which
block, when and what stays resident; the copy stream, the upload, the readiness event, the compute stream's wait, the
device drain after an eviction and the teardown are the library's (Dell audit doubt D3, boundary audit B row 16).

Card-less: the library's calls are replaced by a recorder, so these cells prove the wiring, not the copies (the copies
are nbxtensor's own test, tests/test_a_weight_block_uploads_on_its_copy_stream.py). The CUDA copy stream itself needs
a discrete card's run under a zero3 plan (Prism never selects zero3 on unified memory).

What these tests would do if the code were wrong: a host weight not handed to the upload, the copy stream not passed,
the readiness event not kept for the wait, the wait or the drain skipped, the stream left open — RED."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nbxtensor.memory import block_upload as BU
from neurobrix.core.strategies.zero3 import Zero3Strategy


class _Host:
    is_cpu = True


class _Seq:
    def __init__(self):
        self.rebound = []

    def rebind_partial(self, m):
        self.rebound.append(dict(m))
        return list(m)

    def recompute_op_devices_for_slots(self, slots):
        pass

    def materialize_slots_depending_on(self, slots):
        pass


@pytest.fixture
def lib(monkeypatch):
    log = []

    def upload_block(host, device, stream=0):
        log.append(("upload", sorted(host), device, stream))
        return {k: ("dev", k) for k in host}, (55 if stream else None)
    monkeypatch.setattr(BU, "upload_block", upload_block)
    monkeypatch.setattr(BU, "wait_for_upload", lambda ev: log.append(("wait", ev)))
    monkeypatch.setattr(BU, "wait_device", lambda: log.append(("drain",)))
    monkeypatch.setattr(BU, "drop_upload_events", lambda evs: log.append(("drop", list(evs))))
    monkeypatch.setattr(BU, "close_copy_stream", lambda s, evs=(): log.append(("close", s, list(evs))))
    monkeypatch.setattr(BU, "open_copy_stream", lambda d: log.append(("open", d)) or 9)
    return log


def _state(stream):
    w0, w1 = _Host(), _Host()
    return {"seq": _Seq(), "is_triton": True, "exec_dev_idx": 1, "transfer_stream": stream,
            "blocks": {0: {"first_op": 0, "last_op": 1, "weight_tensor_ids": ["param::a", "param::b", "param::gone"]}},
            "cpu_originals": {"param::a": w0, "param::b": w1}, "gpu_cache": {}, "block_events": {},
            "resident": set(), "current_block": -1}


def _strategy():
    s = object.__new__(Zero3Strategy)
    s.exec_device = "cuda:1"
    s._ratchet, s._installed = {}, {}
    s._loaded_components, s._pinned_components = set(), set()
    s.context = SimpleNamespace(allocations={})
    return s


def test_a_block_goes_up_on_the_copy_stream_and_the_compute_stream_waits_for_it(lib):
    s, st = _strategy(), _state(stream=9)
    s._prefetch_block(st, 0, True)
    assert lib == [("upload", ["param::a", "param::b"], 1, 9)]
    assert st["gpu_cache"][0] == {"param::a": ("dev", "param::a"), "param::b": ("dev", "param::b")}
    assert st["block_events"] == {0: 55}
    s._wait_for_block(st, 0, True)
    assert lib[-1] == ("wait", 55) and st["block_events"] == {}
    s._evict_block(st, 0, True)
    assert lib[-1] == ("drain",) and st["gpu_cache"] == {}
    assert st["seq"].rebound[-1] == {"param::a": st["cpu_originals"]["param::a"],
                                     "param::b": st["cpu_originals"]["param::b"]}


def test_without_a_copy_stream_the_upload_is_synchronous_and_nothing_is_waited_on(lib):
    s, st = _strategy(), _state(stream=0)
    s._prefetch_block(st, 0, True)
    s._wait_for_block(st, 0, True)
    assert lib == [("upload", ["param::a", "param::b"], 1, 0)] and st["block_events"] == {}


def test_the_pass_end_drops_unwaited_events_and_the_uninstall_closes_the_stream(lib):
    s, st = _strategy(), _state(stream=9)
    s._ratchet["lm"] = st
    s._prefetch_block(st, 0, True)
    s._reset_ratchet("lm")
    assert ("drop", [55]) in lib and st["block_events"] == {}
    s._ratchet["lm"] = st
    s._uninstall("lm")
    assert lib[-1] == ("close", 9, [])
