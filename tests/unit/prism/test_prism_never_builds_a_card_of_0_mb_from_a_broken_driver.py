"""Prism never builds a card of 0 MB from a broken driver: a card whose sharing tool answers nothing is read through
the runtime, and a runtime that refuses the read (error 700) is raised by name, never read as 0 MB free.

Moved from the library's tests (`tests/test_a_device_question_the_runtime_refuses_is_raised_by_name.py`, 2026-11-13),
where it imported this engine across the border; the library keeps its own device questions, the engine keeps the
one that is Prism's: `memory_budget.read_device_sharing`, written here against that public function.

What this stands on: a fake GPU runtime (nbxtensor's own seam, `backend.runtime.device_memory`) that refuses the read
of every card, and no sharing tool (`autodetect.device_sharing` answers None); no device.

What this test does if the code were wrong: a refusing runtime read as 0 MB free (the 2026-11-13 fault), or
swallowed into a reading, returns a DeviceReading instead of raising DeviceQueryError naming the code - red; a healthy
runtime that stopped answering with its real free figure is red too.
"""
import pytest

from nbxtensor.backend import runtime as R
from nbxtensor.memory.allocator import DeviceAllocator, DeviceQueryError

_TABLE = {"device_count": "cudaGetDeviceCount", "set_device": "cudaSetDevice", "mem_get_info": "cudaMemGetInfo",
          "no_device_rc": 100}


class _Runtime:
    def __init__(self, free, refuse):
        self.free, self.refuse, self.current = list(free), dict(refuse), 0

    def device_count(self):
        return 0, len(self.free)

    def get_device(self):
        return 0, self.current

    def set_device(self, i):
        self.current = i
        return 0

    def mem_get_info(self):
        return self.refuse.get(self.current, 0), self.free[self.current], 32 << 30


@pytest.fixture
def runtime(monkeypatch):
    def install(free, refuse=None):
        fake = _Runtime(free, refuse or {})
        monkeypatch.setattr(R, "device_memory", lambda: fake)
        monkeypatch.setattr(R, "backend_table", lambda: _TABLE)
        monkeypatch.setattr(R, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(R, "invalidate_current_device_cache", lambda: None)
        monkeypatch.setattr(DeviceAllocator, "clear_last_error", staticmethod(lambda: None))
        monkeypatch.setattr(DeviceAllocator, "restore_device", staticmethod(lambda prev: setattr(fake, "current", prev)))
        from neurobrix.pytorch.prism import autodetect
        monkeypatch.setattr(autodetect, "device_sharing", lambda index: None)     # no sharing tool answers
        return fake
    return install


def test_prism_never_builds_a_card_of_0_mb_from_a_broken_driver(runtime):
    runtime([5 << 30, 9 << 30], refuse={0: 700, 1: 700})
    from neurobrix.pytorch.prism import memory_budget as B
    with pytest.raises(DeviceQueryError, match=r"error 700"):
        B.read_device_sharing(1)


def test_a_card_a_healthy_runtime_answers_is_read_with_its_real_figures_unmeasured_for_sharing(runtime):
    runtime([5 << 30, 9 << 30])
    from neurobrix.pytorch.prism import memory_budget as B
    r = B.read_device_sharing(1)
    assert r.capacity_mb == 32 << 10 and r.free_mb == 9 << 10
    assert not r.measured
