"""A replayed launch hands its driver the byte extent of each pointer's tensor, as the launch it recorded did.

The Mac's RUN 40 (2026-11-05, Allegro-TI2V triton on an M4 Pro): at the replay's entry Metal held 7 098 MB, our allocator
7 097 (the 6 275 MB slab and 822 beside); 4 s later, at the 23rd replayed action, Metal held 27 998 MB and our allocator
still 7 097 — and the command buffer died "Insufficient Memory"; the same launches without the replay passed. The Metal
driver binds a pointer by wrapping memory and asks the launcher for each tensor's extent (`wants_extents`), wrapping the
rest of the allocation without it; the recorder kept the parameters and dropped the extents, so every replayed interior
pointer was wrapped to the end of the slab (~0.9 GB per action, held resident by the open command buffer).

What this test does if the code were wrong: the recorder not handed the extents, or the adapter not replaying them —
the driver sees no `extents` (red at the first test); two launches of one kernel with different tensors sharing one
adapter — the second replays the first's extents (red at the second); a driver that binds by address (CUDA) handed an
`extents` it does not take (red at the third). Card-less."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurobrix.triton import replay as R


class _Driver:
    def __init__(self):
        self.calls = []

    def launch(self, function, grid, block, shared, stream, params, **kw):
        self.calls.append((list(params), kw))


@pytest.fixture
def recording(monkeypatch):
    import nbxtensor.backend.launcher as L
    drv = _Driver()
    monkeypatch.setattr(L, "active_driver", lambda: drv)
    monkeypatch.setattr(R.STATE, "recording", True)
    monkeypatch.setattr(R.STATE, "census_current", None)
    monkeypatch.setattr(R.STATE, "records", [])
    monkeypatch.setattr(R.STATE, "note_device", lambda: None)
    monkeypatch.setattr(R, "_NBX_ADAPTERS", {})
    return drv


def _prep():
    return SimpleNamespace(name="k", function=object(), block=(128, 1, 1), shared=0, trailing=None,
                           signature={"x": "*fp16", "y": "*fp16", "n": "i32", "BLOCK": "constexpr"})


def _replay(record):
    adapter, g0, g1, g2, stream, flat = record[1]
    adapter.run(g0, g1, g2, stream, None, None, None, None, None, *flat)


def test_the_replay_hands_the_driver_the_extents_the_launch_had(recording):
    R._nbx_record(_prep(), (4, 1, 1), [("ptr", 0x10000), ("ptr", 0x20000), ("i32", 7)], [4096, 8192, None])
    _replay(R.STATE.records[-1])
    params, kw = recording.calls[-1]
    assert kw.get("extents") == [4096, 8192, None]
    assert params == [("ptr", 0x10000), ("ptr", 0x20000), ("i32", 7)]


def test_two_launches_of_one_kernel_keep_their_own_extents(recording):
    prep = _prep()
    R._nbx_record(prep, (4, 1, 1), [("ptr", 0x10000), ("ptr", 0x20000), ("i32", 7)], [4096, 4096, None])
    R._nbx_record(prep, (8, 1, 1), [("ptr", 0x30000), ("ptr", 0x40000), ("i32", 9)], [65536, 65536, None])
    for rec in R.STATE.records:
        _replay(rec)
    assert [kw["extents"] for _p, kw in recording.calls] == [[4096, 4096, None], [65536, 65536, None]]


def test_a_driver_that_binds_by_address_is_handed_no_extents(recording):
    R._nbx_record(_prep(), (4, 1, 1), [("ptr", 0x10000), ("ptr", 0x20000), ("i32", 7)], None)
    _replay(R.STATE.records[-1])
    assert "extents" not in recording.calls[-1][1]


def test_the_launcher_hands_its_recorder_the_extents_it_hands_the_driver():
    import inspect
    import nbxtensor.backend.launcher as L
    body = inspect.getsource(L.launch_prepared) if hasattr(L, "launch_prepared") else inspect.getsource(L)
    i_ext = body.index('extra["extents"] = ')
    i_rec = body.index('_RECORDER(prep, grid, params, extra.get("extents"))')
    assert i_ext < i_rec < body.index("drv.launch(prep.function")
