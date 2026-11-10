"""`NBXTensor.to("cuda:N")` moves nothing, whatever N. Before that contract changes, the calls that name ANOTHER
device than the tensor's are counted (memory/to_device_census.py): off unless NBX_TO_DEVICE_CENSUS names a file,
then one JSON line a process, by caller. It measures; it changes no result.
What this would do if the code were wrong: a count that misses the callers a refusal would break — or an instrument
that costs a run something when nobody asked for it."""
import json

import numpy as np
import pytest

from nbxtensor.memory import to_device_census as census
from nbxtensor.tensor import NBXTensor


class _Dev:
    def __init__(self, kind, index=None):
        self.type, self.index = kind, index


@pytest.mark.parametrize("target, want", [("cuda:3", "cuda:3"), ("cpu", "cpu"), ("cuda", None), ("float16", None),
                                          ("cuda:x", None), (_Dev("cuda", 2), "cuda:2"), (_Dev("cuda"), None),
                                          (_Dev("cpu"), "cpu"), (7, None), (None, None)])
def test_what_names_a_device(target, want):
    assert census.device_named(target) == want


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setattr(census, "PATH", str(tmp_path / "census.jsonl"))
    monkeypatch.setattr(census, "calls", 0)
    monkeypatch.setattr(census, "other", {})
    return tmp_path / "census.jsonl"


def test_a_call_naming_another_device_is_counted_by_caller_and_the_tensor_is_returned_as_before(on):
    t = NBXTensor.from_numpy(np.zeros(4, np.float32))
    here = f"cuda:{t._device_idx}"
    assert t.to("cuda:7") is t and t.to(device="cuda:7") is t and t.to("cpu") is t      # nothing moves, as today
    assert t.to(here) is t and t.to("cuda") is t                                         # its own device, no index
    assert t.to("float32") is t
    assert census.calls == 4, "the four calls that name a device (a bare 'cuda' and a dtype name none)"
    assert sorted(k.split(" ", 1)[1] for k in census.other) == [f"{here} -> cpu", f"{here} -> cuda:7"]
    assert sum(census.other.values()) == 3
    assert all(__file__.rstrip("c") in k for k in census.other), census.other             # the caller of .to
    census.append_counts(str(on))
    line = json.loads(on.read_text())
    assert line["calls"] == 4 and line["other"] == census.other and isinstance(line["pid"], int)


def test_unasked_nothing_is_counted(monkeypatch):
    monkeypatch.setattr(census, "PATH", None)
    monkeypatch.setattr(census, "calls", 0)
    monkeypatch.setattr(census, "other", {})
    t = NBXTensor.from_numpy(np.zeros(4, np.float32))
    assert t.to("cuda:7") is t
    assert census.calls == 0 and census.other == {}
