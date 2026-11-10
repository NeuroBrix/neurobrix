"""`NBXTensor.from_numpy(arr, device=...)`: an array placed on a NAMED device selects that device first. Without the
argument it lands on the current device — whatever a call before left current — and `.to("cuda:N")` moves nothing
afterwards (it answers "already on device" whatever N), so `from_numpy(x).to(device=...)`, written in five places,
was right only by the luck of the call before it (the Dell reading the contract, 2026-11-10).
What this would do if the code were wrong: on a plan across two cards, a table or a block uploaded to the other card
and read as its own — a wrong result with no error."""
import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


@pytest.fixture
def order(monkeypatch):
    log, real_set, real_malloc = [], DeviceAllocator.set_device, DeviceAllocator.malloc_cuda
    current = DeviceAllocator.get_device()

    def select(d):
        log.append(("select", int(d)))
        return real_set(current)                       # one device here: the selection is recorded, the card unchanged
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(select))
    monkeypatch.setattr(DeviceAllocator, "malloc_cuda",
                        staticmethod(lambda n, *a, **k: (log.append(("alloc",)), real_malloc(n, *a, **k))[1]))
    return log


@pytest.mark.parametrize("device, index", [(3, 3), ("cuda:2", 2), (0, 0)])
def test_the_named_device_is_selected_before_the_array_s_memory_is_taken(order, device, index):
    t = NBXTensor.from_numpy(np.arange(6, dtype=np.float32), device=device)
    assert order == [("select", index), ("alloc",)]
    assert np.array_equal(t.numpy(), np.arange(6, dtype=np.float32))


def test_without_a_device_nothing_is_selected(order):
    NBXTensor.from_numpy(np.zeros(4, np.float32))
    assert order == [("alloc",)]


@pytest.mark.parametrize("device", ["cuda", "cpu", "cuda:x", -1, True, 2.0])
def test_a_device_that_names_no_index_is_refused_never_read_as_device_0(order, device):
    with pytest.raises(ValueError, match="names no device index"):
        NBXTensor.from_numpy(np.zeros(4, np.float32), device=device)
    assert order == []


def test_no_upload_in_the_library_or_the_triton_engine_is_written_as_from_numpy_then_to_a_device():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    bad = []
    for base in (root / "nbxtensor" / "src", root / "src" / "neurobrix" / "triton", root / "src" / "neurobrix" / "kernels"):
        for f in base.rglob("*.py"):
            for i, line in enumerate(f.read_text().split("\n"), 1):
                if re.search(r"NBXTensor\.from_numpy\([^#]*\)\.to\((device=|f?[\"']cuda)", line):
                    bad.append(f"{f.relative_to(root)}:{i}")
    assert not bad, bad
