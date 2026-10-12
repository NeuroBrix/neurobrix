"""The compiled loader keeps at most `io.pinned_uploads_in_flight` non-blocking uploads outstanding.

A pinned block goes back to PyTorch's host cache only once its copy is done, and that cache never gives
memory back: unbounded, deepseek-moe layer_streaming grew RssShmem 1.0 -> 3.7 GB over one run
(2026-10-10). Every upload is recorded by an event; before a new one, the oldest past the depth is
waited on. Injection: the `len(in_flight) >= PINNED_IN_FLIGHT` wait removed -> the first test goes red
(outstanding reaches every tensor of the shard).
"""
import pytest
import torch
from safetensors.torch import save_file

from neurobrix.core import workspace
from neurobrix.core.io import weight_loader as wl


class _Events:
    """torch.cuda.Event stand-in: counts the uploads recorded and not yet waited on."""

    def __init__(self):
        self.outstanding = 0
        self.most = 0
        self.waits = 0
        ledger = self

        class _Event:
            def record(self):
                ledger.outstanding += 1
                ledger.most = max(ledger.most, ledger.outstanding)

            def synchronize(self):
                ledger.outstanding -= 1
                ledger.waits += 1

        self.Event = _Event


@pytest.fixture
def events(monkeypatch):
    e = _Events()
    monkeypatch.setattr(torch.cuda, "Event", e.Event)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    real_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to",
                        lambda self, *a, **k: self if a and str(a[0]).startswith("cuda") else real_to(self, *a, **k))
    return e


def test_no_more_uploads_are_outstanding_than_configured(events, tmp_path):
    n = 7                                   # more tensors than any depth worth configuring
    save_file({f"w{i}": torch.ones(4, 4) for i in range(n)}, str(tmp_path / "s.safetensors"))
    loader = wl.WeightLoader.__new__(wl.WeightLoader)
    out = loader._load_with_pinned_dma(str(tmp_path / "s.safetensors"), "cuda:0", None, True)
    assert len(out) == n
    assert events.most == wl.PINNED_IN_FLIGHT, (events.most, wl.PINNED_IN_FLIGHT)
    assert events.waits == n - wl.PINNED_IN_FLIGHT


def test_a_missing_depth_is_refused_by_name(monkeypatch, tmp_path):
    y = tmp_path / "system.yml"
    y.write_text("io:\n  num_workers: 8\n")
    monkeypatch.setattr(workspace, "SYSTEM_YML", y)
    with pytest.raises(RuntimeError, match="pinned_uploads_in_flight"):
        workspace.pinned_uploads_in_flight()
