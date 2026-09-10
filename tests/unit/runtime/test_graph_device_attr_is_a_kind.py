"""A device attribute in the graph names a KIND, never a card. The index it
carries is the tracer's GPU (48 of the zoo's containers carry `cuda:1`..);
placement belongs to Prism. The sequential dispatcher resolved it literally
when it had no runtime device and refused chatterbox's sequential oracle
under a pinned card ("invalid device ordinal", 2026-09-06)."""
from __future__ import annotations

import pytest
import torch

from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher, placement_device


def _gpu():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu(), reason="needs a GPU")
def test_a_foreign_cuda_index_lands_on_the_current_card():
    cur = torch.cuda.current_device()
    assert placement_device("cuda:7") == torch.device("cuda", cur)
    assert placement_device(torch.device("cuda", 3)) == torch.device("cuda", cur)
    assert placement_device("cpu") == torch.device("cpu")
    d = NativeATenDispatcher()                                   # no runtime device handed over
    assert d._resolve_attr_value({"type": "device", "value": "cuda:7"}) == torch.device("cuda", cur)
    d2 = NativeATenDispatcher(device="cuda:0")                   # Prism's device wins when given
    assert d2._resolve_attr_value({"type": "device", "value": "cuda:7"}) == torch.device("cuda:0")


def test_cpu_stays_cpu_without_a_gpu():
    assert placement_device("cpu") == torch.device("cpu")
