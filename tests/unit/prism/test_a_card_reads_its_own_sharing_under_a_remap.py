"""A card's sharing facts are read at ITS board, whatever CUDA_VISIBLE_DEVICES renumbers.

2026-09-27: `memory_budget.read_device_sharing(dev.index)` passes the index the process numbers
its device by, and `autodetect.device_sharing` asked nvidia-smi for that number as a BOARD. Under
`CUDA_VISIBLE_DEVICES=1` the process's device 0 is board 1, and the plan budgeted card 1 with
card 0's occupants (the regression matrix runs one card per job, each remapped to cuda:0). With
the old reading the remapped cases below return board 0's facts: they fail.
"""
import subprocess
import types

import pytest

from neurobrix.core.prism import autodetect

GPUS = ("0, GPU-aaaa0000-1111, Disabled, 9000, 16384\n"
        "1, GPU-bbbb0000-2222, Disabled, 300, 16384\n"
        "2, GPU-cccc0000-3333, Disabled, 0, 32768\n")
APPS = ("GPU-aaaa0000-1111, 111, 8700\n"          # another process holds 8.7 GB on board 0
        "GPU-bbbb0000-2222, 222, 300\n")           # board 1 carries only the reader's own context


@pytest.fixture
def smi(monkeypatch):
    def run(cmd, **_k):
        out = GPUS if any("query-gpu" in c for c in cmd) else APPS
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(autodetect.subprocess, "run", run)
    monkeypatch.setattr(autodetect.os, "getpid", lambda: 222)      # the reader is pid 222


@pytest.mark.parametrize("visible", ["1", "GPU-bbbb0000-2222", "GPU-bbbb", "2,1"])
def test_the_remapped_device_reads_its_own_board(smi, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    idx = 1 if visible == "2,1" else 0
    facts = autodetect.device_sharing(idx)
    assert facts["held_by_others_mb"] == 0.0 and facts["own_context_mb"] == 300.0
    assert facts["free_mb"] == 16384 - 300


def test_no_remap_reads_the_board_itself(smi, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert autodetect.device_sharing(0)["held_by_others_mb"] == 8700.0


@pytest.mark.parametrize("visible,idx", [("7", 0), ("GPU-ffff", 0), ("1", 1)])
def test_a_device_that_maps_to_no_board_is_unread(smi, monkeypatch, visible, idx):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert autodetect.device_sharing(idx) is None
