"""A probe of one card restores the previous device only when the process had selected it: selecting a device makes the
runtime create its context there, and "restoring" device 0 in a fresh process left 306 MB on card 0 — a card the plan
did not name (the rack, 2026-11-10: a run on a profile naming card 2).

What these tests would do if the code were wrong: the restore is called for a device never selected (card-less: the
recorded calls show set_device(0)); on two cards, a fresh process probing card 1 holds a context on card 0 — RED."""
import subprocess
import sys
import textwrap

import pytest

from nbxtensor.memory import device_memory as DM
from nbxtensor.memory.allocator import DeviceAllocator


def test_a_device_never_selected_is_not_restored(monkeypatch):
    calls = []
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(lambda i: calls.append(int(i))))
    monkeypatch.setattr(DM, "SELECTED", set())
    DeviceAllocator.restore_device(0)
    assert calls == []
    DM.SELECTED.add(3)
    DeviceAllocator.restore_device(3)
    assert calls == [3]


def test_a_fresh_process_probing_one_card_holds_no_context_on_another():
    if DeviceAllocator.device_count() < 2:
        pytest.skip("two cards are needed: the one probed and the one that must stay untouched")
    code = textwrap.dedent('''
        import os, subprocess
        from nbxtensor.memory.allocator import DeviceAllocator as D
        assert D.free_memory_mb(1) is not None
        D.device_total_bytes(1); D.mem_get_info(1)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout
        mine = [l for l in apps.splitlines() if l.split(",")[0].strip() == str(os.getpid())]
        print(len(mine))
    ''')
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-600:]
    assert out.stdout.strip() == "1", f"contexts held by the probing process: {out.stdout.strip()} (one card was probed)"
