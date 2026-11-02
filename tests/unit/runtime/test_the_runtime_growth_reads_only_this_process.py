"""The runtime's growth (`core/runtime/runtime_growth.py`) is measured on what the driver attributes to THIS process,
never on the card's free/total (the supervisor 2026-11-02 09:50): Allegro-TI2V's record read 7 430 MB with a second run
on its card, 654 MB alone. A second process holding 1 GiB on the same card while the reading is taken must not move it.

What this test would do if the code read the card: the reading grows by the neighbour's 1 GiB (red). One card."""
import subprocess
import sys
import textwrap
import time

import pytest

from neurobrix.core.runtime import runtime_growth as RG

pytestmark = pytest.mark.usefixtures("needs_device")

NEIGHBOUR = textwrap.dedent("""
    import sys, time
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.set_device(0)
    p = DA.malloc_cuda(1 << 30, pooled=False)
    print("held", flush=True)
    sys.stdin.read()
""")


def test_a_neighbour_on_the_card_does_not_move_the_reading():
    before = RG.untracked_mb([0])
    if before is None:
        pytest.skip("this backend gives no per-process accounting: the growth is not measured (said by record)")
    n = subprocess.Popen([sys.executable, "-c", NEIGHBOUR], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert n.stdout.readline().strip() == "held"
        time.sleep(0.5)
        during = RG.untracked_mb([0])
    finally:
        n.stdin.close()
        n.wait(timeout=60)
    assert abs(during[0] - before[0]) < 256, (before, during)
