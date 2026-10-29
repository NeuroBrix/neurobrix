"""In a census shadow a value parked on the host is a shadow address, never memory: a piece sliced along a token axis
takes its slice of such a value with a host `contiguous()` (`NBXTensor._contiguous_cpu`, numpy over the pointer), and
mochi-1-preview's census at the 8 GB rung died there with a segmentation fault (2026-10-28). The shadow answers the copy
unread, as every value read.

What this test would do if the code were wrong: the copy reads the shadow address — the subprocess dies by signal (red)."""
import os
import subprocess
import sys

SCRIPT = r"""
from neurobrix.kernels import census
census.install()
from nbxtensor.tensor import NBXTensor, NBXDtype, DeviceAllocator
ptr = DeviceAllocator.malloc_host_pinned(4 * 6 * 4)
host = NBXTensor(ptr, (4, 6), (6, 1), NBXDtype.float32, "cpu", 0, pinned=True)
view = host[:, 1:4]
assert not view.is_contiguous()
out = view.contiguous()
assert tuple(out.shape) == (4, 3) and out.is_contiguous(), out.shape
print("ok")
"""


def test_a_host_copy_of_a_shadow_value_is_answered_unread():
    env = dict(os.environ, NBX_CENSUS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, (r.returncode, r.stderr[-800:])
