"""No environment variable changes what a kernel computes or how it is launched (3.13, the owner's rule: a value is a fact
of the device file or of a certificate, never a host setting). Six switches overrode a decision their device file
already made, or kept an alternative branch the served path never took: the flash kernel's warps and stages
(NBX_FLASH_NUM_WARPS / NBX_FLASH_NUM_STAGES, over the file's flash rows), the dequantising GEMV's width adaptation
(NBX_DGEMV_ADAPT, over `dequant_gemv.min_programs`), the power-of-two head-dim detour (NBX_D128_DETOUR, over
`flash.pow2_head_dim_correct`), the depthwise convolution route (NBX_DEPTHWISE_DISABLE) and a garbage-collecting retry of
a refused allocation (NBX_GC_ON_OOM, against the rule that a refused malloc is raised, never rescued). They are removed.
What this test does if one comes back: red, naming the file that reads it."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "nbxtensor"
REMOVED = ("NBX_FLASH_NUM_WARPS", "NBX_FLASH_NUM_STAGES", "NBX_DGEMV_ADAPT", "NBX_D128_DETOUR", "NBX_DEPTHWISE_DISABLE",
           "NBX_GC_ON_OOM")


def test_no_removed_switch_is_read():
    read = re.compile(r"""(environ\.get|environ\[|getenv)\(?\s*["'](NBX_[A-Z0-9_]+)""")
    found = [(str(p.relative_to(SRC)), m.group(2)) for p in SRC.rglob("*.py") for m in read.finditer(p.read_text())
             if m.group(2) in REMOVED]
    assert not found, found
