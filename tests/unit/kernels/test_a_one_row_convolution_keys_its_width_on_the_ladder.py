"""A convolution whose spatial extent is one row is a 1-D convolution over a sequence, and
that sequence is the request's: chatterbox's vocoder keyed every layer on the exact number of
speech frames (1 716 tokens and 1 856 tokens gave two full sets of exact keys, 2026-09-21), so
no certified directory could ever serve a speech of another length. Such a convolution keys
its width on the profile's ladder like a GEMM keys M; the output width in the key is derived
from the input's bucket top by the convolution's own arithmetic, so a certifier synthesising
the input at the top forms the very key the census recorded. A 2-D convolution keeps its
exact extents.

What this test would do if the code were wrong: the one-row keys read 4021 (exact) instead of
4096, or the 2-D key reads a bucket; a certifier's synthetic input at the top then keys
differently from the census and the key stays uncertified forever.

Shapes: width 4021 sits inside the 128-step of the Volta ladder (top 4096); kernel 3 with
padding 1 keeps the width, so the derived output top must equal 4096 too; the 2-D control
uses 37 rows so no rule about one row can fire.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

PROBE = textwrap.dedent("""
    import json, os
    from neurobrix.kernels import census
    census.install(hardware_profile={"devices": [{"brand": "nvidia", "model": "Tesla V100-SXM2-16GB",
                                                  "compute_capability": "7.0", "architecture": "volta"}]})
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    x1 = NBXTensor.empty((1, 64, 1, 4021), "float16", 0)
    w1 = NBXTensor.empty((64, 64, 1, 3), "float16", 0)
    W.conv2d_wrapper(x1, w1, None, (1, 1), (0, 1), (1, 1), False, 0, 1)
    wd = NBXTensor.empty((64, 1, 1, 3), "float16", 0)
    W.conv2d_wrapper(x1, wd, None, (1, 1), (0, 1), (1, 1), False, 0, 64)
    x2 = NBXTensor.empty((1, 64, 37, 4021), "float16", 0)
    W.conv2d_wrapper(x2, w1, None, (1, 1), (0, 1), (1, 1), False, 0, 1)
    print("DONE")
""")


def test_one_row_keys_bucket_their_width_and_two_d_keys_stay_exact(tmp_path):
    rec = tmp_path / "keys"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
           "NBX_KEY_RECORD": str(rec), "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = rec.read_text().splitlines()
    conv = [l for l in lines if "conv2d_forward_kernel::" in l]
    dw = [l for l in lines if "depthwise_conv2d_kernel::" in l]
    assert len(conv) == 2 and len(dw) == 1, lines
    one_row = next(l for l in conv if "(1, 64, 1, " in l)
    two_d = next(l for l in conv if "(1, 64, 37, " in l)
    assert "::(1, 64, 1, 4096, 64, 1, 4096, 1, 3, 1, 1, 0, 1, 1, 1, 1, " in one_row, one_row
    assert "(1, 64, 37, 4021, 64, 37, 4021," in two_d, two_d
    assert dw[0].split("::")[-1].startswith("(64, 1, 4096, 1, 4096, 1, 3,"), dw
