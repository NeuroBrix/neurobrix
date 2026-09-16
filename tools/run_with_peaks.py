#!/usr/bin/env python
"""Run one `neurobrix run` in-process and print, at exit, the peak bytes each
card held — the measurement a Prism estimate is judged against ("a plan is
budgeted under the memory model it is executed under").

Two instruments, one per engine, printed on the same line shape:
  [PEAK] engine=<compiled|triton> cuda:<i> peak=<MB>
The compiled engine's peak is torch's caching-allocator watermark
(`max_memory_allocated`); the triton engine's is the DeviceAllocator's
(`NBX_ALLOC_STATS=1` prints it at exit as `peak_driver`; this tool re-reads
the same counter). torch is imported here only for the compiled arm — this
is a diagnostic under tools/, never under src/.

Usage: run_with_peaks.py [neurobrix run args...]
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    from neurobrix.cli import main as cli_main
    argv = list(sys.argv[1:])
    triton = "--triton" in argv or "--triton-sequential" in argv
    os.environ.setdefault("NBX_ALLOC_STATS", "1")
    sys.argv = ["neurobrix"] + argv
    try:
        rc = cli_main()
    except SystemExit as e:
        rc = int(e.code or 0)
    if triton:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        for d in sorted(DeviceAllocator._cuda_peak_bytes):
            print(f"[PEAK] engine=triton cuda:{d} peak={DeviceAllocator._cuda_peak_bytes[d] / 2**20:.0f}", flush=True)
    else:
        import torch
        for d in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_allocated(d)
            if peak:
                print(f"[PEAK] engine=compiled cuda:{d} peak={peak / 2**20:.0f}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
