#!/usr/bin/env python
"""Run one `neurobrix run` in-process and print, at exit, the peak bytes each
card held — the measurement a Prism estimate is judged against ("a plan is
budgeted under the memory model it is executed under").

Two instruments, one per engine, printed on the same line shapes:
  [PEAK] engine=<compiled|triton> cuda:<i> peak=<MB>
  [PEAK] engine=<compiled|triton> component=<name> cuda:<i> resident=<MB> above=<MB> calls=<n>
The compiled engine's peak is torch's caching-allocator watermark
(`max_memory_allocated`); the triton engine's is the DeviceAllocator's
(`NBX_ALLOC_STATS=1` prints it at exit as `peak_driver`; this tool re-reads
the same counter). torch is imported here only for the compiled arm — this
is a diagnostic under tools/, never under src/.

The per-component line is what Prism's per-component activation figure is judged
against: the watermark is reset once the component's weights are loaded
(`RuntimeExecutor._ensure_weights_loaded`), `resident` is what the card held at that
point (its weights plus whatever earlier components left resident), and `above` is the
largest rise over `resident` across every call of the component
(`RuntimeExecutor._execute_component` — the one entry both engines' flows call). A
component whose weights stream DURING the call (zero3, layer streaming) has those
weights inside `above`; read the plan's strategy before comparing.

Usage: run_with_peaks.py [neurobrix run args...]
"""
from __future__ import annotations

import os
import sys


class _Counters:
    """The two engines' allocator counters behind one shape: live(dev), peak(dev), reset(dev)."""

    def __init__(self, triton: bool):
        self.triton = triton
        self.whole: dict = {}

    def whole_peak(self, d: int) -> int:
        return max(self.whole.get(d, 0), self.peak(d))

    def devices(self):
        if self.triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            return sorted(DeviceAllocator._cuda_live_bytes)
        import torch
        return list(range(torch.cuda.device_count()))

    def live(self, d: int) -> int:
        if self.triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            return DeviceAllocator._cuda_live_bytes.get(d, 0)
        import torch
        return torch.cuda.memory_allocated(d)

    def peak(self, d: int) -> int:
        if self.triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            return DeviceAllocator._cuda_peak_bytes.get(d, 0)
        import torch
        return torch.cuda.max_memory_allocated(d)

    def reset(self, d: int) -> None:
        # the whole-run watermark survives the per-component resets
        self.whole[d] = max(self.whole.get(d, 0), self.peak(d))
        if self.triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            DeviceAllocator.reset_peak_memory(d)
            return
        import torch
        torch.cuda.reset_peak_memory_stats(d)


def _instrument_components(counters: _Counters, record: dict) -> None:
    """Wrap the executor's two component entries; `record[(comp, dev)]` = [resident, above, calls]."""
    from neurobrix.core.runtime.executor import RuntimeExecutor
    load, run = RuntimeExecutor._ensure_weights_loaded, RuntimeExecutor._execute_component
    resident = {}

    def ensure_weights_loaded(self, comp_name):
        out = load(self, comp_name)
        resident[comp_name] = {}
        for d in counters.devices():
            resident[comp_name][d] = counters.live(d)
            counters.reset(d)
        return out

    def execute_component(self, comp_name, *a, **k):
        out = run(self, comp_name, *a, **k)
        for d, base in (resident.get(comp_name) or {}).items():
            rec = record.setdefault((comp_name, d), [base, 0, 0])
            rec[0] = max(rec[0], base)
            rec[1] = max(rec[1], counters.peak(d) - base)
            rec[2] += 1
        return out

    RuntimeExecutor._ensure_weights_loaded = ensure_weights_loaded
    RuntimeExecutor._execute_component = execute_component


def main() -> int:
    from neurobrix.cli import main as cli_main
    argv = list(sys.argv[1:])
    triton = "--triton" in argv or "--triton-sequential" in argv
    engine = "triton" if triton else "compiled"
    os.environ.setdefault("NBX_ALLOC_STATS", "1")
    counters = _Counters(triton)
    record: dict = {}
    _instrument_components(counters, record)
    sys.argv = ["neurobrix"] + argv
    try:
        rc = cli_main()
    except SystemExit as e:
        rc = int(e.code or 0)
    for (comp, d), (base, above, calls) in sorted(record.items()):
        print(f"[PEAK] engine={engine} component={comp} cuda:{d} resident={base / 2**20:.0f} "
              f"above={above / 2**20:.0f} calls={calls}", flush=True)
    for d in counters.devices():
        peak = counters.whole_peak(d)
        if peak:
            print(f"[PEAK] engine={engine} cuda:{d} peak={peak / 2**20:.0f}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
