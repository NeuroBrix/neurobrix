#!/usr/bin/env python3
"""Host<->GPU DMA health probe (owners' forensic tool, 2026-09-03): warm
pinned H2D / D2H bandwidth, kernel launch latency and fp16 tensor-core
throughput per visible GPU, next to the PCIe link state nvidia-smi
reports. Healthy V100 SXM2 on a Gen3 x16 path: H2D/D2H 10-12 GB/s,
launch ~5-10 us, matmul 60-90 TFLOPS. Measured after the 2026-09-02/03
Bus Fatal resets: links "8GT/s x16 ok" on every port up to the CPU root
port, compute normal, yet H2D 1.0-1.2 GB/s, D2H 0.46-0.49 GB/s, launch
17-23 us — the DMA path degraded ~10x while every link reports healthy.

  python3 tools/pcie_dma_probe.py            # all GPUs
"""
import subprocess
import time

import torch

print(subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,temperature.gpu",
                      "--format=csv"], capture_output=True, text=True).stdout.strip())
for i in range(torch.cuda.device_count()):
    d = f"cuda:{i}"
    a = torch.randn(4096, 4096, device=d, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=d, dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(d); t = time.time()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize(d); tflops = 20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12
    x = torch.empty(1, device=d)
    for _ in range(500):
        x.add_(1.0)
    torch.cuda.synchronize(d); t = time.time()
    for _ in range(5000):
        x.add_(1.0)
    torch.cuda.synchronize(d); launch_us = (time.time() - t) * 1e6 / 5000
    h = torch.empty(256 * 1024 * 1024, dtype=torch.uint8).pin_memory()
    g = torch.empty_like(h, device=d)
    g.copy_(h, non_blocking=True); torch.cuda.synchronize(d); t = time.time()
    for _ in range(4):
        g.copy_(h, non_blocking=True)
    torch.cuda.synchronize(d); h2d = 1.0 / (time.time() - t)
    t = time.time()
    for _ in range(4):
        h.copy_(g, non_blocking=True)
    torch.cuda.synchronize(d); d2h = 1.0 / (time.time() - t)
    # Driver-call latency (the "drifting cudaFree tax" class of the perf
    # register): 1000 x 1 MiB cudaMalloc + cudaFree through the engine's
    # own allocator, pool bypassed. Healthy: tens of us per call.
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    DeviceAllocator.set_device(i)
    ptrs = []
    torch.cuda.synchronize(d); t = time.time()
    for _ in range(1000):
        ptrs.append(DeviceAllocator.malloc_cuda(1 << 20))
    t_m = (time.time() - t) * 1e6 / 1000
    t = time.time()
    for p in ptrs:
        DeviceAllocator._pool_enabled = False
        DeviceAllocator.free_cuda(p)
    t_f = (time.time() - t) * 1e6 / 1000
    print(f"{d}: pinned H2D {h2d:.2f} GB/s | D2H {d2h:.2f} GB/s | launch {launch_us:.1f} us | "
          f"fp16 matmul {tflops:.1f} TFLOPS | cudaMalloc {t_m:.0f} us | cudaFree {t_f:.0f} us")
