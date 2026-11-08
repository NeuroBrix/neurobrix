"""An AMD GPU's memory is what rocminfo states, in its unit; an APU is described as unified memory, like an integrated
Intel GPU and an Apple chip; a GPU agent whose memory nothing states is refused by name.

Measured on 13480627: `_parse_rocminfo` read each pool's `Size` as bytes and kept it only above 1e9, but rocminfo prints
KB (ROCm/rocminfo rocminfo.cc amd-staging, `DisplayPoolInfo`: `sz = pool_i->pool_size/1024`, label "KB") — no pool ever
passed, and every AMD card, an MI300X's 192 GB as a laptop's 780M, got an invented 16 384 MB. The rocm-smi path invented
the same figure when sysfs and rocm-smi stated nothing.

The outputs are REAL rocminfo GPU agent blocks, pulled line by line from the issue bodies (indentation and order kept,
trailing padding and some `*******` separators dropped by the fetch, the per-dimension sub-lines of the 1150, 1151,
1100 and 942 blocks left out; every pool Size checked against its hex):
- RADEON_780M: Phoenix gfx1103 APU, https://github.com/ROCm/TheRock/issues/2882 (2026-01-13)
- STRIX_POINT: gfx1150 APU, https://github.com/ollama/ollama/issues/9999 (2025-03-26)
- STRIX_HALO: gfx1151 APU, https://github.com/ROCm/ROCm/issues/5643 (2025-11-08)
- RX_7900_XTX: gfx1100 discrete, https://github.com/ROCm/TheRock/issues/9484 (2026-10-26)
- MI300X: gfx942 discrete (three GLOBAL pools), https://github.com/ROCm/ROCm/issues/4414 (2025-02-25)

What this file does if the code were wrong: a KB size read as bytes, an invented memory, an APU described as a discrete
card or without its largest allocation, a pool other than the first GLOBAL coarse-grained one, gfx1150 as rdna4, a GPU
whose memory nothing states planned on — red.
"""
from __future__ import annotations

import subprocess

import pytest

from neurobrix.core.prism import autodetect

_POOLS = """  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
      Size:                    {size}
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Recommended Granule:2048KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 2
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    {size}
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Recommended Granule:2048KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
{extra}    Pool {group}
      Segment:                 GROUP
      Size:                    64(0x40) KB
      Allocatable:             FALSE
      Alloc Granule:           0KB
      Alloc Recommended Granule:0KB
      Alloc Alignment:         0KB
      Accessible by all:       FALSE
"""

RADEON_780M = """Agent 2
  Name:                    gfx1103
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon 780M Graphics
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    1
  Device Type:             GPU
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      2048(0x800) KB
  Chip ID:                 5567(0x15bf)
  ASIC Revision:           7(0x7)
  Cacheline Size:          128(0x80)
  Max Clock Freq. (MHz):   2700
  BDFID:                   50432
  Internal Node ID:        1
  Compute Unit:            12
  SIMDs per CU:            2
  Shader Engines:          1
  Shader Arrs. per Eng.:   2
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Memory Properties:       APU
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  Workgroup Max Size per Dimension:
    x                        1024(0x400)
    y                        1024(0x400)
    z                        1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    1024(0x400)
  Grid Max Size:           4294967295(0xffffffff)
  Grid Max Size per Dimension:
    x                        2147483647(0x7fffffff)
    y                        65535(0xffff)
    z                        65535(0xffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 67
  SDMA engine uCode::      23
  IOMMU Support::          None
""" + _POOLS.format(size="29360128(0x1c00000) KB", extra="", group=3)

STRIX_POINT = """Agent 2
*******
  Name:                    gfx1150
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    1
  Device Type:             GPU
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      2048(0x800) KB
  Chip ID:                 5390(0x150e)
  ASIC Revision:           4(0x4)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2900
  BDFID:                   25856
  Internal Node ID:        1
  Compute Unit:            16
  SIMDs per CU:            2
  Shader Engines:          1
  Shader Arrs. per Eng.:   2
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Memory Properties:       APU
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    1024(0x400)
  Grid Max Size:           4294967295(0xffffffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 25
  SDMA engine uCode::      11
  IOMMU Support::          None
""" + _POOLS.format(size="24464204(0x1754b4c) KB", extra="", group=3)

STRIX_HALO = """*******
Agent 2
*******
  Name:                    gfx1151
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    1
  Device Type:             GPU
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      2048(0x800) KB
    L3:                      32768(0x8000) KB
  Chip ID:                 5510(0x1586)
  ASIC Revision:           0(0x0)
  Cacheline Size:          128(0x80)
  Max Clock Freq. (MHz):   2900
  BDFID:                   50432
  Internal Node ID:        1
  Compute Unit:            40
  SIMDs per CU:            2
  Shader Engines:          2
  Shader Arrs. per Eng.:   2
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Memory Properties:       APU
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    1024(0x400)
  Grid Max Size:           4294967295(0xffffffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 32
  SDMA engine uCode::      17
  IOMMU Support::          None
""" + _POOLS.format(size="125829120(0x7800000) KB", extra="", group=3)

RX_7900_XTX = """Agent 2
  Name:                    gfx1100
  Uuid:                    GPU-389d0bfc21ba5827
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    1
  Device Type:             GPU
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      6144(0x1800) KB
    L3:                      98304(0x18000) KB
  Chip ID:                 29772(0x744c)
  ASIC Revision:           0(0x0)
  Cacheline Size:          128(0x80)
  Max Clock Freq. (MHz):   2371
  BDFID:                   768
  Internal Node ID:        1
  Compute Unit:            96
  SIMDs per CU:            2
  Shader Engines:          6
  Shader Arrs. per Eng.:   2
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Memory Properties:
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    1024(0x400)
  Grid Max Size:           4294967295(0xffffffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 602
  SDMA engine uCode::      27
  IOMMU Support::          None
""" + _POOLS.format(size="25149440(0x17fc000) KB", extra="", group=3)

MI300X = """Agent 3
  Name:                    gfx942
  Uuid:                    GPU-8af705259683161c
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    2
  Device Type:             GPU
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      4096(0x1000) KB
    L3:                      262144(0x40000) KB
  Chip ID:                 29857(0x74a1)
  ASIC Revision:           1(0x1)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2100
  BDFID:                   7168
  Internal Node ID:        2
  Compute Unit:            304
  SIMDs per CU:            4
  Shader Engines:          32
  Shader Arrs. per Eng.:   1
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Memory Properties:
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          64(0x40)
  Workgroup Max Size:      1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    2048(0x800)
  Grid Max Size:           4294967295(0xffffffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 166
  SDMA engine uCode::      22
  IOMMU Support::          None
""" + _POOLS.format(size="201310208(0xbffc000) KB", group=4, extra="""    Pool 3
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Recommended Granule:2048KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
""")


def _rocminfo(monkeypatch, text):
    def run(cmd, *a, **k):
        if cmd[0] == "rocminfo":
            return subprocess.CompletedProcess(cmd, 0, text, "")
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(autodetect.subprocess, "run", run)


@pytest.mark.parametrize("text, model, gfx, kb, arch", [
    (RX_7900_XTX, "AMD Radeon RX 7900 XTX", "gfx1100", 25149440, "rdna3"),
    (MI300X, "AMD Instinct MI300X", "gfx942", 201310208, "cdna3"),
], ids=["rx_7900_xtx", "mi300x"])
def test_a_discrete_card_has_its_vram_in_its_unit(monkeypatch, text, model, gfx, kb, arch):
    _rocminfo(monkeypatch, text)
    (dev,) = autodetect._parse_rocminfo()
    assert (dev["model"], dev["compute_capability"], dev["architecture"]) == (model, gfx, arch)
    assert dev["memory_mb"] == kb // 1024                    # 24 GiB on the 7900 XTX, 196 592 MB on the MI300X
    assert "unified_memory" not in dev and "max_allocation_mb" not in dev


@pytest.mark.parametrize("text, gfx, kb", [(RADEON_780M, "gfx1103", 29360128), (STRIX_POINT, "gfx1150", 24464204),
                                           (STRIX_HALO, "gfx1151", 125829120)],
                         ids=["radeon_780m", "strix_point", "strix_halo"])
def test_an_apu_is_unified_memory(monkeypatch, text, gfx, kb):
    _rocminfo(monkeypatch, text)
    (dev,) = autodetect._parse_rocminfo()
    assert dev["unified_memory"] is True and dev["memory_mb"] == kb // 1024 == dev["max_allocation_mb"]
    assert dev["compute_capability"] == gfx and dev["architecture"] == "rdna3"      # gfx11 targets, gfx1150 included


def test_two_agents_are_two_devices_and_an_apu_profile_is_one_pool_with_the_host(monkeypatch, tmp_path):
    _rocminfo(monkeypatch, RX_7900_XTX + RX_7900_XTX.replace("Agent 2", "Agent 3"))
    assert [d["index"] for d in autodetect._parse_rocminfo()] == [0, 1]
    _rocminfo(monkeypatch, RADEON_780M)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(autodetect, "_parse_nvidia_smi", lambda: [])
    cpu = {"model": "AMD Ryzen 7 7840U", "cores": 8, "threads": 16, "ram_mb": 32 * 1024, "architecture": "x86_64",
           "features": []}
    monkeypatch.setattr(autodetect, "_detect_cpu", lambda os_type: dict(cpu))
    monkeypatch.setattr(autodetect, "_measure_runtime_base_mb", lambda: {})
    monkeypatch.setattr(autodetect, "_measure_io_workers", lambda: None)
    monkeypatch.setattr(autodetect, "_detect_system_vendor", lambda os_type: "custom")
    monkeypatch.setattr(autodetect.platform, "system", lambda: "Linux")
    (dev,) = autodetect.detect_hardware()["devices"]
    assert dev["host_memory_mb"] == 32 * 1024 and dev["unified_memory"] is True


def test_an_agent_whose_memory_nothing_states_is_refused_by_name(monkeypatch):
    no_pool = RX_7900_XTX.split("  Pool Info:")[0]
    _rocminfo(monkeypatch, no_pool)
    (dev,) = autodetect._parse_rocminfo()
    assert "refused" in dev and "gfx1100" in dev["refused"] and "memory_mb" not in dev
    # the rocm-smi path: no sysfs file states it (this rack's DRM cards are NVIDIA's) and rocm-smi is absent -> unknown
    monkeypatch.setattr(autodetect.subprocess, "run", lambda cmd, *a, **k: (_ for _ in ()).throw(FileNotFoundError(cmd[0])))
    assert autodetect._amd_get_vram_mb(0) is None
