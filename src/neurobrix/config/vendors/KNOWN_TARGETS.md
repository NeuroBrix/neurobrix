# Known compile targets — what this list is for, and what it is NOT for

**This list carries no hard limit, and none may ever be read from it.**

It exists for exactly two things:

1. **Good defaults where no measurement exists yet.** A target nobody has
   certified still has to start somewhere.
2. **Naming a target in a certificate path** —
   `config/autotune/<vendor>/<profile>/<kernel>.<dtype>.json`.

Every hard limit comes from the device. The ceiling on shared memory per block
is asked of the driver (`launcher.max_shared_memory_per_block()`); the cost of a
kernel is asked of the compiler (`launcher.prepare()` → `metadata.shared`). Both
are exact.

**An entry here that contradicts the driver is false by construction**, and the
test that guards this file says so:
`tests/unit/config/test_known_targets_carry_no_limits.py`.

## Why a table cannot decide whether a configuration fits

Measured on this rig, 2026-09-10, one flash-attention tile
(`BLOCK_M=128, BLOCK_N=64, BLOCK_HEADDIM=128`), compiled by Triton 3.6:

| target | shared memory the compiler needs |
|---|---|
| sm_70 | 98 304 bytes (96 KB) |
| sm_86, as reported by a user on an A40 | 164 352 bytes (160 KB) |

The same tile costs 96 KB on one target and 160 KB on another. The cost is a
function of the **target** as much as of the tile, so no table of cards can hold
the answer — not because such a table would be imprecise, but because the
quantity it would have to hold is not a property of the card alone.

That is the defect this file was written after: a card declaring 99 KB was given
a tile sized for one declaring 163, and every language model refused to start on
it. And it was wider than that card — `sdpa_thresholds` exists only in the seven
profiles shipped here, so for **any** card whose profile is not shipped the
pruning did not happen at all.

---

## NVIDIA — the axis is the compute capability

Sources, consulted 2026-09-10: NVIDIA CUDA C Programming Guide, "Compute
Capabilities" and Table 15 *Technical Specifications per Compute Capability*;
NVIDIA CUDA GPUs product list; the Jetson Thor renumbering from NVIDIA developer
documentation; the sm_120/sm_100 incompatibility as reported in
`pytorch/pytorch#159207`.

| capability | family | notes |
|---|---|---|
| 7.0 | Volta | V100 — the rig this engine was built on |
| 7.2 | Xavier | Jetson AGX Xavier |
| 7.5 | Turing | T4, RTX 20xx, Quadro RTX |
| 8.0 | Ampere datacenter | A100 |
| 8.6 | Ampere consumer and professional | **A40**, A10, RTX 30xx — the target whose failure produced this file |
| 8.7 | Jetson Orin | |
| 8.9 | Ada | L4, L40S, RTX 40xx |
| 9.0 | Hopper | H100, H200 |
| 10.0 | Blackwell datacenter | B100, B200 |
| 10.3 | Blackwell Ultra | B300, GB300 |
| 11.0 | Jetson Thor | **spelled `sm_110` by the toolchain.** It was `sm_101` in CUDA 12.8 and 12.9 and was renumbered in CUDA 13.0 — a binary built for one name does not load under the other |
| 12.0 | Blackwell consumer | RTX 50xx, RTX PRO 6000 Blackwell |
| 12.1 | GB10 | the integrated CPU+GPU part; differs from 12.0 by that integration |

**`sm_120` is NOT compatible with `sm_100` binaries.** They are separate
compilation targets despite sharing the Blackwell name — a cubin built for the
datacenter part does not load on the consumer part, and the family name gives no
warning. This is the kind of trap that costs a day.

## AMD — the axis is the gfx target

Sources, consulted 2026-09-10: AMD ROCm compatibility matrix
(`rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html`) and
`ROCm/TheRock` `SUPPORTED_GPUS.md`.

| gfx | family |
|---|---|
| gfx900 – gfx906 | Vega |
| gfx908 | CDNA1 — MI100 |
| gfx90a | CDNA2 — MI210, MI250, MI250X |
| gfx942, gfx950 | CDNA3 — MI300X, MI325X; and MI350/MI355X at gfx950 |
| gfx1010 – gfx1013 | RDNA1 |
| gfx1030 – gfx1036 | RDNA2 |
| gfx1100 – gfx1103 | RDNA3 |
| gfx1150 – gfx1153 | RDNA3.5 |
| gfx1200, gfx1201 | RDNA4 — RX 9060 XT, RX 9070/XT, Radeon AI PRO R9700 |

Noted from the source rather than smoothed over: **ROCm's compatibility matrix
classifies `gfx950` under CDNA3**, while AMD's product material describes the
MI350 series as CDNA4. The gfx target is what the toolchain acts on, so the gfx
target is what this table is keyed on.

## Intel — the ceiling is uniform, and it is queryable

Sources, consulted 2026-09-10: Intel oneAPI GPU Optimization Guide, *Shared
Local Memory*; Intel OpenCL Developer Guide for Processor Graphics, *Memory
Hierarchy*.

I went looking for a per-family table of local-memory ceilings and **there is no
such table to find, because there is no such variation to tabulate.** The
per-work-group ceiling on Intel GPUs is **65 536 bytes (64 KB)**, and it is read
from the device — `CL_DEVICE_LOCAL_MEM_SIZE` in OpenCL,
`device.get_info<sycl::info::device::max_local_mem_size>()` in SYCL.

What DOES vary by family is a different quantity — the SLM capacity per Xe-core,
which a work-group does not get to use alone:

| family | SLM per Xe-core / subslice |
|---|---|
| Xe-LP | 128 KB per subslice |
| Xe-HPG | up to 128 KB |
| Xe2 | 192 KB |
| Xe3 | 256 KB |
| Xe-HPC | 512 KB |

Two work-groups at 64 KB each fill one 128 KB Xe-core. Reading the per-core
capacity as a per-work-group budget would be the A40 defect again, in another
vendor's units.

## Apple

The place is reserved and deliberately empty here: the Metal profiles and their
per-variant files are the Mac's work, and this file does not touch them. The
axis there is the GPU family (`apple7` = M1, `apple8` = M2, `apple9` = M3/M4),
which `config/vendors/apple/apple_silicon.yml` already declares.
