# Writing Hardware Profiles

Hardware profiles tell NeuroBrix what your machine looks like — GPUs, CPU, memory, interconnects. The **Prism solver** reads the profile and picks the best execution strategy automatically.

## When Do You Need a Custom Profile?

Most users don't. On the first launch on a machine that has no hardware profile, NeuroBrix creates that profile once; every later launch reads it. A profile given with `--hardware` is read as it is. The profile it creates describes the whole machine:

```bash
neurobrix serve --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers
# first launch on this machine: creates ~/.neurobrix/hardware/default.yml once (GPUs, CPU, interconnects)
```

Write a custom profile when you want to:

- **Target specific GPUs** — use only 1 of your 4 GPUs for a specific model
- **Reserve resources** — keep some GPUs free for other workloads
- **Test strategies** — force a particular allocation (e.g., single-GPU vs multi-GPU)
- **Deploy to known hardware** — ship a profile matching your production servers

## Profile Location

Profiles live in two places:

```
~/.neurobrix/hardware/           # your machine's generated profiles ($NEUROBRIX_HOME/hardware)
├── default.yml                  # the whole machine
└── default-<tag>.yml            # a process that sees only some of the cards

neurobrix/config/hardware/       # the committed profiles, inside the installed package
├── v100-32g.yml                 # Single V100 32GB
├── v100-16g.yml                 # Single V100 16GB
└── my-custom.yml                # Your custom profile
```

A generated profile (`default`, `default-<tag>`) found inside the package is refused by name: it belongs in `~/.neurobrix/hardware/`. Any other name given to `--hardware` is read from the package's `config/hardware/`, and `--hardware` takes a profile name, not a path. A pip upgrade or reinstall replaces that folder: keep a copy of your own profile outside the package and put it back after an upgrade.

Use the profile name (without `.yml`) as the `--hardware` argument:

```bash
neurobrix serve --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers --hardware v100-32g
```

## Full Profile Reference

Here is a complete profile with every field explained:

```yaml
# Header comment (convention: describe the hardware)
# Hardware Profile: 2 x Tesla V100-SXM2-32GB

# Unique identifier for this profile
id: my-2xv100-32g

# Server/workstation vendor (auto-detected from DMI/SMBIOS)
vendor: dell

# Default compute dtype for this hardware
# Prism assigns this dtype to models that don't override it
#   float16  — Volta, Turing, Ampere (V100, T4, RTX 20/30/40, A100)
#   bfloat16 — Hopper+ (H100, H200) and CPUs with AMX
#   float32  — CPU fallback (no half-precision acceleration)
preferred_dtype: float16

# Quick summary of the hardware
summary:
  total_gpus: 2
  total_vram_gb: 64.0
  total_ram_gb: 251.5
  topology: Multi-GPU    # "Single-GPU", "Multi-GPU", or "CPU-Only"

# CPU information
cpu:
  model: "Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz"
  cores: 40              # Physical cores
  threads: 80            # Logical threads (hyperthreading)
  ram_mb: 257530         # Total system RAM in MB
  architecture: x86_64   # x86_64, aarch64, arm64
  features:              # Instruction set extensions
    - fma                # Fused multiply-add
    - sse4_1
    - sse4_2
    - avx
    - f16c               # Half-precision conversion
    - avx2               # 256-bit SIMD
    - avx512f            # 512-bit SIMD
    - avx512_vnni        # INT8 inference acceleration

# GPU devices (empty list for CPU-only machines)
devices:
  - index: 0
    brand: nvidia                      # nvidia, amd, intel, apple or tenstorrent
    model: Tesla V100-SXM2-32GB
    memory_mb: 32768                   # VRAM in MB
    compute_capability: "7.0"          # NVIDIA-specific
    supports_dtypes:                   # What this GPU can compute
      - float32
      - float16
    architecture: volta                # volta, ampere, hopper, rdna3, xe, etc.
    pcie_version: "3.0"

  - index: 1
    brand: nvidia
    model: Tesla V100-SXM2-32GB
    memory_mb: 32768
    compute_capability: "7.0"
    supports_dtypes:
      - float32
      - float16
    architecture: volta
    pcie_version: "3.0"

# GPU interconnect topology
interconnect:
  groups:
    - index: 0
      members: [0, 1]       # Device indices in this mesh
      type: full_mesh        # full_mesh or point_to_point
      tech: nvlink           # nvlink, xgmi (AMD), pcie
      bandwidth_gbps: 300    # Peer bandwidth in GB/s

# Fallback when GPUs are not in the same interconnect group
pcie_fallback:
  version: "3.0"
  lanes: 16
  bandwidth_gbps: 32

# Free-text notes (auto-generated or manual)
notes: "2 x Tesla V100-SXM2-32GB, NVLink mesh, Dell C4140"
```

## Field Reference

### Top-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique profile identifier, used internally |
| `vendor` | No | Machine vendor (dell, supermicro, etc.) |
| `preferred_dtype` | No | The compute dtype the planner takes when every device of the profile supports it |
| `summary` | Yes | Quick stats for display and solver hints |
| `cpu` | Yes | CPU information (required even with GPUs) |
| `devices` | Yes | List of GPU devices (empty `[]` for CPU-only) |
| `interconnect` | No | GPU-to-GPU links. Omit for single-GPU or CPU-only. |
| `pcie_fallback` | No | PCIe specs for cross-group GPU communication |
| `notes` | No | Free-text description |

### CPU Features

The `cpu.features` list tells NeuroBrix what accelerated operations are available on CPU:

| Feature | Architecture | Significance |
|---------|-------------|--------------|
| `avx2` | x86_64 | 256-bit SIMD, minimum for reasonable CPU inference |
| `avx512f` | x86_64 | 512-bit SIMD, significant speedup for matrix ops |
| `avx512_vnni` | x86_64 | INT8 dot-product acceleration |
| `amx_bf16`, `amx_int8`, `amx_tile` | x86_64 (Sapphire Rapids+) | Hardware bf16/int8 matrix multiply |
| `avx512_bf16` | x86_64 | bf16 dot products |
| `neon` | aarch64/arm64 | 128-bit SIMD (ARM) |
| `fp16` | arm64 (Apple Silicon) | Native half-precision compute |
| `fma` | x86_64 | Fused multiply-add |
| `f16c` | x86_64 | Half-precision conversion instructions |

### GPU `supports_dtypes`

| Dtype | When to Include |
|-------|----------------|
| `float32` | Always (every GPU supports fp32) |
| `float16` | Volta+ (V100, all RTX, A100, T4, etc.) |
| `bfloat16` | Ampere+ (A100, RTX 30/40, H100) |

### `preferred_dtype` Guidelines

| Hardware | Recommended | Why |
|----------|-------------|-----|
| V100, T4, RTX 20 series | `float16` | No bf16 support |
| A100, RTX 30/40, L40S | `float16` or `bfloat16` | Both work, bf16 avoids overflow |
| H100, H200 | `bfloat16` | Native bf16 tensor cores |
| CPU with AMX | `bfloat16` | Hardware bf16 matrix multiply |
| CPU with NEON + FP16 | `float16` | Apple Silicon half-precision |
| CPU (generic) | `float32` | No half-precision acceleration |

!!! tip "`preferred_dtype`"
    Prism takes it when every device of the profile supports it (`supports_dtypes`).

## CPU-Only Profiles

NeuroBrix runs on machines without any GPU. Small models run on the CPU, for development and testing.

```yaml
# Hardware Profile: CPU-Only Workstation
id: cpu-workstation
preferred_dtype: float32
summary:
  total_gpus: 0
  total_vram_gb: 0
  total_ram_gb: 32.0
  topology: CPU-Only
cpu:
  model: "AMD Ryzen 9 7950X"
  cores: 16
  threads: 32
  ram_mb: 32768
  architecture: x86_64
  features:
    - avx2
    - fma
devices: []
notes: "CPU-only workstation, no discrete GPU"
```

For CPU-only machines:

- `devices` is an empty list `[]`
- `topology` is `CPU-Only`
- `preferred_dtype` is typically `float32` (or `bfloat16` if the CPU has AMX)
- Prism allocates everything to CPU memory
- Models that fit in RAM will run — TinyLlama (1.1B), small diffusion models, etc.
- A model larger than that is planned with the `cpu_streaming` strategy: one component is loaded at a time, so only the largest component must fit in RAM (slower)

!!! tip "CPU performance"
    CPU inference is slower than GPU, but perfectly usable for small models. TinyLlama-1.1B generates coherent text on a modern CPU. Use `neurobrix serve` to avoid reloading weights on each request.

## Single-GPU Profiles

To target one specific GPU on a multi-GPU machine:

```yaml
# Use only GPU 0 (V100 32GB)
id: single-v100-32g
preferred_dtype: float16
summary:
  total_gpus: 1
  total_vram_gb: 32.0
  total_ram_gb: 251.5
  topology: Single-GPU
cpu:
  model: "Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz"
  cores: 40
  threads: 80
  ram_mb: 257530
  architecture: x86_64
  features: [avx2, avx512f, fma]
devices:
  - index: 0
    brand: nvidia
    model: Tesla V100-SXM2-32GB
    memory_mb: 32768
    compute_capability: "7.0"
    supports_dtypes: [float32, float16]
    architecture: volta
    pcie_version: "3.0"
notes: "Single V100 32GB — reserve other GPUs for training"
```

## Mixed-GPU Profiles

NeuroBrix handles heterogeneous GPU configurations. The Prism solver accounts for different VRAM sizes when allocating:

```yaml
devices:
  - index: 0
    brand: nvidia
    model: Tesla V100-SXM2-16GB
    memory_mb: 16384
    # ...
  - index: 1
    brand: nvidia
    model: Tesla V100-SXM2-32GB
    memory_mb: 32768
    # ...
```

Prism plans each GPU with its own memory size, so a component that needs more memory can go to the larger card.

## Interconnect Configuration

The `interconnect` section defines how GPUs communicate. The link bandwidth enters the time Prism estimates for a multi-GPU plan, so a fast link makes a spread plan cheaper.

```yaml
# NVLink mesh (all GPUs can talk directly)
interconnect:
  groups:
    - index: 0
      members: [0, 1, 2, 3]
      type: full_mesh
      tech: nvlink
      bandwidth_gbps: 300

# Partial NVLink (pairs connected, cross-pair via PCIe)
interconnect:
  groups:
    - index: 0
      members: [0, 1]
      type: full_mesh
      tech: nvlink
      bandwidth_gbps: 300
    - index: 1
      members: [2, 3]
      type: full_mesh
      tech: nvlink
      bandwidth_gbps: 300

# PCIe only (no high-speed links)
interconnect:
  groups:
    - index: 0
      members: [0, 1]
      type: full_mesh
      tech: pcie
      bandwidth_gbps: 32
```

Write the bandwidth of your own links, from your vendor's documentation.

## Validating a Profile

After creating a profile, test it:

```bash
# Check that NeuroBrix finds your profile
neurobrix info --hardware

# Test with a model
neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --hardware my-custom
```

A profile NeuroBrix cannot use is refused with a message that names the problem: an unknown GPU brand, a profile name that does not exist (the message lists the profiles it found), or a key the plan needs and the profile lacks.
