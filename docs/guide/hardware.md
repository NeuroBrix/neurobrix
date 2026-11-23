# Hardware Profiles

NeuroBrix uses hardware profiles to decide how a model runs. The **Prism solver** reads the profile and places each model component on the available devices.

## The machine's own profile (default)

On the first launch on a machine that has no hardware profile, NeuroBrix creates that profile once; every later launch reads it. A profile given with `--hardware` is read as it is.

```bash
neurobrix serve --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers
# first launch on this machine: creates ~/.neurobrix/hardware/default.yml once
```

The profile describes:

- **GPUs**: NVIDIA, AMD, Intel and Apple GPUs plan on their vendor's profile. Other accelerators that are listed by the system (Tenstorrent and others) are named, and refused by name while no vendor profile exists for them.
- **CPU**: model, cores, threads, RAM, instruction set features (AVX2, AVX-512, AMX, NEON)
- **Interconnect**: NVLink, XGMI, PCIe
- **OS**: Linux, macOS and Windows

The profile is saved to `~/.neurobrix/hardware/default.yml` (`default-<tag>.yml` for a process that sees only some of the cards), and reused until you delete it. `NEUROBRIX_HOME` moves that directory. Committed profiles stay in the package's `config/hardware/`; a generated profile found there is refused by name.

## Using `--hardware` (optional)

The `--hardware` flag names a committed profile instead of the machine's own. Use it to target a **specific subset** of your hardware:

```bash
# Use only a single V100 32GB (even if you have 4)
neurobrix serve --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers --hardware v100-32g

# Use a multi-GPU profile for large models
neurobrix serve --model deepseek-ai/deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
```

This is useful when you want to:

- Reserve some GPUs for other workloads
- Test how a model runs on a single GPU vs. multiple
- Match a specific deployment target

## CPU-Only Support

NeuroBrix runs on machines without any GPU. The machine's profile is then a CPU-only profile with `devices: []` and `topology: CPU-Only`.

```bash
# Works on a laptop or VM with no GPU
neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
neurobrix chat
```

CPU inference is slower than GPU, but it works for small models like TinyLlama-1.1B. It is useful for development and testing.

## Available Profiles

### Our test machines (V100)

| Profile | GPU | VRAM | Use Case |
|---------|-----|------|----------|
| `v100-32g` | V100 32GB | 32 GB | Most models |
| `v100-16g` | V100 16GB | 16 GB | Smaller models |

| Profile | GPUs | Total VRAM | Interconnect |
|---------|------|-----------|--------------|
| `v100-32g-x2-nvlink` | 2x V100 32GB | 64 GB | NVLink |
| `c4140-4xv100-custom-nvlink` | 4x V100 (2x 16 GB, 2x 32 GB) | 96 GB | NVLink |
| `c4140-4xv100-16GB-nvlink` | 4x V100 16GB | 64 GB | NVLink |

### Other committed profiles

The package also ships `a10-24g`, `a100-40g`, `a100-80g`, `h100-80g`, `l40s-48g`, `t4-16g`, `rtx2080ti-11g`, `rtx3060-12g`, `rtx3080-10g`, `rtx3090-24g`, `rtx4080-16g`, `rtx4090-24g` and `cpu-only-x86`. `neurobrix info --hardware` lists every profile on your install. These profiles are shipped, not validated by us.

## How Prism Allocates

Prism tries a set of strategies, from the ones that keep every weight on the GPU to the ones that stream weights or run on the host. On a machine with several GPUs the list is:

```
single_gpu → single_gpu_lifecycle → component_placement → pipeline_parallel → block_scatter → weight_sharding → component_placement_lazy → lazy_sequential → zero3 → op_level_tiling → layer_streaming → cpu_execution → cpu_streaming
```

1. **Single GPU**: everything fits on one GPU.
2. **Single GPU with lifecycle**: components used once (a vision encoder, for example) are loaded and released.
3. **Component placement**, **pipeline parallel**, **block scatter**, **weight sharding**: the model stays in GPU memory, spread across several GPUs.
4. **Op-level tiling**: the model stays on the GPU; the operations that overflow are cut.
5. **Zero3**, **lazy sequential**, **layer streaming**: weights cross from host memory to the GPU as they are needed.
6. **CPU execution** and **CPU streaming**: the host computes.

A machine with one GPU skips the multi-GPU strategies. A CPU-only machine goes straight to tiling, streaming and the CPU strategies.

Prism keeps the plans that fit in memory. It ranks them by what they do with the weights (resident on one GPU, resident across GPUs, tiled, streamed, host), and inside a rank by the plan's estimated time. `neurobrix run --explain-plan` prints the plan it chose and the ones it rejected.

## Custom Profiles

For a full guide on writing your own hardware profiles — including every YAML field, CPU features, GPU dtypes, interconnect configuration, and CPU-only setups — see [Writing Hardware Profiles](hardware-profiles.md).
