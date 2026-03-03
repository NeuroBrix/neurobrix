# Hardware Profiles

NeuroBrix uses hardware profiles to automatically determine execution strategies. The **Prism solver** reads the profile and allocates model components across available devices.

## Auto-Detection (Default)

When you run NeuroBrix without `--hardware`, it automatically detects your system:

```bash
neurobrix serve --model 1600m-1024
# Detects all GPUs, CPU, RAM, interconnects → creates default.yml
```

Auto-detection covers:

- **GPUs**: NVIDIA, AMD, Intel, Apple Silicon, Tenstorrent, and 5 other vendors
- **CPU**: Model, cores, threads, RAM, instruction set features (AVX2, AVX512, AMX, NEON)
- **Interconnect**: NVLink, XGMI, PCIe topology and bandwidth
- **OS**: Linux, macOS, and Windows

The generated profile is saved to `config/hardware/default.yml` and reused until you delete it.

## Using `--hardware` (Optional Override)

The `--hardware` flag lets you target a **specific subset** of your hardware instead of using the full machine:

```bash
# Use only a single V100 32GB (even if you have 4)
neurobrix serve --model 1600m-1024 --hardware v100-32g

# Use a multi-GPU profile for large models
neurobrix serve --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
```

This is useful when you want to:

- Reserve some GPUs for other workloads
- Test how a model runs on a single GPU vs. multiple
- Match a specific deployment target

## CPU-Only Support

NeuroBrix runs on machines without any GPU. Auto-detection creates a valid CPU-only profile with `devices: []` and `topology: CPU-Only`.

```bash
# Works on a laptop or VM with no GPU
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0
neurobrix chat
```

CPU inference is slower than GPU, but perfectly functional for small models like TinyLlama-1.1B. Great for development, testing, and edge deployment.

## Available Profiles

### Single GPU

| Profile | GPU | VRAM | Use Case |
|---------|-----|------|----------|
| `v100-32g` | V100 32GB | 32 GB | Most models |
| `v100-16g` | V100 16GB | 16 GB | Smaller models |

### Multi-GPU

| Profile | GPUs | Total VRAM | Interconnect |
|---------|------|-----------|--------------|
| `v100-32g-x2-nvlink` | 2x V100 32GB | 64 GB | NVLink |
| `c4140-4xv100-custom-nvlink` | 4x V100 (mixed) | 96 GB | NVLink |
| `c4140-4xv100-16GB-nvlink` | 4x V100 16GB | 64 GB | NVLink |

## How Prism Allocates

The Prism solver uses a **recursive cascade strategy**:

```
single_gpu → lifecycle → pipeline_parallel → component_placement → block_scatter → weight_sharding → component_placement_lazy → lazy_sequential → zero3
```

1. **Single GPU**: Fits everything on one GPU? Done.
2. **Lifecycle**: Load/unload transient components (vision encoder, etc.)
3. **Pipeline Parallel**: Per-layer sequential fill across GPUs (like Accelerate)
4. **Component Placement**: Whole components on different GPUs
5. **Block Scatter**: Block-level best-fit distribution across GPUs
6. **Weight Sharding**: Weight-file round-robin across GPUs
7. **Zero3**: CPU offloading with on-demand GPU transfer

The solver picks the **highest-throughput** strategy that fits in memory.

## Custom Profiles

For a full guide on writing your own hardware profiles — including every YAML field, CPU features, GPU dtypes, interconnect configuration, and CPU-only setups — see [Writing Hardware Profiles](hardware-profiles.md).
