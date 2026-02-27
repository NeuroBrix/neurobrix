# Hardware Profiles

NeuroBrix uses hardware profiles to automatically determine execution strategies. The **Prism solver** reads the profile and allocates model components across available GPUs.

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
| `c4140-4xv100-custom-nvlink` | 4x V100 32GB | 128 GB | NVLink |
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

## Using a Profile

```bash
neurobrix serve --model PixArt-Sigma-XL-2-1024-MS --hardware v100-32g
neurobrix serve --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
```

## Custom Profiles

Hardware profiles are YAML files in the NeuroBrix config directory. To create a custom profile, see the existing profiles for the format:

```yaml
name: "my-gpu"
gpus:
  - id: "cuda:0"
    model: "RTX 4090"
    vram_bytes: 25769803776  # 24GB
    supports_dtypes: ["float32", "float16", "bfloat16"]
interconnect: null
```
