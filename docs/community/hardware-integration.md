# Hardware Integration Program

## Partner with NeuroBrix

NeuroBrix is building the **universal AI runtime** — a single engine that executes and trains any AI model on any hardware. Our goal is hardware-agnostic execution across every GPU, accelerator, and compute platform.

We invite hardware manufacturers to join our **Hardware Integration Program** to ensure their platforms are first-class citizens in the NeuroBrix ecosystem.

---

## Why Integrate with NeuroBrix

| Benefit | Description |
|---------|-------------|
| **Universal Model Access** | Every model in the NeuroBrix Hub becomes instantly available on your hardware |
| **Automatic Optimization** | The Prism solver automatically selects optimal execution strategies for your platform |
| **Zero Porting Cost** | Model developers never need to write hardware-specific code |
| **Growing Ecosystem** | Tap into NeuroBrix's expanding catalog of AI models across image, LLM, audio, and video |
| **Enterprise Readiness** | NeuroBrix targets enterprise deployments — your hardware gets production-grade AI support |

---

## Integration Tiers

### Tier 1 — Community Profile

**Cost:** Free
**Effort:** Minimal
**What you provide:** A YAML hardware profile describing your device specifications.

```yaml
id: your-gpu-24g
name: "Your GPU 24GB"
gpus:
  - vram_gb: 24
    compute_capability: "9.0"
    name: "Your GPU Model"
interconnect: null
total_vram_gb: 24
gpu_count: 1
```

**What you get:**
- Your hardware listed in NeuroBrix's supported hardware catalog
- Community-contributed Prism solver support
- Basic execution via PyTorch backend

### Tier 2 — Verified Integration

**Cost:** Free (joint engineering effort)
**Effort:** Moderate
**What we do together:**

- Validate execution correctness across model families
- Optimize Prism solver strategies for your hardware topology
- Benchmark and publish performance profiles
- Test multi-device configurations (NVLink, PCIe, custom interconnects)

**What you get:**
- "Verified by NeuroBrix" badge for your hardware
- Official benchmarks published on [neurobrix.es](https://neurobrix.es)
- Priority bug fixes for your platform
- Listed as a verified partner on the NeuroBrix website

### Tier 3 — Deep Integration

**Cost:** Sponsorship or engineering partnership
**Effort:** Significant
**What we build together:**

- Custom Triton kernels optimized for your architecture
- Native backend support (beyond PyTorch)
- Custom execution strategies in the Prism solver
- Hardware-specific compilation passes
- Joint documentation and developer guides

**What you get:**
- Everything in Tier 2
- Custom kernel optimizations for peak performance
- Co-branded case studies and benchmarks
- Joint marketing and event participation
- Early access to NeuroBrix roadmap and beta features
- Direct engineering support channel

---

## Current Hardware Support

NeuroBrix currently supports NVIDIA GPUs via PyTorch + CUDA + Triton:

| Profile | Configuration |
|---------|---------------|
| V100 16GB | Single GPU |
| V100 32GB | Single, 2x, 3x (PCIe), 2x/4x (NVLink) |
| Custom multi-GPU | Mixed VRAM configurations with NVLink |

**On the roadmap:** AMD ROCm, Intel oneAPI, Apple Metal, Qualcomm AI Engine, custom ASIC support.

---

## Integration Process

1. **Contact us** at [partners@neurobrix.es](mailto:partners@neurobrix.es) with:
   - Your hardware specifications (GPU model, VRAM, interconnects)
   - Your target integration tier
   - Your timeline and engineering resources available

2. **Technical assessment** — Our team evaluates compatibility and defines the integration scope.

3. **Joint development** — We work together on profiles, solver strategies, and optimizations.

4. **Validation and benchmarking** — End-to-end testing across model families.

5. **Publication** — Results published on NeuroBrix Hub with your hardware as a verified platform.

---

## Contact

**Hardware Integration Program**
Email: [partners@neurobrix.es](mailto:partners@neurobrix.es)
Web: [neurobrix.es](https://neurobrix.es)
GitHub: [github.com/NeuroBrix](https://github.com/NeuroBrix)

---

NeuroBrix is developed and maintained by **WizWorks OÜ**, a property of **Neural Networks Holding LTD**.
