# Hardware Integration Program

## Partner with NeuroBrix

NeuroBrix is building a **universal AI inference runtime**: a single engine that runs AI models on any hardware. Our goal is hardware-agnostic execution across every GPU, accelerator and compute platform. Hardware support is data: a new device is described by files, not by code branches.

We invite hardware manufacturers to join our **Hardware Integration Program** to ensure their platforms are first-class citizens in the NeuroBrix ecosystem.

---

## Why Integrate with NeuroBrix

| Benefit | Description |
|---------|-------------|
| **Universal Model Access** | Every model in the NeuroBrix Hub becomes instantly available on your hardware |
| **Planning from your facts** | Prism plans each model from the facts your hardware profile declares |
| **Zero Porting Cost** | Model developers never need to write hardware-specific code |
| **Growing Ecosystem** | Tap into NeuroBrix's expanding catalog of AI models across image, LLM, audio, and video |
| **Enterprise Readiness** | NeuroBrix targets enterprise deployments — your hardware gets production-grade AI support |

---

## Integration Tiers

### Tier 1 — Community Profile

**Cost:** Free
**Effort:** Minimal
**What you provide:** A YAML hardware profile describing your device, in the layout of the profiles under
`src/neurobrix/config/hardware/`, and the vendor file of its architecture
(`src/neurobrix/config/vendors/<vendor>/<architecture>.yml`). See
[Writing hardware profiles](../guide/hardware-profiles.md).

```yaml
id: your-gpu-24g
vendor: your-vendor
preferred_dtype: bfloat16
summary:
  total_gpus: 1
  total_vram_gb: 24
  topology: Single-GPU
devices:
- index: 0
  brand: your-vendor
  model: Your GPU Model
  memory_mb: 24576
  compute_capability: '9.0'
  supports_dtypes: [float32, float16, bfloat16]
  architecture: your-architecture
  pcie_version: '4.0'
interconnect:
  groups: []
pcie_fallback:
  version: '4.0'
  lanes: 16
  bandwidth_gbps: 64
```

**What you get:**
- Your hardware listed in NeuroBrix's supported hardware catalog
- Community-contributed Prism solver support
- Execution on the PyTorch branch where PyTorch supports your device

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
- Official benchmarks published on [neurobrix.org](https://neurobrix.org)
- Priority bug fixes for your platform
- Listed as a verified partner on the NeuroBrix website

### Tier 3 — Deep Integration

**Cost:** Sponsorship or engineering partnership
**Effort:** Significant
**What we build together:**

- Triton kernel settings measured and certified on your architecture, in nbxtensor
- A native nbxtensor backend for your device (no PyTorch)
- Custom execution strategies in the Prism solver
- Hardware-specific compilation passes
- Joint documentation and developer guides

**What you get:**
- Everything in Tier 2
- Kernel settings measured on your hardware
- Co-branded case studies and benchmarks
- Joint marketing and event participation
- Early access to NeuroBrix roadmap and beta features
- Direct engineering support channel

---

## Current Hardware Support

NeuroBrix runs on NVIDIA GPUs, on Apple Silicon and on the CPU. Each architecture has a vendor file; each
machine plans from its hardware profile.

| Platform | Status |
|----------|--------|
| NVIDIA Volta (V100 16 GB and 32 GB) | Validated: one card and multi-card (PCIe and NVLink, mixed 16 GB + 32 GB) |
| NVIDIA Ampere (A100), Hopper (H100) | Profiles shipped, not validated by us |
| NVIDIA Turing (T4, RTX 20xx), Ada (L4, L40S, RTX 40xx), Blackwell (B200) | On `main`, not in 0.6.16: profiles from NVIDIA's documentation, not validated by us; these run on the PyTorch branch |
| Apple Silicon (M1 to M5 families) | Triton branch on Metal through nbxtensor, and PyTorch on MPS; validated on an M4 Pro |
| CPU only | Plans and runs on the host, no GPU required |
| AMD (CDNA, RDNA) and Intel GPUs | AMD and Intel GPUs, through the same Triton kernels, are targets not yet validated on a card. |

A card whose architecture has no profile is refused by name, and `--hardware` reads a profile you give.

**Targets:** AMD ROCm and Intel GPUs validated on real cards, Qualcomm AI Engine, custom ASICs.

---

## Integration Process

1. **Contact us** at [partners@neurobrix.org](mailto:partners@neurobrix.org) with:
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
Email: [partners@neurobrix.org](mailto:partners@neurobrix.org)
Web: [neurobrix.org](https://neurobrix.org)
Repository: [GitHub](https://github.com/NeuroBrix/neurobrix) | [GitLab](https://gitlab.com/neurobrix/Neurobrix)

---

Developed by Hocine Benkelaya · Supported and funded by [Neural Network Holding Ltd](https://neuralnetworkholding.com) · Commercial distribution by [WizWorks OÜ](https://wizworks.io).
