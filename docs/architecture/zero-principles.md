# The ZERO Principles

> NeuroBrix's core design philosophy

## Overview

NeuroBrix is built on three absolute principles that govern every line of code in the runtime. These are not guidelines — they are **invariants** that must never be violated.

---

## ZERO HARDCODE

**No hardcoded values. Everything comes from configuration files.**

```python
# FORBIDDEN
if model_name == "sana":
    sample_size = 128
hidden_dim = 2048
batch_size = 2

# REQUIRED
sample_size = profile.get("sample_size")
if sample_size is None:
    raise ValueError(f"ZERO FALLBACK: 'sample_size' not in {profile_path}")
```

**Valid data sources:**
- The container: `manifest.json`, `topology.json`, `runtime/defaults.json`, `components/<name>/runtime.json`,
  `profile.json`, `graph.json`
- The family config: `config/families/<family>.yml`
- The hardware profile of the machine, and the vendor file of its architecture:
  `config/vendors/<vendor>/<architecture>.yml`. The engine's vendor files hold what the planner prices; the library
  holds what its kernels read.

If a value is needed at run time, it is read from one of these sources. There are no magic numbers in the
engine. A value that is missing is added at its source (the container, the family file or the hardware
profile), never as a default in engine code. New hardware support is a new vendor file, not a Python branch
per architecture.

---

## ZERO FALLBACK

**No default values. Missing data = explicit crash.**

```python
# FORBIDDEN
value = config.get("key", 128)        # Silent default
x = x or 77                           # Implicit fallback
batch_size = batch_size if batch_size else 1

# REQUIRED
value = config.get("key")
if value is None:
    raise ValueError(
        f"ZERO FALLBACK: 'key' not found in config.\n"
        f"Expected in: {config_path}\n"
        f"This value belongs in the container, the family config or the hardware profile."
    )
```

**Why?** Explicit errors allow for fast debugging. A default value can mask a bug for weeks. When the system crashes, the error message tells you exactly what's missing and where it should have been populated.

---

## ZERO SEMANTIC

**The runtime does not know "image", "audio", or "text". It knows tensors.**

```python
# FORBIDDEN (in the runtime)
if family == "image":
    output = decode_image(latents)
elif family == "llm":
    output = generate_text(tokens)

# REQUIRED (abstract semantics)
output = executor.execute(inputs)  # The topology determines the flow
```

| Business Concept (Forbidden) | Abstract Concept (NBX) |
|-----------------------------|------------------------|
| "Image" | Tensor [B, C, H, W] |
| "Latent space" | Tensor [B, Ch, Sh, Sw] |
| "Token sequence" | Tensor [B, SeqLen] |
| "Attention mask" | Tensor [B, 1, Sq, Sk] |
| "Timestep" | Tensor [B] |
| "Guidance scale" | Scalar float |

The runtime executes graphs mechanically. All domain-specific behavior is encoded in the `.nbx` container's topology and execution flow.

---

## Tensor work lives in nbxtensor

Beside the three ZERO principles, one boundary.

**nbxtensor is NeuroBrix's torch.** Everything that touches tensors (tensors, devices, memory, arenas,
copies, compute kernels, replay, the tensor side of the expert store) lives in
[nbxtensor](https://github.com/NeuroBrix/nbxtensor), a separate library, never in the engine. The engine is one
of its users. The engine keeps policy: Prism, model families, flows, the CLI, serving, and which expert sits in
which slot. On the Triton branch, where the PyTorch branch calls torch, the engine calls nbxtensor in the same
place and the same way. The PyTorch branch computes with torch, its own library; that is not a breach.

---

## Why These Principles Matter

These constraints produce a runtime that is:

1. **Universal**: The same engine runs diffusion models, LLMs, MoE, audio, and video without modification
2. **Debuggable**: When something fails, the error message points to the exact missing data
3. **Predictable**: No hidden behavior, no silent fallbacks, no model-specific code paths
4. **Maintainable**: A new model whose flow the engine already knows needs no change to the runtime, only its container
