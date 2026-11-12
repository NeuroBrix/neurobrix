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
- `topology.json` — Execution flow and connections
- `components/<name>/runtime.json` — Component attributes
- `components/<name>/graph.json` — TensorDAG
- `defaults.json` — User-tunable parameters
- `hardware/<profile>.yml` — Hardware profile

If a value is needed at runtime, it must come from one of these sources. There are no magic numbers in the engine.

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
        f"This value should be populated during the build phase."
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

**nbxtensor is the ecosystem's torch** (owner rule, 2026-11-12). Everything that touches tensors or follows torch's
logic (tensors, devices, memory, arenas, copies, compute kernels, replay, the tensor side of the experts store) lives in
nbxtensor, never in the engine. nbxtensor is a library with its own role: it replaces torch, and the engine is one of
its users, like any other. The engine keeps policy: Prism, model families, flows, the CLI, serving, which expert sits in
which slot and how many slots Prism grants. Every project of the ecosystem is separate (the engine, nbxtensor,
NBXGrammar, NBX Code, the container builder, the hub), and a function is filed in the project it belongs to. Engine code
that does tensor work is a defect as severe as a wrong answer: its fix ships at once, never held back to be grouped into
a release. The PyTorch engine computes with torch, its own library; that is not a breach.

The brick ratchet (`tools/brick_ratchet.py`, its test
`tests/unit/brick_ratchet/test_the_brick_ratchet_refuses_what_gets_worse.py`) counts the tensor mechanisms each engine
file holds; a count only goes down.

---

## Why These Principles Matter

These constraints produce a runtime that is:

1. **Universal**: The same engine runs diffusion models, LLMs, MoE, audio, and video without modification
2. **Debuggable**: When something fails, the error message points to the exact missing data
3. **Predictable**: No hidden behavior, no silent fallbacks, no model-specific code paths
4. **Maintainable**: Adding a new model family requires zero changes to the runtime
