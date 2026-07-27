# NeuroBrix Roadmap

**Official roadmap — 2026-07-27. Replaces every earlier version.**

NeuroBrix is a universal deep-learning inference engine: one runtime,
any model, any hardware, zero model-specific code. The goal of this
project is neither fundraising nor a sale. The goal is to **finish the
engine** — performant, model-agnostic, hardware-agnostic.

Where we stand (v0.4.x): 42 published models across 10 categories —
image, video, LLM, code, VLM, omni multimodal understanding, TTS, STT,
speech understanding, upscalers — every model validated in four
execution modes (PyTorch sequential, PyTorch compiled, Triton
sequential, Triton compiled), automatic multi-GPU placement, a
tool-calling agent loop on the serving daemon.

Four phases, in order.

---

## Phase 1 — v0.5: the complete omni family (generative outputs)

v0.4 delivered omni **understanding** — text, image, audio and video
inputs. v0.5 completes the family with **generative outputs**:

- **Speech out** — the talker branch: the model answers with voice.
- **Image out** — the image-generation branch of the omni lineage.
- **Two to three fully validated generative omni models** — never a
  single representative: these are the models that draw users, and
  depth beats tokenism.

House standard unchanged: four execution modes, cross-engine numerical
gates, inspectable artifacts for every closure.

## Phase 2 — AMD: ready-to-light ROCm/CDNA paths

Integrate the ROCm/CDNA code paths cleanly — gated, documented, and
data-driven from vendor/architecture profiles — **without test
execution**, since no AMD GPU is available to the project yet. The code
arrives ready to light the day the hardware does. No support claim
before first light.

## Phase 3 — Metal: Triton on Apple GPUs

A primary goal. The Triton execution mode must run on Apple Metal
GPUs — even if that means building our own Triton-to-Metal path. This
is a large chantier, undertaken with open eyes: it begins with a
sourced state-of-the-art review (Triton upstream, existing Metal
efforts, MLIR backends) and an honest scoping before any line of code.

## Phase 4 — Optimization: benchmarks first, then the kill

**Method before work.** A reproducible benchmark harness on well-known
models — three columns: established runtimes (vLLM, and above all
Ollama) / our PyTorch mode / our Triton mode — with documented
methodology and profiling that says where every millisecond lives. We
optimize **only** what the profile designates, in measured-gain order.

Then the program, in layers — each carrying its own truth gate:

1. **Graph algebra.** A value-flow analysis over the sequential ATen
   trace eliminates what the GPU never needs to compute: constant
   folding, common-subexpression elimination, dead code, identities
   (×1, +0, transpose-of-transpose, full slices), and cancelling
   patterns within a subsequence (+x…−x, ×v…÷v, values known ahead of
   time). Exact identities — integer and shape algebra — are removed
   byte-preservingly. Floating-point cancellations are a real win but
   go through a dedicated drift-gate policy, never claimed byte-equal.
2. **Kernel fusion.** A data-driven pass pattern-matches the simplified
   graph — vertical chains (matmul + element-wise epilogues: bias,
   activation, norm + residual, gated MLPs) and horizontal groups
   (same-shape parallel ops) — and emits fused Triton kernels from
   templates, cached per model. Never a hand-written per-model fusion:
   always the pass that reads the graph.
3. **Execution replay.** The deterministic allocator and symbolic
   shapes let the resolved execution plan be frozen per shape bucket
   and replayed without per-op dispatch — killing the launch tax that
   dominates autoregressive decoding. On the PyTorch side, native CUDA
   Graph capture per bucket is evaluated as well.
4. **The megakernel horizon.** The 2026 research frontier compiles an
   entire block into one persistent kernel with specialized
   producer-consumer warps, reporting 10–50 % latency gains over
   mainstream runtimes. Those systems require a fine-grained model DAG,
   hand-written in their case. **Our sequential trace already IS that
   DAG — for every model** — which turns megakernel synthesis into a
   model-agnostic build pass. Honest, sourced scoping before any line.
5. **Speculative decoding as a mode.** Draft + verify: under greedy
   decoding, verification guarantees token-identical output — the only
   1.3–3× class of speedup that is byte-identical by construction.
   Shipped as an optional mode: faster AND provably identical.

In support: a paged KV cache on the serving daemon, asynchronous weight
prefetch, and compute/transfer overlap on multi-GPU placements.

**Optimization never negotiates truth.** Byte-identity gates the exact
transforms; drift-gates the floating-point ones; the full-zoo
regression battery gates the infrastructure. We have the detailed
graph, our own kernels, and the models' anatomy — every ingredient
needed to be the best, and it will be proven at the benchmark, not in
prose.
