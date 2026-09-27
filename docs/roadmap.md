# NeuroBrix Roadmap

**Official roadmap — 2026-09-09. Replaces every earlier version.**

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

Five phases, in order.

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

## Phase 3 — Metal: Triton on Apple GPUs — IN PROGRESS

A primary goal. The Triton execution mode must run on Apple Metal
GPUs — even if that means building our own Triton-to-Metal path. This
is a large workstream, undertaken with open eyes: it begins with a
sourced state-of-the-art review (Triton upstream, existing Metal
efforts, MLIR backends) and an honest scoping before any line of code.

**Where it stands.** First light has passed on the public branch
`metal-first-light`: a complete language model executed end to end on an
Apple GPU, with the engine's own Metal allocator, a vendor-agnostic
launcher behind the same contract the CUDA driver satisfies, and no torch
dependency anywhere in the Triton path. Integration into `main` is under
way, one proven piece at a time, each with its own gate on both kinds of
hardware. Nothing here is claimed as shipped until it is on `main`.

### Two capabilities the Apple census named, queued behind the current chantier

Both come from the 2026-09-22 Apple census (`docs/reference/apple-own-queue-2026-09-24.md`).
Neither is a bug: each is a capability the house family does not have yet, and the doctrine's
answer to a missing capability is to EXTEND NBXTensor and the kernel family, never to reach
back for torch. They wait until the current chantier closes; they are named here so they are
scheduled rather than remembered.

**C1 — `aten::index_put` with value broadcasting.** Today:

    NotImplementedError: aten::index_put values numel 401408 != idx*tail 2048
    and not scalar — value broadcasting unwired

The kernel handles a scalar value and a value whose element count matches `idx * tail`; it
does not handle a value that must BROADCAST across the indexed positions. *Unlocks:*
`Qwen3-VL-30B-A3B-Thinking` in the Triton modes, and with it the deepstack VLM family, whose
vision-token scatter is exactly this pattern. It is also the last op standing between that
model and a census on both machines.

**C2 — the KV-cache path for decode branches.** Today:

    ZERO FALLBACK: decode branches need the KV cache path (no KV wrapper on this session)

A flow whose generation splits into branches reaches decode without a KV wrapper bound to the
session, and refuses rather than degrade. *Unlocks:* `VibeVoice-1.5B`, and more generally any
flow whose decode is entered from more than one branch — the refusal is correct today and the
capability is what removes it. Related to the `triton/flow` stage work that still imports from
`core/flow/stages/` for VibeVoice's DDPM, a documented temporary violation of R33 that this
chantier is the natural moment to close.

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

---

## Release decisions

Items that shape a specific release — an upgrade path, a placement rule, a directory the
release must carry — are recorded in `docs/reference/release-decisions.md`, one dated entry
each with the measurement that raised it, and move to the CHANGELOG when closed.

## Phase 5 — A graphical interface

The engine is driven from a terminal today, and that is a floor on who can
use it. The last phase puts a graphical interface over it: running a model,
seeing what is installed and what the hub carries, following a run while it
happens.

One rule decides its architecture. The CLI and the serving daemon stay the
engine's only entry points; the interface drives them and never opens a
second path into the runtime. A window that reached into the engine directly
would be a second surface to keep correct, and the two would drift.
