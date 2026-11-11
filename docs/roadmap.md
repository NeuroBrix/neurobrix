# NeuroBrix Roadmap

**Roadmap and vision — 2026-11-11. Replaces every earlier version.**

NeuroBrix is not a product. It is an **ecosystem** that gives every GPU a life and uses artificial
intelligence in all its forms: one engine, one tensor library, one grammar engine, one container
format, a catalogue of models, and the tools around them.

## The vision

1. **No hardware is obsolete.** Software walks away from a machine long before its silicon stops
   working. NeuroBrix gives the hardware you already own a new life.
2. **AI is not an API, and not only a language model.** It is a neural network: text, image, audio,
   video, measurement, prediction. One engine runs them all.
3. **The real world needs a function, not a giant.** One model that does one job, on the hardware
   already there.
4. **Small and large alike.** One engine, one library, one set of tools and a catalogue of
   specialised models, for a laptop as for a rack.
5. **Everything is ours, and open.** Our own engine, tensor library, grammar engine, kernels and
   certification, under Apache 2.0. We study other projects and learn from the best of each.
6. **From inference to training.** The ecosystem will reach fine-tuning and training. A model has
   an envelope it was trained for; taking it beyond that envelope calls for a fine-tune, and
   NeuroBrix will do it itself.

The design principles that follow from it: universal (no code that knows a model, a family or a
card), data-driven (every value is carried by the container or by a hardware profile), no hidden
fallback. The engine is designed to refuse only what is truly impossible on a machine: what is
merely slow is meant to run, and the user to be told. The code is built in bricks: each one has a
single responsibility and a written contract.

## Where we stand

Published: **neurobrix 0.6.7**, **nbxtensor 0.1.2**, **nbxgrammar 0.1.1**. The hub lists **49
models** in seven families (language and code, vision-language and omni, speech recognition and
audio understanding, speech synthesis, image generation, video generation, upscaling). Each one
runs through the same command line. There are two engines: the Triton engine, built on nbxtensor
without PyTorch, and an optional PyTorch engine. A release is validated model by model through the
real entry point, with the output looked at; a model that does not yet confirm on one engine or
platform is listed as such. Validated on NVIDIA and Apple GPUs, on Linux, Windows and macOS. `neurobrix serve --http` speaks the OpenAI API, with tools and constrained decoding.

What changed in each version is in [CHANGELOG.md](../CHANGELOG.md). A version number is published
when a batch of work is consistent and stable; small fixes are grouped, and a fix for a broken
model or a wrong value ships without waiting.

## What comes next, in order

Nothing below carries a date. Each step is published when it is proven on real hardware, model by
model, with the output looked at.

1. **Adding a model becomes a declaration.** The tools that build containers are being reorganised
   so that a new model is described by an entry — its identity and the envelope it was made for —
   rather than by new code. This is what lets the catalogue grow quickly without breaking what
   already runs.
2. **New models.** Coding models first, because the agent below needs them; then more
   vision-language, speech-synthesis and image-generation models.
3. **NBX Code** (alias **Nibux**, command `nibux`). A local coding agent written in Rust,
   independent of the model: it will run against `neurobrix serve` on your own machine, with its
   tools confined to a sandbox.
4. **The experimental Intel backend, carried further.** It is in nbxtensor today, written without
   Intel hardware in our lab and labelled as such: experimental, not validated by us, open to
   reports from people who own the cards. AMD GPUs remain a target through the same Triton kernels.
5. **nbxtensor 2.0 in Rust.** The tensor library rewritten in Rust; later, the engine itself. The grammar engine, nbxgrammar, was the first Rust brick.
6. **Training and fine-tuning.** nbxtensor will grow from inference into training, on the same
   certified kernels.

## Work that runs through every step

- **Speed.** Kernel fusion, graph-level simplification (operations that cancel out are never
  computed) and per-model tuning, each one measured against a reproducible benchmark and gated on
  the model's output staying correct.
- **Small machines.** The goal: a model larger than the memory of a machine will be read from disk
  in pieces rather than refused, and the plan will say what it costs in time.
- **Community-certified hardware.** The goal: every GPU class is certified once by its owners,
  shared through a pull request, and served to everyone who has that card.
- **Serving.** The OpenAI-compatible server, long conversations beyond the memory of the card, and
  what a coding agent needs from it.

## Outside this roadmap

[NeuroBrix Studio](https://github.com/NeuroBrix/neurobrix-studio), the desktop application, is
developed in its own repository. It is a client of the engine and never a second way to run models:
the command line and the serving daemon stay the engine's only entry points.
