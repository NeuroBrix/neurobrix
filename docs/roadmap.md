# NeuroBrix Roadmap

**Roadmap and vision.**

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

The design principles that follow from it: universal (no code that knows a model, a family or a
card), data-driven (every value is read from the container, the family config or the hardware profile), no hidden
fallback. The engine is designed to refuse only what is truly impossible on a machine: what is
merely slow is meant to run, and the user to be told. The code is built in bricks: each one has a
single responsibility and a written contract.

## Where we stand

Published: **neurobrix 0.6.16**, **nbxtensor 0.1.11**, **nbxgrammar 0.1.1**. The main branch carries further work that is not released yet. The hub lists **49
models** in seven families (language and code, vision-language and omni, speech recognition and
audio understanding, speech synthesis, image generation, video generation, upscaling). Each one
runs through the same command line. There are two engines: the Triton engine, built on nbxtensor
without PyTorch, and an optional PyTorch engine. A release is validated model by model through the
real entry point, with the output looked at. Validated on NVIDIA GPUs on Linux and on Apple GPUs on macOS. Windows: support is on main and comes with the next release. `neurobrix serve --http` speaks the OpenAI API, with tools and constrained decoding.

What changed in each version is in [CHANGELOG.md](../CHANGELOG.md). A version number is published
when a batch of work is consistent and stable; small fixes are grouped, and a fix for a broken
model or a wrong value ships without waiting.

## What comes next

Nothing below carries a date.

- **Speed.** Kernel fusion and graph-level simplification (operations that cancel out are never computed), each one
  measured against a reproducible benchmark and gated on the model's output staying correct.

## Outside this roadmap

[NeuroBrix Studio](https://github.com/NeuroBrix/neurobrix-studio), the desktop application, is
developed in its own repository. It is a client of the engine and never a second way to run models:
the command line and the serving daemon stay the engine's only entry points.
