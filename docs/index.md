# NeuroBrix

NeuroBrix is an open-source inference engine that runs language models, vision-language and
omni models, speech recognition, speech synthesis, image and video generation, and image
upscaling from one container format (`.nbx`) and one command-line tool. It runs published
models; it does not train them.

```bash
pip install neurobrix
```

On Linux with an NVIDIA GPU this is all you need: without PyTorch, a run takes the Triton
engine. For the PyTorch engine, the Apple GPU or a CPU-only machine, see
[Installation](getting-started/installation.md).

```bash
# Download a model from the hub
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0 --accept-license

# Run it (no --hardware needed: the machine's profile is used)
neurobrix run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Explain recursion."
```

## Key features

| Feature | Description |
|---------|-------------|
| **One runtime** | The same engine runs language, speech, image, video and multimodal models, with no per-model pipeline |
| **NBX format** | Self-contained archive with graph, weights, topology and runtime defaults |
| **Automatic placement** | Plans each component onto one GPU, several GPUs, or host memory from the machine's hardware profile |
| **Two compute branches** | PyTorch, and nbxtensor (NeuroBrix's own Triton kernels), over the same container. With no flag, a run takes PyTorch where it is installed and the Triton engine where it is not |
| **Serve mode** | A daemon keeps weights loaded between requests, with an optional OpenAI-compatible HTTP API |

## Status

The current release on PyPI is **0.6.16**, for NVIDIA GPUs on Linux, Apple Silicon (Metal) and CPU.
These pages describe `main`, which is ahead of that release; where `main` differs, the page says so.
The hub lists the downloadable models; each release's notes on
[GitHub](https://github.com/NeuroBrix/neurobrix/releases) say more about each platform.

All models are the work of their respective authors and stay under their own licenses. See the
[Model Hub](getting-started/hub.md).

## Quick links

- [Getting Started](getting-started/index.md): install and run a first model
- [CLI Reference](reference/cli.md): command documentation
- [Architecture](architecture/index.md): how the runtime works
- [NBX Format](architecture/nbx-format.md): the model container
