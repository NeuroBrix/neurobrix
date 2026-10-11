# NeuroBrix

NeuroBrix is an open-source inference engine that runs language models, vision-language and
omni models, speech recognition, speech synthesis, image and video generation, and image
upscaling from one container format (`.nbx`) and one command-line tool. It runs published
models; it does not train them.

```bash
pip install neurobrix
```

Install PyTorch for your hardware first; see [Installation](getting-started/installation.md).

```bash
# Download a model from the hub
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Run it (hardware is detected automatically)
neurobrix run --model TinyLlama-1.1B-Chat-v1.0 --prompt "Explain recursion."
```

## Key features

| Feature | Description |
|---------|-------------|
| **One runtime** | The same engine runs language, speech, image, video and multimodal models, with no per-model pipeline |
| **NBX format** | Self-contained archive with graph, weights, topology and runtime defaults |
| **Automatic placement** | Plans each component onto one GPU, several GPUs, or host memory from the detected hardware |
| **Two compute branches** | PyTorch (default) and NeuroBrix's own Triton kernels, over the same container |
| **Serve mode** | A daemon keeps weights loaded between requests |

## Status

The current release is **0.5.4**, for **NVIDIA GPUs (CUDA) and CPU**. Apple Silicon is being
integrated on `main` and is not in a release yet. The hub lists the downloadable models; each
release's notes on [GitHub](https://github.com/NeuroBrix/neurobrix/releases) list what is known
not to work.

All models are the work of their respective authors and stay under their own licenses. See the
[Model Hub](getting-started/hub.md).

## Quick links

- [Getting Started](getting-started/index.md): install and run a first model
- [CLI Reference](reference/cli.md): command documentation
- [Architecture](architecture/index.md): how the runtime works
- [NBX Format](architecture/nbx-format.md): the model container
