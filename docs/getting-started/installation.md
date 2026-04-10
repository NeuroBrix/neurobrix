# Installation

## Step 1: Install PyTorch with CUDA

NeuroBrix requires **PyTorch with CUDA support**. The default `torch` package on PyPI is **CPU-only** and will not work with GPU inference.

Install the CUDA-enabled version from PyTorch's own package index:

```bash
# For CUDA 12.4 (RTX 30xx, 40xx, A100, H100)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs like V100)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

> **Why is this needed?** PyTorch CUDA wheels are ~2.5 GB and platform-specific, so they are hosted on PyTorch's own servers, not on PyPI. If you skip this step, pip will pull the CPU-only `torch` and you will get: `AssertionError: Torch not compiled with CUDA enabled`.

Verify CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

## Step 2: Install NeuroBrix

```bash
pip install neurobrix
```

With Triton kernel acceleration (Linux only):

```bash
pip install neurobrix[triton]
```

With all optional dependencies:

```bash
pip install neurobrix[full]
```

## Requirements

| Dependency | Minimum Version | Purpose | Required |
|-----------|----------------|---------|----------|
| Python | 3.10 | Runtime | Yes |
| PyTorch | 2.1 | Tensor operations, CUDA | Yes |
| safetensors | 0.4.0 | Weight loading | Yes |
| numpy | 1.24.0 | Array operations | Yes |
| pyyaml | 6.0 | Configuration | Yes |
| sentencepiece | 0.1.99 | SentencePiece tokenizers | Yes |
| tokenizers | 0.14.0 | HuggingFace fast tokenizers | Yes |
| jinja2 | 3.0.0 | Chat template rendering | Yes |
| soundfile | 0.12.0 | Audio file I/O | Yes |
| Pillow | 10.0.0 | Image file I/O | Yes |
| requests | 2.28.0 | HTTP client (hub downloads) | Yes |
| tqdm | 4.65.0 | Progress bars | Yes |
| triton | 2.1.0 | Custom GPU kernels | Optional (Linux only) |
| librosa | 0.10.0 | Audio resampling | Optional (`[audio]`) |
| mistral-common | 1.0.0 | Tekken tokenizer | Optional (`[mistral]`) |
| tiktoken | 0.5.0 | Tiktoken tokenizer | Optional (`[tiktoken]`) |
| transformers | 4.30.0 | Feature extractors | Optional (`[full]`) |

## Platform Support

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| **Linux** | CUDA, Triton kernels | Full support, recommended for production |
| **Windows** | CUDA | Fully supported since v0.1.0a9. Triton not available |
| **macOS** | Apple Silicon (MPS) | M1–M5 supported since v0.1.1. bf16 on M2+ |

## Verify Installation

```bash
neurobrix --help
```

## Troubleshooting

### "Torch not compiled with CUDA enabled"

You installed the CPU-only PyTorch from PyPI. Fix:

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### "No module named 'triton'"

Triton is only available on Linux. On Windows/macOS, NeuroBrix automatically skips Triton and uses the default compiled execution mode. If you see this error, you may be on an older version — upgrade:

```bash
pip install --upgrade --pre neurobrix
```

### "No module named 'sentencepiece'" (or other missing modules)

Upgrade to v0.1.0a10+ which includes all dependencies:

```bash
pip install --upgrade --pre neurobrix
```

## Development Mode

For development without installing:

```bash
git clone https://gitlab.com/neurobrix/Neurobrix.git
cd neurobrix
PYTHONPATH=src python -m neurobrix --help
```
