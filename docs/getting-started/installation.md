# Installation

NeuroBrix needs Python 3.10 or newer and PyTorch 2.1 or newer. The current release, 0.5.4,
supports **NVIDIA GPUs (CUDA) and CPU**.

## Step 1: Install PyTorch for your hardware

Install PyTorch **before** NeuroBrix. `pip install neurobrix` on its own pulls the default
`torch` wheel from PyPI, which may be built for a CUDA version your driver or GPU cannot use.
PyTorch then reports no GPU and the engine runs on CPU.

### NVIDIA GPU (Linux)

Install the NVIDIA driver, then the PyTorch CUDA build your driver supports (see the selector
on [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/<cuda-build>   # e.g. cu126
```

### NVIDIA V100 (Volta)

Current CUDA 13 builds of PyTorch no longer support Volta. Use a `cu126` build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

`torch 2.10.0+cu126` and `torch 2.14.0+cu126` were checked on the project's V100 machines. The
default PyPI `torch` and the cu128, cu129, cu130 and cu132 builds were not usable there.

### CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Apple Silicon

Not in a release yet. Triton execution on Apple GPUs is being integrated on `main` and is planned
for 0.5.5.

## Step 2: Install NeuroBrix

```bash
pip install neurobrix
```

The runtime dependencies are PyTorch, numpy, safetensors, PyYAML, pydantic, packaging, Jinja2,
Pillow, soundfile, imageio-ffmpeg, requests and tqdm. Tokenizers and audio front-ends are read
from the model's container, so no vendor model library is needed to run a model.

## Step 3: Check the installation

```bash
neurobrix doctor
```

`doctor` checks that the `neurobrix` command is on your PATH, which PyTorch build is installed,
and whether PyTorch can see the GPUs your driver reports. It exits non-zero when it finds a
problem that will block a run.

## Troubleshooting

### "Torch not compiled with CUDA enabled", or `doctor` reports no GPU

The installed PyTorch is a CPU build, or a CUDA build your driver or GPU cannot serve. Reinstall
it from the matching index (Step 1):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/<cuda-build>
```

## Development install

```bash
git clone https://github.com/NeuroBrix/neurobrix.git   # or https://gitlab.com/neurobrix/neurobrix.git
cd neurobrix
pip install -e ".[dev]"
neurobrix --help
```

Without installing, `PYTHONPATH=src python -m neurobrix --help` also works from the checkout.
