# Installation

NeuroBrix needs Python 3.10 or newer. The current release on PyPI is **0.6.16**. This page
describes `main`; where `main` differs from 0.6.16, it says so.

NeuroBrix has two engines:

- The **Triton engine** (`--triton`) runs on [nbxtensor](https://github.com/NeuroBrix/nbxtensor)
  and Triton. It needs no PyTorch.
- The **PyTorch engine** (`--compiled`, `--sequential`) needs PyTorch, installed from the index
  that matches your hardware.

With no mode flag, a run takes the PyTorch engine where PyTorch is installed and the Triton
engine where it is not. Its launch line says which.

## NVIDIA GPU (Linux), Triton engine

Install the NVIDIA driver, then:

```bash
pip install neurobrix
neurobrix doctor
```

Without PyTorch the run takes the Triton engine (`engine: triton (torch not installed)`). The
CUDA runtime library comes with nbxtensor; nothing else is needed.

## NVIDIA GPU (Linux), PyTorch engine

Install the PyTorch CUDA build your driver supports (see the selector on
[pytorch.org](https://pytorch.org/get-started/locally/)), then NeuroBrix with its `pytorch`
extra. `pip install torch` on its own pulls the default PyTorch wheel from PyPI, which may be
built for a CUDA version your driver or GPU cannot use. PyTorch then reports no GPU and the
engine runs on CPU.

```bash
pip install typing-extensions      # from PyPI first: the PyTorch index carries only its source
pip install torch --index-url https://download.pytorch.org/whl/<cuda-build>   # e.g. cu126
pip install "neurobrix[pytorch]"
neurobrix doctor
```

On `main`, the PyTorch engine on an NVIDIA or AMD GPU needs torch 2.12.0 or newer and refuses
an older one by name before any weight loads. 0.6.16 accepts torch 2.1.0 or newer.

### NVIDIA V100 (Volta)

Current CUDA 13 builds of PyTorch no longer support Volta. Use a `cu126` build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Apple Silicon (macOS)

The Triton engine runs on the Apple GPU through Metal. PyPI publishes no Triton for macOS:
`install-metal` installs NeuroBrix's Triton build for Apple GPUs
([NeuroBrix/triton-ext](https://github.com/NeuroBrix/triton-ext)), each wheel checked against
the sha256 written in nbxtensor. Wheels exist for macOS 14 or newer, CPython 3.10 to 3.13.

```bash
pip install neurobrix
python -m nbxtensor install-metal
neurobrix doctor
```

Xcode's Metal toolchain is the other requirement.
The PyTorch engine also runs on the Apple GPU, through MPS: `pip install torch "neurobrix[pytorch]"`.

## Windows

Windows: support is on main and comes with the next release. On main it runs the PyTorch engine (`--compiled`,
`--sequential`); PyPI publishes no triton for Windows, so `--triton` is refused by name.

## CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "neurobrix[pytorch]"
```

To run on the CPU of a machine that has an NVIDIA GPU, set `CUDA_VISIBLE_DEVICES=-1` in the
shell that starts `neurobrix run` or `neurobrix serve`. On Windows this is read on `main`, not
by 0.6.16.

## What gets installed

`pip install neurobrix` installs nbxtensor, Triton (on `main`: on Linux x86_64 and aarch64;
0.6.16 asks for it on every platform but macOS), numpy,
safetensors, PyYAML, pydantic, packaging, Jinja2, Pillow, soundfile, imageio-ffmpeg, requests
and tqdm, and the grammar engine `nbxgrammar` where a wheel of it exists (Linux x86_64 and
aarch64, macOS arm64 and x86_64, Windows x86_64). PyTorch is not among them: it is the
`pytorch` extra. Tokenizers and audio front-ends are read from the model's container, so no
vendor model library is needed to run a model.

## Check the installation

```bash
neurobrix doctor
```

`doctor` checks that the `neurobrix` command is on your PATH, whether the Triton engine can run
(nbxtensor, Triton and a GPU runtime), which PyTorch build is installed if any, and whether it
can see the GPUs your driver reports. It exits non-zero when neither engine can run.

## Troubleshooting

### "Torch not compiled with CUDA enabled", or `doctor` reports no GPU for PyTorch

The installed PyTorch is a CPU build, or a CUDA build your driver or GPU cannot serve. Reinstall
it from the matching index:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/<cuda-build>
```

## Development install

```bash
git clone https://github.com/NeuroBrix/neurobrix.git   # or https://gitlab.com/neurobrix/Neurobrix.git
cd neurobrix
pip install -e ".[dev]"
neurobrix --help
```

Without installing, `PYTHONPATH=src python -m neurobrix --help` also works from the checkout,
once its dependencies are installed.
