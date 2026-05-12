# Installing `triton-cpu` for NeuroBrix CPU-pure triton inference

NeuroBrix's `--triton` and `--triton-sequential` modes use NeuroBrix's
custom `@triton.jit` kernels (RMS norm, SDPA, conv2d, fused
upsample+conv, etc.). On GPU these compile and run via the standard
Triton CUDA backend that ships with PyTorch. On a **CPU-only host**
they require the upstream `triton-cpu` package, which adds an LLVM
backend to Triton that compiles the same `@triton.jit` source to
x86-64 / arm64 host code.

NeuroBrix **never auto-installs** this package. You must install it
explicitly. Rationale: security, trust, air-gapped deployments,
alignment with mature ML runtimes (llama.cpp, vLLM, transformers,
torch itself do not auto-install optional backends). See
`src/neurobrix/triton/cpu_backend.py` docstring for the full rationale.

## Quick install (recommended)

```bash
pip install triton-cpu
```

This pulls the wheel from PyPI matching your Python version and OS.

## Supported platforms (verified by NeuroBrix CI as of 2026-05)

| Platform                | Wheel available | Status                             |
|-------------------------|-----------------|------------------------------------|
| Linux x86_64 (glibc 2.27+) | yes           | primary target — CI green          |
| Linux aarch64           | yes             | secondary — basic ops validated    |
| macOS arm64 (Apple Silicon) | yes         | secondary — basic ops validated    |
| macOS x86_64            | no              | build-from-source only             |
| Windows                 | no              | not supported by upstream          |

If your distro is older (e.g. CentOS 7, RHEL 7, Ubuntu 18.04) the
PyPI wheel may reject the install with a glibc-version error. In that
case, build from source (next section).

## Build from source

`triton-cpu` is part of the `triton-lang/triton-cpu` repository. The
build pulls a pinned LLVM commit (~30 minutes on a modern workstation,
~15 GB peak disk).

```bash
git clone https://github.com/triton-lang/triton-cpu.git
cd triton-cpu
pip install ninja cmake
pip install -e python
```

Verify the install:

```bash
python -c "import triton_cpu; print(triton_cpu.__version__)"
```

If you see an `ImportError` referencing `libtinfo.so.6` on older
distros, install your distro's `ncurses` package (`apt install
libncurses6` on Debian/Ubuntu).

## Air-gapped install

Download the wheel on a connected machine:

```bash
pip download triton-cpu -d ./triton-cpu-wheel --no-deps
```

Transfer the directory to the air-gapped host and:

```bash
pip install --no-index --find-links ./triton-cpu-wheel triton-cpu
```

## Verifying with NeuroBrix

After install, run:

```bash
CUDA_VISIBLE_DEVICES="" neurobrix run \
    --model TinyLlama-1.1B-Chat-v1.0 \
    --prompt "Hello" --max-tokens 5 \
    --triton --hardware cpu-only-x86
```

You should see `Engine: TRITON` in the output and 5 coherent tokens.
If you see `TritonCPUNotInstalledError`, the install did not take
effect in the active Python environment — check `which python`
matches the venv where you ran `pip install triton-cpu`.

## Known limitations on triton-cpu today

See `src/neurobrix/triton/triton_cpu_coverage_gaps.md` for the
upstream open issues that block specific NeuroBrix paths (notably the
fp16 Dot3D accuracy gap, triton-cpu issue #147, which means fp16
Sana 4Kpx inference on CPU pure is not yet supported via this
backend — use `--compiled` instead).

## Alternative: `--compiled` on CPU

If you do not want to install `triton-cpu`, NeuroBrix's `--compiled`
mode already runs on CPU via the mature PyTorch CPU backend. It has
no extra install requirement, is bit-similar to GPU inference for
fp32 / bf16 paths, and has no equivalent fp16 upstream gaps. This is
the recommended production path for CPU-only inference today.
