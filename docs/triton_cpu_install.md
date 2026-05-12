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

## Install — build from source (only path today)

As of 2026-05, **upstream does not publish a PyPI wheel** for
`triton-cpu`. The project README states "It's still work in progress"
and install is build-from-source only. NeuroBrix gates the
`--triton` CPU path on a successful build of this upstream project.

```bash
git clone https://github.com/triton-lang/triton-cpu.git
cd triton-cpu
pip install -r python/requirements.txt
pip install -e .
```

Prerequisites (typical Triton build chain — install via your distro's
package manager before `pip install`):

- LLVM (pinned by upstream; the build pulls and configures it).
- CMake ≥ 3.18.
- Ninja.
- A C++17-capable host compiler (gcc ≥ 9 or clang ≥ 10).

Expected build duration: ~30 minutes on a modern workstation.
Expected peak disk: ~15 GB (LLVM build tree).

Verify the install:

```bash
TRITON_CPU_BACKEND=1 python3 -c "
import triton.runtime.driver
print('active:', triton.runtime.driver.active.__class__.__name__)
print('target:', triton.runtime.driver.active.get_current_target())
"
```

A working install prints a CPU target (architecture string like
`cpu` or `x86_64`) and does not raise.

## Why no auto-install

NeuroBrix never auto-fetches or auto-builds optional backends. See
`src/neurobrix/triton/cpu_backend.py` docstring for the full
rationale (security, trust, air-gapped deployments, alignment with
mature ML runtimes). When you invoke `--triton` on a CPU-only host
without `triton-cpu` built, NeuroBrix raises
`TritonCPUNotInstalledError` with the install pointer.

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
