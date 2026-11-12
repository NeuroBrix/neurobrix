# Contributing to NeuroBrix

Thank you for your interest in contributing to NeuroBrix — the universal AI runtime engine.

Developed by Hocine Benkelaya · Supported and funded by [Neural Network Holding Ltd](https://neuralnetworkholding.com) · Commercial distribution by [WizWorks OÜ](https://wizworks.io). It is released under the [Apache 2.0 License](LICENSE).

---

## Ways to Contribute

### Report Bugs

Found something broken? Open a Bug Report on [GitHub](https://github.com/NeuroBrix/neurobrix/issues/new/choose) | [GitLab](https://gitlab.com/neurobrix/neurobrix/-/issues/new) with:

- NeuroBrix version (`neurobrix --version`) and the output of `neurobrix doctor`
- Hardware (GPU model and count, or CPU only) and the execution mode flag used
- Model name and family
- Steps to reproduce
- Full error traceback

### Request Features

Have an idea? Open a Feature Request on [GitHub](https://github.com/NeuroBrix/neurobrix/issues/new/choose) | [GitLab](https://gitlab.com/neurobrix/neurobrix/-/issues/new) describing:

- The problem you're solving
- Your proposed solution
- Alternative approaches you considered

### Request Model Support

Want a specific model in the NeuroBrix Hub? Open a Model Request on [GitHub](https://github.com/NeuroBrix/neurobrix/issues/new/choose) | [GitLab](https://gitlab.com/neurobrix/neurobrix/-/issues/new) with:

- Model name and source (Hugging Face link, paper, etc.)
- Model family (LLM, code, vision-language, multimodal, speech-to-text, speech understanding, text-to-speech, image, video, upscaler)
- Why this model matters to you

### Contribute Code

We welcome contributions across the entire stack:

| Area | Description | Skill Level |
|------|-------------|-------------|
| **Kernels** | Triton GPU kernels for attention, normalization, activations | Advanced |
| **Hardware Profiles** | YAML profiles for new GPU configurations | Beginner |
| **Core Runtime** | Executor, hardware placement, compiled sequences | Advanced |
| **CLI** | New commands, UX improvements, output formatting | Intermediate |
| **Documentation** | Guides, API docs, tutorials, translations | Beginner |
| **Testing** | Test coverage, regression tests, benchmarks | Intermediate |
| **NBX Format** | Container format tooling, validation, inspection | Advanced |
| **Serving** | Daemon improvements, session management, API | Intermediate |

---

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.1+ (a CUDA build for GPU work; see the [installation guide](docs/getting-started/installation.md))
- An NVIDIA GPU for kernel development; the unit tests run without one
- Git

### Clone and Install

```bash
# From GitHub
git clone https://github.com/NeuroBrix/neurobrix.git
# or from GitLab
git clone https://gitlab.com/neurobrix/neurobrix.git

cd neurobrix
pip install -e ".[dev]"
```

### Project Structure

```
src/neurobrix/
  cli/           Command-line interface
  core/          Runtime engine (executor, placement, dtype, flows, strategies)
  triton/        Triton compute branch (sequence, flows, weight loading)
  kernels/       Triton GPU kernels and the NBXTensor type
  nbx/           .nbx container format
  serving/       Model serving daemon
  agent/         Tool-calling agent loop on the daemon
  config/        Hardware profiles, vendor and family configs
tests/
  unit/          Unit tests (no GPU needed)
  regression/    Regression tests (need installed models and a GPU)
```

---

## Code Standards

NeuroBrix follows strict engineering principles. Please read these before writing code.

### The ZERO Principles

Every contribution must respect the three ZERO principles:

| Principle | Rule |
|-----------|------|
| **ZERO HARDCODE** | All values must come from the `.nbx` container or hardware profile. Never hardcode model-specific values. |
| **ZERO FALLBACK** | If data is missing, the system must crash with a clear error. Never provide silent defaults. |
| **ZERO SEMANTIC** | The runtime must never contain domain knowledge. It does not know what "an image" or "a token" is. |

### Tensor work lives in nbxtensor

**nbxtensor is the ecosystem's torch** (owner rule, 2026-11-12). Everything that touches tensors or follows torch's
logic (tensors, devices, memory, arenas, copies, compute kernels, replay, the tensor side of the experts store) lives in
nbxtensor, never in the engine. nbxtensor is a library with its own role: it replaces torch, and the engine is one of
its users, like any other. The engine keeps policy: Prism, model families, flows, the CLI, serving, which expert sits in
which slot and how many slots Prism grants. Every project of the ecosystem is separate (the engine, nbxtensor,
NBXGrammar, NBX Code, the container builder, the hub), and a function is filed in the project it belongs to. Engine code
that does tensor work is a defect as severe as a wrong answer: its fix ships at once, never held back to be grouped into
a release. The PyTorch engine computes with torch, its own library; that is not a breach.

For a contributor: a tensor helper is never added to the engine "for now"; the library is improved instead. A tensor
operation, a copy, a memory fact or a kernel the engine needs and nbxtensor lacks is added to nbxtensor, with its test,
and the engine calls it.

The brick ratchet enforces it: `tests/unit/brick_ratchet/test_the_brick_ratchet_refuses_what_gets_worse.py`
(`tools/brick_ratchet.py`, its patterns in `tools/brick_ratchet.json` under `tensor_logic`) counts the tensor mechanisms
each engine file holds. A file's count only goes down, a file with none gains none, and the engine's core imports none
of the library's tensor mechanics directly. `python tools/brick_ratchet.py --tensor-hits` lists them.

### Code Hygiene

- **Delete unused code immediately.** No dead functions, no unused imports, no orphan files.
- **No commented-out code.** Use git history instead.
- **No `TODO: Remove` comments.** Remove it now.
- **Fix problems when detected.** Don't defer.
- **Every function must have a docstring** explaining what it does, its parameters, and return values.

### Style

- Follow PEP 8 with 120 character line width
- Use type hints on all function signatures
- Prefer explicit over implicit
- Name variables and functions descriptively

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:

| Prefix | Use |
|--------|-----|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `kernel/` | GPU kernel work |
| `profile/` | Hardware profiles |
| `docs/` | Documentation |

### 2. Make Your Changes

- Keep commits focused and atomic
- Write clear commit messages: `fix(prism): handle zero-VRAM edge case in solver`
- Ensure all ZERO principles are respected

### 3. Test Your Changes

```bash
# Unit tests (no GPU needed)
PYTHONPATH=src python -m pytest tests/unit

# Validate a container still passes the integrity checks
neurobrix validate <model.nbx> --level deep --strict
```

### 4. Submit the PR

- Fill out the pull request template completely
- Certificates for a GPU class (the README's "Certify your GPU and share it") use the `certificate`
  template: open the pull request with `?template=certificate.md` in its URL
- Reference any related issues
- Describe what changed and why
- Include test results or benchmarks if applicable

### 5. Code Review

All PRs require review from a maintainer. We look for:

- Adherence to ZERO principles
- No hardcoded values or silent defaults
- Clean, documented code
- Adequate test coverage
- No regressions in existing functionality

---

## Hardware Profile Contributions

Adding a hardware profile is one of the easiest ways to contribute. Create a YAML file:

```yaml
# src/neurobrix/config/hardware/rtx4090-24g.yml
id: rtx4090-24g
vendor: nvidia
preferred_dtype: bfloat16
summary:
  total_gpus: 1
  total_vram_gb: 24
  topology: Single-GPU
devices:
- index: 0
  brand: nvidia
  model: NVIDIA GeForce RTX 4090
  memory_mb: 24576
  compute_capability: '8.9'
  supports_dtypes: [float32, float16, bfloat16]
  architecture: ada
interconnect:
  groups: []
```

Use an existing file in `src/neurobrix/config/hardware/` as the template, and see
[Writing Hardware Profiles](docs/guide/hardware-profiles.md). Submit it as a PR.

---

## Reporting Security Issues

**Do not open public issues for security vulnerabilities.** Please read our [Security Policy](SECURITY.md) for responsible disclosure instructions.

---

## Community

- **Issues:** [GitHub](https://github.com/NeuroBrix/neurobrix/issues) | [GitLab](https://gitlab.com/neurobrix/neurobrix/-/issues)
- **Email:** [contributors@neurobrix.org](mailto:contributors@neurobrix.org)

---

## Contributor Assignment Agreement

Before your first pull request is merged, you sign the [NeuroBrix Contributor Assignment
Agreement](https://github.com/NeuroBrix/neurobrix/blob/main/docs/legal/contributor-assignment-agreement.md) — once, for
every NeuroBrix project (neurobrix, nbxtensor, NBXGrammar, NBX Code, the hub). You assign the rights in your contribution
to the owner of the NeuroBrix project (named in the NOTICE file) and its successors, and receive back a license to use
your own contribution as you wish; the owner may license the projects under other terms; no trademark right is involved.
To sign, post on your pull request the comment `I have read the NeuroBrix Contributor Assignment Agreement and I hereby
sign it.` — the `NeuroBrix CLA` check then passes. Contributing for a company? Your employer signs the [entity
agreement](https://github.com/NeuroBrix/neurobrix/blob/main/docs/legal/entity-contributor-assignment-agreement.md) first:
write to contact@neurobrix.org.

## License

NeuroBrix is published under the [Apache License 2.0](LICENSE).
