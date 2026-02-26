# Contributing to NeuroBrix

Thank you for your interest in contributing to NeuroBrix — the universal AI runtime engine.

NeuroBrix is developed and maintained by **WizWorks OÜ**, a property of **Neural Networks Holding LTD**, and released under the [Apache 2.0 License](LICENSE).

---

## Ways to Contribute

### Report Bugs

Found something broken? Open a [Bug Report](https://github.com/NeuroBrix/neurobrix/issues/new?template=bug_report.yml) with:

- NeuroBrix version (`neurobrix info`)
- Hardware profile used
- Model name and family
- Steps to reproduce
- Full error traceback

### Request Features

Have an idea? Open a [Feature Request](https://github.com/NeuroBrix/neurobrix/issues/new?template=feature_request.yml) describing:

- The problem you're solving
- Your proposed solution
- Alternative approaches you considered

### Request Model Support

Want a specific model in the NeuroBrix Hub? Open a [Model Request](https://github.com/NeuroBrix/neurobrix/issues/new?template=model_request.yml) with:

- Model name and source (Hugging Face link, paper, etc.)
- Model family (image, LLM, audio, video)
- Why this model matters to you

### Contribute Code

We welcome contributions across the entire stack:

| Area | Description | Skill Level |
|------|-------------|-------------|
| **Kernels** | Triton GPU kernels for attention, normalization, activations | Advanced |
| **Hardware Profiles** | YAML profiles for new GPU configurations | Beginner |
| **Core Runtime** | Executor, Prism solver, compiled sequences | Advanced |
| **CLI** | New commands, UX improvements, output formatting | Intermediate |
| **Documentation** | Guides, API docs, tutorials, translations | Beginner |
| **Testing** | Test coverage, regression tests, benchmarks | Intermediate |
| **NBX Format** | Container format tooling, validation, inspection | Advanced |
| **Serving** | Daemon improvements, session management, API | Intermediate |

---

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (for kernel development)
- Git

### Clone and Install

```bash
git clone https://github.com/NeuroBrix/neurobrix.git
cd neurobrix
pip install -e ".[dev]"
```

### Project Structure

```
neurobrix/
  cli/           Command-line interface
  core/          Runtime engine (executor, prism, dtype, flows)
  kernels/       Triton GPU kernels
  nbx/           .nbx container format
  serving/       Model serving daemon
  config/        Hardware profiles, family configs
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
# Run the test suite
python -m pytest tests/

# For kernel changes, run with GPU
python -m pytest tests/kernels/ --gpu

# Validate a model still works
neurobrix validate <model.nbx> --level deep --strict
```

### 4. Submit the PR

- Fill out the pull request template completely
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
# config/hardware/your-gpu-profile.yaml
id: rtx4090-24g
name: "NVIDIA RTX 4090 24GB"
gpus:
  - vram_gb: 24
    compute_capability: "8.9"
    name: "RTX 4090"
interconnect: null
total_vram_gb: 24
gpu_count: 1
```

Place it in `config/hardware/` and submit a PR.

---

## Reporting Security Issues

**Do not open public issues for security vulnerabilities.** Please read our [Security Policy](SECURITY.md) for responsible disclosure instructions.

---

## Community

- **Issues:** [github.com/NeuroBrix/neurobrix/issues](https://github.com/NeuroBrix/neurobrix/issues)
- **Discussions:** [github.com/NeuroBrix/neurobrix/discussions](https://github.com/NeuroBrix/neurobrix/discussions)
- **Email:** [contributors@neurobrix.es](mailto:contributors@neurobrix.es)

---

## License

By contributing to NeuroBrix, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

All contributions remain the intellectual property of Neural Networks Holding LTD.
