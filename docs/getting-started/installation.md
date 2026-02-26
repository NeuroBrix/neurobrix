# Installation

## From PyPI

```bash
pip install neurobrix
```

With Triton kernel support:

```bash
pip install neurobrix[cuda]
```

## Requirements

| Dependency | Minimum Version | Purpose |
|-----------|----------------|---------|
| Python | 3.10 | Runtime |
| PyTorch | 2.0 | Tensor operations, CUDA |
| safetensors | 0.4.0 | Weight loading |
| numpy | 1.24.0 | Array operations |
| pyyaml | 6.0 | Configuration |
| triton | 2.1.0 | Custom GPU kernels (optional) |

## Verify Installation

```bash
neurobrix --help
```

## Development Mode

For development without installing:

```bash
git clone https://github.com/NeuroBrix/neurobrix.git
cd neurobrix
PYTHONPATH=src python -m neurobrix --help
```
