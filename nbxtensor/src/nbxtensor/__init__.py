"""nbxtensor — an inference-first tensor library, torch-free.

The tensor library of NeuroBrix, built to stand on its own: dtypes, shapes,
strides and views, memory that is planned rather than discovered, device
backends (CUDA, Apple Metal, CPU) and Triton kernels that declare their own
cost. NeuroBrix's Triton engine is its first client; it imports this package
and never re-implements what lives here.

What is in the library today, area by area, is the README's status table;
the versions are numbered as `docs/nbxtensor/DESIGN.md` section 7 says (the
first publication is 0.0.1, each fix after it increments, the Rust rewrite is
2.0.0).
"""

#: the one version number; `pyproject.toml` states the same (tests/test_version.py)
__version__ = "0.0.1"
__all__: list = []
