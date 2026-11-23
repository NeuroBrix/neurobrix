"""THE TORCH THE PyTorch BRANCH IS VALIDATED WITH: one version, written once, here (owner's order 23/11/2026, row 275).

The engine is frozen on that torch: the PyTorch branch (`--compiled`, `--sequential`) runs on torch 2.14.0 and on no
other. pyproject.toml's `pytorch` extra pins the same release (`torch==2.14.0`; a CUDA build such as 2.14.0+cu126
satisfies it, its local label being ignored by a bare `==`), and a test reads the extra against this constant, so the
two cannot drift. The Triton branch never imports torch and never reaches this file.
"""
from __future__ import annotations

from typing import Any

#: the release (no local label) of the torch the PyTorch branch is validated with
VALIDATED_TORCH = "2.14.0"


def release_of(version: str) -> str:
    """`2.14.0+cu126` -> `2.14.0`: the release part, local label dropped."""
    return str(version).split("+", 1)[0]


def require_validated_torch(torch: Any = None) -> None:
    """Refuse, by name, a torch whose release is not VALIDATED_TORCH; nothing is skipped or guessed instead."""
    if torch is None:
        import torch
    have = release_of(torch.__version__)
    if have != VALIDATED_TORCH:
        raise RuntimeError(
            f"ZERO FALLBACK: this engine is validated with torch {VALIDATED_TORCH}, no other; this environment has "
            f"torch {torch.__version__}: install the validated version (pip install 'neurobrix[pytorch]', which asks "
            f"for torch=={VALIDATED_TORCH}; for a CUDA build install torch=={VALIDATED_TORCH} from the PyTorch index "
            f"first: pip install torch=={VALIDATED_TORCH} --index-url https://download.pytorch.org/whl/<cuda-build>)")
