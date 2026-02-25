"""
CLI shared constants and utilities.

All command modules import from here — single source of truth for paths.
"""

from pathlib import Path

from neurobrix import __version__

# Package root for accessing bundled config (hardware/, vendors/)
PACKAGE_ROOT = Path(__file__).parent.parent

# User home cache for registry-installed models
NEUROBRIX_HOME = Path.home() / ".neurobrix"
STORE_DIR = NEUROBRIX_HOME / "store"    # Downloaded .nbx files
CACHE_DIR = NEUROBRIX_HOME / "cache"    # Extracted models (runtime)

# Registry
REGISTRY_URL = "https://neurobrix.es"


def find_model(model_name: str) -> Path:
    """
    Find a model in ~/.neurobrix/cache/.

    DATA-DRIVEN: No --family required at runtime.
    Family is stored in manifest.json (set at import time).

    Search: ~/.neurobrix/cache/{model_name}/

    Args:
        model_name: Name of the model to find

    Returns:
        Path to the model.nbx file or extracted cache directory

    Raises:
        FileNotFoundError: If model not found
    """
    cache_model = CACHE_DIR / model_name
    if cache_model.exists() and (cache_model / "manifest.json").exists():
        nbx_in_cache = cache_model / "model.nbx"
        if nbx_in_cache.exists():
            return nbx_in_cache
        # Extracted directory (no .nbx, just manifest + components)
        return cache_model

    # Not found — provide helpful error
    available = []
    if CACHE_DIR.exists():
        for model_dir in sorted(CACHE_DIR.iterdir()):
            if model_dir.is_dir() and (model_dir / "manifest.json").exists():
                available.append(f"  - {model_dir.name}")

    error_msg = f"Model '{model_name}' not found.\n"
    if available:
        error_msg += "Installed models:\n" + "\n".join(available)
    else:
        error_msg += (
            "No models installed.\n"
            "  Install from registry: neurobrix import <org>/<model>"
        )

    raise FileNotFoundError(error_msg)


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes > 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes > 1024**2:
        return f"{size_bytes / (1024**2):.0f} MB"
    elif size_bytes > 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes} B"
