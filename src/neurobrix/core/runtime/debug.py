"""
NeuroBrix Debug Control

Controls the expensive debug statistics throughout the runtime (tensor.item() calls that cause a GPU sync).

Usage:
    from neurobrix.core.runtime.debug import DEBUG

    if DEBUG:
        _log.debug(f"[Debug] tensor mean: {tensor.mean().item()}")

Environment variable:
    NBX_DEBUG=1  - compute the debug statistics (slower, forces GPU sync); shown with -v / NBX_LOG=debug
    NBX_DEBUG=0  - Disable debug output (default, fast)
"""
import os

# Check environment variable (default: False for performance)
_debug_env = os.environ.get("NBX_DEBUG", "0")
DEBUG: bool = _debug_env.lower() in ("1", "true", "yes", "on")
