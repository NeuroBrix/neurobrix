"""
NeuroBrix Debug Control

Controls verbose logging throughout the runtime.
When disabled, eliminates expensive tensor.item() calls that cause GPU sync.

Usage:
    # Check if debug is enabled
    from neurobrix.core.runtime.debug import DEBUG

    if DEBUG:
        print(f"[Debug] tensor mean: {tensor.mean().item()}")

    # Or use conditional print
    from neurobrix.core.runtime.debug import debug_print
    debug_print(f"[Debug] tensor mean: {tensor.mean().item()}")  # Only evaluates if DEBUG=True

Environment variable:
    NBX_DEBUG=1  - Enable verbose debug output (slower, forces GPU sync)
    NBX_DEBUG=0  - Disable debug output (default, fast)

CLI flag:
    --debug      - Enable verbose debug output
"""
import os

# Check environment variable (default: False for performance)
_debug_env = os.environ.get("NBX_DEBUG", "0")
DEBUG: bool = _debug_env.lower() in ("1", "true", "yes", "on")


def debug_print(*args, **kwargs) -> None:
    """Print only if debug mode is enabled."""
    if DEBUG:
        print(*args, **kwargs)


