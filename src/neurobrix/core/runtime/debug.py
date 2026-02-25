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


def set_debug(enabled: bool) -> None:
    """Programmatically set debug mode."""
    global DEBUG
    DEBUG = enabled


def debug_print(*args, **kwargs) -> None:
    """Print only if debug mode is enabled."""
    if DEBUG:
        print(*args, **kwargs)


def debug_log_tensor(name: str, tensor, log_stats: bool = True) -> None:
    """
    Log tensor info only if debug mode is enabled.

    IMPORTANT: This function avoids .item() calls when DEBUG=False,
    which prevents expensive GPU synchronization.

    Args:
        name: Tensor name for logging
        tensor: PyTorch tensor
        log_stats: Whether to log mean/std (requires GPU sync)
    """
    if not DEBUG:
        return

    import torch
    if not isinstance(tensor, torch.Tensor):
        print(f"[Debug] {name}: {type(tensor).__name__}")
        return

    shape_str = list(tensor.shape)
    dtype_str = str(tensor.dtype).replace("torch.", "")

    if log_stats and tensor.numel() > 0 and tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
        mean_val = tensor.float().mean().item()
        std_val = tensor.float().std().item()
        print(f"[Debug] {name}: shape={shape_str}, dtype={dtype_str}, mean={mean_val:.4f}, std={std_val:.4f}")
    else:
        print(f"[Debug] {name}: shape={shape_str}, dtype={dtype_str}")


def debug_log_timestep(step_idx: int, total_steps: int, timestep) -> None:
    """Log timestep info only if debug mode is enabled."""
    if not DEBUG:
        return

    import torch
    if isinstance(timestep, torch.Tensor):
        ts_val = timestep.item() if timestep.numel() == 1 else timestep[0].item()
    else:
        ts_val = float(timestep)

    print(f"[Loop] Step {step_idx + 1}/{total_steps} (t={ts_val:.1f})")
