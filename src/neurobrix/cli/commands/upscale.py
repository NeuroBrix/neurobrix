"""`nbx upscale` — image super-resolution subcommand.

Loads an input image, runs it through an upscaler container
(family=upscaler) on the chosen execution mode, and writes the
high-resolution output image.

DATA-DRIVEN (R34): no per-model branching. The graph input
variable is read from `topology.connections` (the unique
`global.*` source). Image preprocessing (rescale factor, pad
alignment) is read from the container's embedded
`modules/processor/preprocessor_config.json`. Output
denormalisation is handled by the shared family-aware
`output_dispatch.save_image`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _resolve_execution_mode(args) -> str:
    """Map the --mode flag to a runtime execution-mode string."""
    mode = getattr(args, "mode", None) or "compiled"
    valid = {"compiled", "sequential", "triton", "triton-sequential"}
    if mode not in valid:
        print(
            f"ERROR: --mode must be one of {sorted(valid)}, got '{mode}'.")
        sys.exit(1)
    # Runtime uses an underscore variant for the sequential triton mode.
    return "triton_sequential" if mode == "triton-sequential" else mode


# Input-variable discovery and image preprocessing moved to the shared
# CLI/daemon brick (core/module/vision/input_processor.py) so the
# serving warm path takes the IDENTICAL preparation — the warm path
# previously fell through to the video contract and the family broke
# on serve warm, both engines (S3_FINDING_SERVE_UPSCALER_WARM,
# 2026-08-26). Thin aliases keep this module's call sites readable.
def _find_input_variable(topology: dict) -> str:
    from neurobrix.core.module.vision.input_processor import (
        find_upscale_input_variable,
    )
    return find_upscale_input_variable(topology)


def _drain_device(execution_mode: str) -> None:
    """Each engine drains its own device around the timed run — the ATen
    branch through torch, the Triton branch through the allocator's
    runtime (R33: a --triton process never loads torch)."""
    if execution_mode in ("compiled", "sequential"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    else:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        if DeviceAllocator.device_count() > 0:
            DeviceAllocator.device_synchronize()


def _load_and_preprocess_image(image_path: str, cache_path: Path):
    from neurobrix.core.module.vision.input_processor import (
        load_upscale_image,
    )
    return load_upscale_image(image_path, cache_path)


def cmd_upscale(args):
    """Image super-resolution via an upscaler container."""
    from neurobrix.nbx import NBXContainer
    from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.executor import RuntimeExecutor
    from neurobrix.core.runtime.output_dispatch import save_image
    from neurobrix.cli.utils import find_model

    if not args.model or not args.input or not args.output:
        print("ERROR: --model, --input and --output are required.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input image not found: {input_path}")
        sys.exit(1)

    execution_mode = _resolve_execution_mode(args)

    print("NeuroBrix Upscale")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Mode: {execution_mode}")
    print("=" * 60)

    nbx_path = find_model(args.model)

    print("\n[1/4] Loading container...")
    container = NBXContainer.load(str(nbx_path))
    manifest = container.get_manifest() or {}
    family = manifest.get("family")
    if family != "upscaler":
        print(
            f"ERROR: model '{args.model}' is family='{family}', not "
            f"'upscaler'. Use `nbx run` for non-upscaler models.")
        sys.exit(1)

    cache_path = container._cache_path
    assert cache_path is not None, "Container cache path must be set"
    with open(cache_path / "topology.json") as f:
        topology = json.load(f)
    input_var = _find_input_variable(topology)

    print("\n[2/4] Solving hardware allocation...")
    if args.hardware:
        hw_profile = load_profile(args.hardware)
    else:
        hw_profile = load_profile(get_or_create_default_profile())
    print(f"   Profile: {hw_profile.id} ({hw_profile.total_vram_gb:.1f} GB)")

    print("\n[3/4] Preprocessing input image...")
    pixel_values, (orig_h, orig_w) = _load_and_preprocess_image(
        args.input, cache_path)
    _, _, in_h, in_w = pixel_values.shape
    print(f"   Input tensor: {tuple(pixel_values.shape)} "
          f"(original {orig_h}×{orig_w}, padded {in_h}×{in_w})")

    input_config = InputConfig(
        batch_size=1, height=in_h, width=in_w, dtype="float32",
    )
    solver = PrismSolver()
    execution_plan = solver.solve_smart(container, hw_profile, input_config)
    print(f"   Strategy: {execution_plan.strategy}")
    # The choice, said out loud. Prism scores every viable strategy and takes
    # the fastest — invisible unless printed, and an engine that decides
    # without saying so is indistinguishable from one that decides badly.
    if getattr(execution_plan, "selection_reason", ""):
        print(f"   Why:      {execution_plan.selection_reason}")
    _cards = sorted({
        d for a in execution_plan.components.values()
        for d in (getattr(a, "devices", None) or [])
    })
    if _cards:
        print(f"   Devices:  {', '.join(_cards)}"
              f"  ({execution_plan.total_memory_mb:.0f} MB planned)")

    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))

    from neurobrix.kernels.wrappers import set_hardware_profile
    set_hardware_profile(hw_profile)
    executor = RuntimeExecutor(pkg, execution_plan, mode=execution_mode)

    print(f"\n[4/4] Running upscale ({execution_mode})...")
    inputs = {input_var: pixel_values}
    try:
        _drain_device(execution_mode)
        t0 = time.time()
        outputs = executor.execute(inputs)
        _drain_device(execution_mode)
        wall = time.time() - t0
    except Exception as e:
        print(f"\n[ERROR] Upscale failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[Timing] Total execution: {wall:.2f}s")

    # The contract is input x scale exactly. The scale lookup and the
    # crop live in save_image — ONE point of truth for both entry
    # points (D-UPSCALE-SERVING-CROP); this caller only passes the
    # original size it owns.
    saved = save_image(outputs, args.output, family, executor, pkg,
                       orig_hw=(orig_h, orig_w))
    print("\n" + "=" * 60)
    print(f"SAVED: {saved}")
    print("=" * 60)
    return 0
