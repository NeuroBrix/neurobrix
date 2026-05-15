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


def _find_input_variable(topology: dict) -> str:
    """Return the unique `global.*` connection source feeding the graph.

    R34: the input variable name is never hardcoded — it is whatever
    the container's topology declares (e.g. `global.pixel_values`).
    """
    sources = []
    for conn in topology.get("connections", []):
        src = conn.get("from", "")
        if src.startswith("global."):
            sources.append(src)
    uniq = sorted(set(sources))
    if not uniq:
        raise RuntimeError(
            "No `global.*` input connection found in topology. "
            "The container does not declare a user-facing image input."
        )
    if len(uniq) > 1:
        raise RuntimeError(
            f"Expected exactly one global input connection, found "
            f"{uniq}. Upscalers take a single image input."
        )
    return uniq[0]


def _load_and_preprocess_image(image_path: str, cache_path: Path):
    """PIL load → CHW float tensor with container-declared preprocessing.

    Reads `modules/processor/preprocessor_config.json` for the
    rescale factor and pad alignment. Falls back to the standard
    1/255 rescale + multiple-of-8 pad when the processor config is
    absent (those are the conventional defaults for SR models).
    """
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    proc_cfg = {}
    proc_file = cache_path / "modules" / "processor" / "preprocessor_config.json"
    if proc_file.exists():
        with open(proc_file) as f:
            proc_cfg = json.load(f)

    rescale_factor = (
        proc_cfg.get("rescale_factor", 1.0 / 255.0)
        if proc_cfg.get("do_rescale", True)
        else 1.0
    )
    pad_size = proc_cfg.get("pad_size", 8) if proc_cfg.get("do_pad", True) else 1

    import numpy as np
    arr = np.asarray(img, dtype=np.float32) * float(rescale_factor)
    # HWC → CHW → NCHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    # Pad H and W up to a multiple of `pad_size` (reflect padding,
    # matching the Swin2SR image processor convention). The padded
    # region is trimmed implicitly by the model's known upscale ratio
    # downstream; SR models are trained to ignore the reflected band.
    _, _, h, w = tensor.shape
    pad_h = (pad_size - h % pad_size) % pad_size
    pad_w = (pad_size - w % pad_size) % pad_size
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(
            tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, (h, w)


def cmd_upscale(args):
    """Image super-resolution via an upscaler container."""
    import torch
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

    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))

    from neurobrix.kernels.wrappers import set_hardware_profile
    set_hardware_profile(hw_profile)
    executor = RuntimeExecutor(pkg, execution_plan, mode=execution_mode)

    print(f"\n[4/4] Running upscale ({execution_mode})...")
    inputs = {input_var: pixel_values}
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = executor.execute(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.time() - t0
    except Exception as e:
        print(f"\n[ERROR] Upscale failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[Timing] Total execution: {wall:.2f}s")

    saved = save_image(outputs, args.output, family, executor, pkg)
    print("\n" + "=" * 60)
    print(f"SAVED: {saved}")
    print("=" * 60)
    return 0
