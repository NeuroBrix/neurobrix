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
import logging
import sys
import time
from pathlib import Path
from nbxtensor import device as _nbx_device

log = logging.getLogger(__name__)


def _mode_as_flag(args):
    """`--mode X` is the engine flag `--X` (one rule, cli/engine_choice.py): set here, so passing both is refused
    there like two flags."""
    if getattr(args, "mode", None):
        setattr(args, args.mode.replace("-", "_"), True)     # the same choice said twice stays one
    return args


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
        if _nbx_device.count() > 0:
            _nbx_device.drain()


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
        log.error("Error: --model, --input and --output are required.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Error: input image not found: {input_path}")
        sys.exit(1)

    # The engine, chosen as `run` chooses it (cli/engine_choice.py): no flag = PyTorch where torch is installed, the
    # Triton engine where it is not; a Triton mode's backend checked; a PyTorch mode without torch refused by name.
    from neurobrix.cli.engine_choice import choose_engine, declare_for
    if (execution_mode := choose_engine(_mode_as_flag(args), log)) is None:
        sys.exit(1)

    from neurobrix.cli.utils import resolve_model_arg
    args.model = resolve_model_arg(args.model)
    nbx_path = find_model(args.model)

    log.debug("[1/4] Loading container...")
    container = NBXContainer.load(str(nbx_path))
    manifest = container.get_manifest() or {}
    family = manifest.get("family")
    if family != "upscaler":
        log.error(
            f"Error: model '{args.model}' is family='{family}', not "
            f"'upscaler'. Use `nbx run` for non-upscaler models.")
        sys.exit(1)

    cache_path = container._cache_path
    assert cache_path is not None, "Container cache path must be set"
    with open(cache_path / "topology.json") as f:
        topology = json.load(f)
    input_var = _find_input_variable(topology)

    log.debug("[2/4] Solving hardware allocation...")
    if args.hardware:
        hw_profile = load_profile(args.hardware)
    else:
        hw_profile = load_profile(get_or_create_default_profile())
    log.debug(f"   Profile: {hw_profile.id} ({hw_profile.total_vram_gb:.1f} GB)")

    log.debug("[3/4] Preprocessing input image...")
    pixel_values, (orig_h, orig_w) = _load_and_preprocess_image(
        args.input, cache_path)
    _, _, in_h, in_w = pixel_values.shape
    log.debug(f"   Input tensor: {tuple(pixel_values.shape)} "
              f"(original {orig_h}×{orig_w}, padded {in_h}×{in_w})")

    input_config = InputConfig(
        batch_size=1, height=in_h, width=in_w, dtype="float32",
    )
    solver = PrismSolver()
    execution_plan = solver.solve_smart(container, hw_profile, input_config,
                                        mode=execution_mode)
    log.debug(f"   Strategy: {execution_plan.strategy}")
    # The choice, said out loud. Prism scores every viable strategy and takes
    # the fastest — invisible unless printed, and an engine that decides
    # without saying so is indistinguishable from one that decides badly.
    if getattr(execution_plan, "selection_reason", ""):
        log.debug(f"   Why:      {execution_plan.selection_reason}")
    _cards = sorted({
        d for a in execution_plan.components.values()
        for d in (getattr(a, "devices", None) or [])
    })
    if _cards:
        log.debug(f"   Devices:  {', '.join(_cards)}"
                  f"  ({execution_plan.total_memory_mb:.0f} MB planned)")
    if getattr(execution_plan, "host_footprint", None):
        from neurobrix.core.prism.host_footprint import summary as _host_summary
        log.debug(f"   Host:     {_host_summary(execution_plan.host_footprint)}")

    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))

    declare_for(execution_mode, hw_profile)        # the Triton branch's declaration only where a Triton kernel runs
    executor = RuntimeExecutor(pkg, execution_plan, mode=execution_mode)

    log.debug(f"[4/4] Running upscale ({execution_mode})...")
    inputs = {input_var: pixel_values}
    try:
        _drain_device(execution_mode)
        t0 = time.time()
        outputs = executor.execute(inputs)
        _drain_device(execution_mode)
        wall = time.time() - t0
    except Exception as e:
        from neurobrix.cli import _fail
        _fail(e, what="the upscale failed: ")

    log.info(f"[Timing] Total execution: {wall:.2f}s")

    # The contract is input x scale exactly. The scale lookup and the
    # crop live in save_image — ONE point of truth for both entry
    # points (D-UPSCALE-SERVING-CROP); this caller only passes the
    # original size it owns.
    saved = save_image(outputs, args.output, family, executor, pkg,
                       orig_hw=(orig_h, orig_w))
    print(f"SAVED: {saved}")
    return 0
