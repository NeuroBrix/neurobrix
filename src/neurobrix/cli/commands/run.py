"""
neurobrix run — Execute inference using NBX Engine.

DATA-DRIVEN DESIGN:
- CLI does NO business logic
- All args are mapped to global.* variables
- Family is read from manifest.json (set at trace/import time)
- The model (via variables.json) decides how to use them

ZERO HARDCODE: Defaults cascade from CLI > runtime/defaults.json > family config
"""

import sys
import json
import time
from pathlib import Path

from neurobrix import __version__
from neurobrix.cli.utils import find_model


def _try_warm_path(args) -> bool:
    """Attempt warm-path execution via running daemon. Returns True if handled."""
    from neurobrix.serving.client import DaemonClient

    if not DaemonClient.is_running():
        return False

    try:
        client = DaemonClient()
        client.connect()
        status = client.status()
    except Exception:
        return False

    # Only use warm path if daemon has the same model loaded
    if status.get("model") != args.model:
        client.close()
        return False

    print(f"[Run] Using warm daemon (PID {DaemonClient.get_pid()})")

    kwargs = {}
    if args.steps is not None:
        kwargs["steps"] = args.steps
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width
    if args.cfg is not None:
        kwargs["cfg"] = args.cfg
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.chat_mode is not None:
        kwargs["chat_mode"] = args.chat_mode
    if args.seed is not None:
        kwargs["seed"] = args.seed

    # Audio models: pass audio_path to daemon
    if getattr(args, 'audio', None):
        kwargs["audio_path"] = args.audio

    # For binary-output families (image, video, audio-wav): pass output_path
    # so daemon saves the file. Text families (llm/vlm/multimodal-text/stt/
    # audio_llm) return text in the JSON response and don't need a server-side
    # save path.
    family = status.get("family")
    if family:
        from neurobrix.core.runtime.output_dispatch import (
            get_output_format,
            resolve_output_path,
            resolve_mode,
        )
        try:
            fmt = get_output_format(family)
        except RuntimeError:
            fmt = "txt"
        if fmt != "txt":
            mode = resolve_mode(family, args)
            output_path = resolve_output_path(args.output, args.model, family, mode)
            kwargs["output_path"] = str(Path(output_path).resolve())

    try:
        result = client.generate(prompt=args.prompt or "", **kwargs)
        client.close()
    except RuntimeError as e:
        print(f"[Run] Daemon error: {e}")
        client.close()
        return False

    # Display result — data-driven by family output_format
    from neurobrix.core.runtime.output_dispatch import get_output_format
    timing = result.get("timing", {})
    total_s = timing.get("total_s", 0)

    try:
        fmt = get_output_format(family) if family else "txt"
    except RuntimeError:
        fmt = "txt"

    if fmt == "txt":
        text = result.get("text") or result.get("transcription") or ""
        tokens = result.get("tokens", 0)
        if tokens:
            print(f"\n[Output] Generated {tokens} tokens in {total_s}s")
        if text:
            print(f"\n{text}")
        elif result.get("output_path"):
            print(f"\nSAVED: {result['output_path']}")
    else:
        saved_path = result.get("output_path")
        if saved_path:
            print(f"\n{'='*70}")
            print(f"SAVED: {saved_path}")
            print(f"{'='*70}")
        else:
            print(f"\n[Output] Generation complete in {total_s}s")

    return True


def cmd_run(args):
    """Generate output using NeuroBrix Runtime."""
    import torch
    from neurobrix.nbx import NBXContainer
    from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.executor import RuntimeExecutor
    from neurobrix.core.config import get_output_processing

    # Auto-detect model from running daemon if --model omitted
    if args.model is None:
        from neurobrix.serving.client import DaemonClient
        if DaemonClient.is_running():
            try:
                client = DaemonClient()
                client.connect()
                status = client.status()
                args.model = status.get("model")
                client.close()
            except Exception:
                pass
        if args.model is None:
            print("ERROR: --model is required when no daemon is running.")
            sys.exit(1)

    # Warm path: if daemon is running with same model, use it
    if _try_warm_path(args):
        sys.exit(0)

    # GUARD: If daemon is running but warm path failed, refuse cold path.
    # Cold run + daemon = double GPU allocation = OOM. One task at a time.
    from neurobrix.serving.client import DaemonClient
    if DaemonClient.is_running():
        daemon_pid = DaemonClient.get_pid()
        print(f"ERROR: A serving daemon is already running (PID {daemon_pid}).")
        print(f"NeuroBrix runs one task at a time — the daemon is using the GPUs.")
        print(f"Either:")
        print(f"  1. Use the daemon:  neurobrix run --model {args.model} --prompt '...'")
        print(f"  2. Stop it first:   neurobrix stop")
        sys.exit(1)

    # DATA-DRIVEN: Find model by scanning all family directories
    try:
        nbx_path = find_model(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 70)
    print(f"NeuroBrix Run v{__version__}")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Hardware: {args.hardware or 'auto-detect'}")
    if args.prompt:
        prompt_display = f"{args.prompt[:50]}..." if len(args.prompt) > 50 else args.prompt
        print(f"Prompt: {prompt_display}")
    if getattr(args, 'audio', None):
        print(f"Audio: {args.audio}")
    print("=" * 70)

    # 1. Load NBX Container
    print("\n[1/4] Loading NBX container...")
    container = NBXContainer.load(str(nbx_path))

    # DATA-DRIVEN: Validate inputs against family YAML inputs.required spec
    manifest = container.get_manifest() or {}
    family = manifest.get("family")
    if family is None:
        print(f"ERROR: 'family' missing in manifest for '{args.model}'.")
        sys.exit(1)

    from neurobrix.core.runtime.output_dispatch import (
        validate_required_inputs,
        resolve_mode,
    )
    try:
        validate_required_inputs(family, args)
        mode = resolve_mode(family, args)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # Multimodal-strict build/mode coherence: a Janus-style .nbx is traced
    # for ONE generation_type at build time. If the user asks for a mode the
    # build cannot serve, error clearly here instead of running the image AR
    # path and writing image tokens to a .txt.
    cache_path = container._cache_path
    assert cache_path is not None
    _topo_path = cache_path / "topology.json"
    if _topo_path.exists():
        with open(_topo_path) as f:
            _topo = json.load(f)
        _build_gen_type = _topo.get("flow", {}).get("generation", {}).get("type", "")
        _mode_gen_type = {"text": "autoregressive_text", "image": "autoregressive_image"}.get(mode or "")
        if _mode_gen_type and _build_gen_type and _mode_gen_type != _build_gen_type:
            _supported_mode = "image" if _build_gen_type == "autoregressive_image" else "text"
            print(
                f"\nERROR: This '{args.model}' build supports only --mode "
                f"{_supported_mode} (its trace generation_type is "
                f"'{_build_gen_type}'). Re-import a build traced for "
                f"--mode {mode} to use that mode."
            )
            sys.exit(1)

    neural_components = container.get_neural_components()
    print(f"   Components: {[c.name for c in neural_components]}")

    # 2. Prism Allocation
    print("\n[2/4] Solving hardware allocation...")
    if args.hardware:
        hw_profile = load_profile(args.hardware)
    else:
        hardware_id = get_or_create_default_profile()
        hw_profile = load_profile(hardware_id)
    print(f"   Profile: {hw_profile.id} ({hw_profile.total_vram_gb:.1f} GB)")

    # Build InputConfig for activation profiling
    cache_path = container._cache_path
    assert cache_path is not None, "Container cache path must be set"
    defaults_path = cache_path / "runtime" / "defaults.json"
    if defaults_path.exists():
        with open(defaults_path) as f:
            cached_defaults = json.load(f)
    else:
        cached_defaults = {}

    # Resolution for the Prism activation estimate MUST match the resolution the
    # executor actually generates at — otherwise Prism over/under-estimates the VAE
    # activation and mis-decides tiling/placement. The executor's merged_defaults
    # fall back to the FAMILY config (config/families/<family>.yml), not just the
    # per-model defaults.json. run.py read only defaults.json + a hardcoded 1024
    # fallback, so for video (family default 512x512) Prism estimated the VAE at
    # 1024x1024 (8x the real activation) and force-tiled a VAE that fits natively
    # — producing tile seams. Mirror the executor's fallback chain: args ->
    # defaults.json -> family config -> 1024.
    from neurobrix.core.config import get_family_defaults as _get_family_defaults
    _fam_defaults = _get_family_defaults(family) if family else {}
    height = (getattr(args, 'height', None) or cached_defaults.get("height")
              or _fam_defaults.get("height") or 1024)
    width = (getattr(args, 'width', None) or cached_defaults.get("width")
             or _fam_defaults.get("width") or 1024)
    vae_scale = cached_defaults.get("vae_scale_factor", 8)
    # Video (5D) runtime dims — None/absent for image/LLM models, where the
    # profiler's video symbol overrides stay inert.
    num_frames = getattr(args, 'num_frames', None) or cached_defaults.get("num_frames")
    temporal_compression = cached_defaults.get("temporal_compression_ratio", 4)

    input_config = InputConfig(
        batch_size=2,  # CFG effectively doubles batch
        height=height,
        width=width,
        dtype="float16",
        vae_scale=vae_scale,
        num_frames=num_frames,
        temporal_compression=temporal_compression,
    )

    solver = PrismSolver()
    execution_plan = solver.solve_smart(container, hw_profile, input_config)

    # Apply CPU optimizations from hardware profile
    if hw_profile.cpu:
        from neurobrix.core.prism.cpu_config import apply_cpu_config
        apply_cpu_config(
            cpu=hw_profile.cpu,
            strategy=execution_plan.strategy,
            device_count=hw_profile.device_count,
            preferred_dtype=hw_profile.preferred_dtype,
        )

    print(f"   Strategy: {execution_plan.strategy}")
    for comp_name, alloc in execution_plan.components.items():
        print(f"   {comp_name} → {alloc.device}")

    # 3. Load RuntimePackage
    print("\n[3/4] Loading runtime...")
    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))

    # 4. Build Universal Inputs Dictionary
    print("\n[4/4] Preparing inputs...")

    inputs = {}
    if args.prompt:
        inputs["global.prompt"] = args.prompt
    if getattr(args, 'audio', None):
        inputs["global.audio_path"] = args.audio
    if getattr(args, 'reference_audio', None):
        inputs["global.reference_audio_path"] = args.reference_audio

    if args.steps is not None:
        inputs["global.num_inference_steps"] = args.steps
    if args.height is not None:
        inputs["global.height"] = args.height
    if args.width is not None:
        inputs["global.width"] = args.width
    if getattr(args, 'num_frames', None) is not None:
        inputs["global.num_frames"] = args.num_frames
    if getattr(args, 'fps', None) is not None:
        inputs["global.fps"] = args.fps
    if getattr(args, 'input_image', None):
        # I2V conditioning image -> global.image [1, 3, 1, H, W] in [-1, 1]
        # (the vendor VideoProcessor.preprocess contract: resize to the run
        # height/width, normalize to [-1, 1], single conditioning frame).
        # Boundary I/O per R34 — PIL here is file reading, not compute.
        import torch as _torch
        import numpy as _np
        from PIL import Image as _PILImage
        _img = _PILImage.open(args.input_image).convert("RGB")
        _h = int(args.height) if args.height else _img.height
        _w = int(args.width) if args.width else _img.width
        if (_img.width, _img.height) != (_w, _h):
            _img = _img.resize((_w, _h), _PILImage.LANCZOS)
        _arr = _np.asarray(_img, dtype="float32") / 127.5 - 1.0
        inputs["global.image"] = (_torch.from_numpy(_arr)
                                  .permute(2, 0, 1).unsqueeze(0).unsqueeze(2))
        # I2V temporal-VAE conditioning: some denoisers (Wan-I2V) encode the
        # first frame WITHIN the full clip, so the VAE sees [1,3,num_frames,H,W]
        # (frame0=image, rest=zeros). Pad global.image to num_frames when the
        # model's vae_encoder declares pad_image_to_num_frames. CogVideoX (single-
        # frame encode) leaves it at 1 frame. Data-driven via registry flag.
        try:
            from neurobrix.core.runtime.registry_flags import get_component_flag
            if get_component_flag(getattr(args, "model", None), "vae_encoder",
                                  "pad_image_to_num_frames", default=False):
                _nf = int(getattr(args, "num_frames", 0) or 0)
                _img5d = inputs["global.image"]
                if _nf > 1 and _img5d.shape[2] == 1:
                    _zeros = _torch.zeros(
                        _img5d.shape[0], _img5d.shape[1], _nf - 1,
                        _img5d.shape[3], _img5d.shape[4], dtype=_img5d.dtype)
                    inputs["global.image"] = _torch.cat([_img5d, _zeros], dim=2)
        except Exception:
            pass
        # CLIP pixel_values for a separate vision image_encoder (Wan-I2V, etc.):
        # a CLIP-preprocessed view of the SAME input image, distinct from the
        # VAE-conditioning global.image above. Data-driven from the embedded
        # modules/image_processor/preprocessor_config.json (resize shortest side
        # to `size`, center-crop `crop_size`, scale to [0,1], normalize by CLIP
        # mean/std). Boundary I/O per R34. .contiguous() before the NBXTensor
        # boundary (channels-last permute view).
        _proc_cfg = cache_path / "modules" / "image_processor" / "preprocessor_config.json"
        if _proc_cfg.exists():
            _pc = json.loads(_proc_cfg.read_text())
            def _dim(d, k, default):
                v = d.get(k)
                if isinstance(v, dict):
                    return int(v.get("height", v.get("shortest_edge", default)))
                return int(v) if v is not None else default
            _rs = _dim(_pc, "size", 224)
            _cs = _dim(_pc, "crop_size", _rs)
            _mean = _np.asarray(_pc.get("image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype="float32")
            _std = _np.asarray(_pc.get("image_std", [0.26862954, 0.26130258, 0.27577711]), dtype="float32")
            _ci = _PILImage.open(args.input_image).convert("RGB")
            if _pc.get("do_resize", True):
                _scale = _rs / min(_ci.width, _ci.height)
                _ci = _ci.resize((max(1, round(_ci.width * _scale)), max(1, round(_ci.height * _scale))), _PILImage.BICUBIC)
            _l = (_ci.width - _cs) // 2
            _t = (_ci.height - _cs) // 2
            _ci = _ci.crop((_l, _t, _l + _cs, _t + _cs))
            _ca = _np.asarray(_ci, dtype="float32") / 255.0
            if _pc.get("do_normalize", True):
                _ca = (_ca - _mean) / _std
            inputs["global.pixel_values"] = (_torch.from_numpy(_ca)
                                             .permute(2, 0, 1).unsqueeze(0).contiguous())

    # VACE control conditioning with no explicit control video: the all-generate
    # (unconditional / pure text→video) path. The vae_encoder encodes a zeros
    # control clip [1,3,num_frames,H,W]; the brick builds control_hidden_states
    # = cat([encode(0), encode(0), ones_mask]). Data-driven via the transformer's
    # vace_control_conditioning flag; only synthesized when global.image is absent.
    if "global.image" not in inputs:
        from neurobrix.core.runtime.registry_flags import get_component_flag as _gcf
        if _gcf(getattr(args, "model", None), "transformer",
                "vace_control_conditioning", default=None):
            import torch as _torch
            _nf = int(getattr(args, "num_frames", 0) or 1)
            _h = int(args.height) if args.height else 480
            _w = int(args.width) if args.width else 832
            inputs["global.image"] = _torch.zeros(1, 3, _nf, _h, _w,
                                                  dtype=_torch.float32)
            print(f"   VACE all-generate control: zeros clip "
                  f"[1,3,{_nf},{_h},{_w}] -> vae_encoder")

    if args.cfg is not None:
        inputs["global.guidance_scale"] = args.cfg
    if args.temperature is not None:
        inputs["global.temperature"] = args.temperature
    if args.repetition_penalty is not None:
        inputs["global.repetition_penalty"] = args.repetition_penalty
    if getattr(args, 'max_tokens', None) is not None:
        inputs["global.max_tokens"] = args.max_tokens
    if args.chat_mode is not None:
        inputs["global.chat_mode"] = args.chat_mode

    print(f"   CLI inputs: {list(inputs.keys())}")

    if args.seed is not None:
        inputs["global.seed"] = args.seed
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"   global.seed = {args.seed}")

    # Universal --set injection
    if args.set:
        for item in args.set:
            if "=" not in item:
                print(f"   [WARNING] Invalid --set format: {item} (expected key=value)")
                continue
            key, value = item.split("=", 1)
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            inputs[key] = value
            print(f"   {key} = {value}")

    print(f"   Total inputs: {len(inputs)}")

    # 5. Determine Execution Engine Mode
    # Mutually exclusive flags: --compiled / --sequential / --triton /
    # --triton-sequential. Default = compiled when no flag is passed.
    _mode_flags = [
        getattr(args, 'compiled', False),
        args.sequential,
        args.triton,
        getattr(args, 'triton_sequential', False),
    ]
    if sum(bool(f) for f in _mode_flags) > 1:
        print("\nERROR: Only one execution mode flag can be passed at a time.")
        print("       Choose one of: --compiled (default), --sequential, --triton, --triton-sequential")
        return 1

    if args.sequential:
        execution_mode = "sequential"
    elif args.triton or getattr(args, 'triton_sequential', False):
        # Triton mode: validate hardware compatibility
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n[ERROR] --triton mode is not compatible with Apple Metal GPUs.")
            print("   Metal/MPS backend does not support Triton kernels.")
            print("   This will be supported in a future version of NeuroBrix.")
            print("   Use default mode (without --triton) for Metal GPU inference.")
            return
        if not torch.cuda.is_available():
            import os
            os.environ.setdefault("TRITON_CPU_BACKEND", "1")
            print("   [Triton] No GPU detected — using Triton CPU backend (experimental)")
        if getattr(args, 'triton_sequential', False):
            execution_mode = "triton_sequential"
        else:
            execution_mode = "triton"
    else:
        # default OR --compiled explicit
        execution_mode = "compiled"

    # 6. Execute
    print("\n[Execute] Running pipeline...")
    print(f"   Engine: {execution_mode.upper()}")
    # Data-driven hardware capability surface for Triton kernel wrappers.
    # Set once per process from the resolved PrismProfile.
    from neurobrix.kernels.wrappers import set_hardware_profile
    set_hardware_profile(hw_profile)
    executor = RuntimeExecutor(pkg, execution_plan, mode=execution_mode)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_exec_start = time.time()
        outputs = executor.execute(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_exec_total = time.time() - t_exec_start
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6a. Print timing summary
    print(f"\n[Timing] Total execution: {t_exec_total:.2f}s")

    # 7. Save output — DATA-DRIVEN by family YAML output_format
    family = pkg.manifest.get("family")
    if family is None:
        raise RuntimeError(
            f"ZERO FALLBACK: 'family' missing in manifest for model '{args.model}'.\n"
            f"Model data incomplete. Re-import: neurobrix remove {args.model} && "
            f"neurobrix import <org>/<model>"
        )

    from neurobrix.core.runtime.output_dispatch import (
        get_output_format,
        resolve_output_path,
        save_output,
    )

    fmt = get_output_format(family)

    # Text-output families: print to stdout; only write file if --output given.
    if fmt == "txt":
        from neurobrix.core.runtime.output_dispatch import _extract_text, _extract_token_count
        text = _extract_text(outputs, executor)
        if text is None:
            print(f"\n[WARNING] No text output found. Available: {list(outputs.keys())}")
            sys.exit(1)
        tokens = _extract_token_count(outputs)
        if tokens:
            print(f"\n[Output] Generated {tokens} tokens")
        if args.output:
            output_path = resolve_output_path(args.output, args.model, family, mode)
            save_output(outputs, output_path, family, executor, pkg, mode=mode)
            print(f"\n[Success] Output saved to: {output_path}")
        else:
            print(f"\n{text}")
        sys.exit(0)

    # Binary-output families (image, video, audio-wav, multimodal-image)
    output_path = resolve_output_path(args.output, args.model, family, mode)
    saved = save_output(outputs, output_path, family, executor, pkg, mode=mode)

    print(f"\n{'='*70}")
    print(f"SAVED: {saved}")
    print(f"{'='*70}")
    return 0
