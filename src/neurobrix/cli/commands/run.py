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

    # For non-LLM non-audio: pass output_path so daemon saves the file
    family = status.get("family")
    if family not in ("llm", "audio"):
        output_path = args.output or f"output_{args.model}.png"
        kwargs["output_path"] = output_path

    try:
        result = client.generate(prompt=args.prompt or "", **kwargs)
        client.close()
    except RuntimeError as e:
        print(f"[Run] Daemon error: {e}")
        client.close()
        return False

    # Display result
    timing = result.get("timing", {})
    total_s = timing.get("total_s", 0)

    if family == "llm":
        text = result.get("text", "")
        tokens = result.get("tokens", 0)
        print(f"\n[Output] Generated {tokens} tokens in {total_s}s")
        if text:
            print(f"\n{text}")
    elif family == "audio":
        # Audio STT: transcription text
        transcription = result.get("transcription")
        if transcription:
            print(f"\n[Transcription] {transcription}")
        # Audio TTS: waveform saved
        saved_path = result.get("output_path")
        if saved_path:
            print(f"\n{'='*70}")
            print(f"SAVED: {saved_path}")
            print(f"{'='*70}")
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

    # DATA-DRIVEN: Validate input modality before proceeding
    manifest = container.get_manifest() or {}
    family = manifest.get("family")
    input_modality = manifest.get("input_modality", "text")

    # Image-to-image models (super-resolution, upscaling)
    if input_modality == "image":
        model_name = manifest.get("model_name", args.model)
        print(f"\nERROR: '{model_name}' is an image-to-image model requiring --input-image.")
        print(f"Image-to-image inference is not yet supported.")
        sys.exit(1)

    # Audio family input validation
    if family == "audio":
        audio_arg = getattr(args, 'audio', None)
        if not audio_arg and not args.prompt:
            print("ERROR: Audio model requires --audio <file> (STT) or --prompt <text> (TTS)")
            sys.exit(1)
    elif not args.prompt:
        print("ERROR: --prompt is required for this model family.")
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

    height = getattr(args, 'height', None) or cached_defaults.get("height", 1024)
    width = getattr(args, 'width', None) or cached_defaults.get("width", 1024)
    vae_scale = cached_defaults.get("vae_scale_factor", 8)

    input_config = InputConfig(
        batch_size=2,  # CFG effectively doubles batch
        height=height,
        width=width,
        dtype="float16",
        vae_scale=vae_scale,
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

    if args.steps is not None:
        inputs["global.num_inference_steps"] = args.steps
    if args.height is not None:
        inputs["global.height"] = args.height
    if args.width is not None:
        inputs["global.width"] = args.width
    if args.cfg is not None:
        inputs["global.guidance_scale"] = args.cfg
    if args.temperature is not None:
        inputs["global.temperature"] = args.temperature
    if args.repetition_penalty is not None:
        inputs["global.repetition_penalty"] = args.repetition_penalty
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
    if args.sequential:
        execution_mode = "native"
    elif args.triton:
        execution_mode = "triton"
    else:
        execution_mode = "compiled"

    # 6. Execute
    print("\n[Execute] Running pipeline...")
    print(f"   Engine: {execution_mode.upper()}")
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

    # 7. Find and Save Output - FAMILY-DRIVEN
    family = pkg.manifest.get("family")
    if family is None:
        raise RuntimeError(
            f"ZERO FALLBACK: 'family' missing in manifest for model '{args.model}'.\n"
            f"Model data incomplete. Re-import: neurobrix remove {args.model} && neurobrix import <org>/<model>"
        )

    # =========================================================================
    # LLM Family: Text/Token Output
    # =========================================================================
    if family == "llm":
        output_tokens = outputs.get("output_tokens")
        if output_tokens is None:
            output_tokens = outputs.get("global.output_tokens")
        if output_tokens is None:
            print(f"\n[WARNING] No output_tokens found")
            print(f"Available: {list(outputs.keys())}")
            sys.exit(1)

        print(f"\n[Output] Generated {output_tokens.shape[-1]} tokens")

        generated_text = None
        if "tokenizer" in executor.modules:
            tokenizer = executor.modules["tokenizer"]
            if hasattr(tokenizer, 'decode'):
                token_ids = output_tokens.flatten().tolist()
                generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        if args.output:
            output_path = args.output
            if output_path.endswith(".json"):
                import json as json_mod
                result = {
                    "model": args.model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": generated_text or str(output_tokens.flatten().tolist()),
                        },
                        "finish_reason": "stop",
                    }],
                }
                with open(output_path, 'w') as f:
                    json_mod.dump(result, f, indent=2, ensure_ascii=False)
            else:
                with open(output_path, 'w') as f:
                    f.write(generated_text if generated_text else str(output_tokens.flatten().tolist()))
            print(f"\n[Success] Output saved to: {output_path}")
        else:
            if generated_text:
                print(f"\n{generated_text}")
            else:
                print(f"\n{output_tokens.flatten().tolist()}")

        sys.exit(0)

    # =========================================================================
    # Audio Family: Transcription (STT) or Waveform (TTS)
    # =========================================================================
    if family == "audio":
        # STT: text transcription
        transcription = outputs.get("global.transcription")
        if transcription:
            print(f"\n[Transcription]")
            print(f"\n{transcription}")
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(transcription)
                print(f"\nSaved to: {args.output}")
            sys.exit(0)

        # TTS: waveform tensor
        waveform = outputs.get("global.output_audio")
        if waveform is not None:
            output_path = args.output or f"output_{args.model}.wav"
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            from neurobrix.core.config import get_output_processing
            audio_cfg = get_output_processing("audio")
            flow_sr = pkg.topology.get("flow", {}).get("audio", {}).get("sample_rate")
            sample_rate = flow_sr or pkg.defaults.get("sample_rate", audio_cfg.get("sample_rate", 24000))
            AudioOutputProcessor.save_waveform(waveform, output_path, sample_rate)
            print(f"\n{'='*70}")
            print(f"SAVED: {output_path}")
            print(f"{'='*70}")
            sys.exit(0)

        print(f"\n[WARNING] Audio model produced no transcription or waveform output")
        print(f"Available: {list(outputs.keys())}")
        sys.exit(1)

    # =========================================================================
    # Image/Video Family: Tensor Output
    # =========================================================================
    output_path = args.output or f"output_{args.model}.png"

    final_output = executor.get_final_output(outputs)
    if final_output is None:
        final_comp = executor.get_final_component()
        print(f"\n[WARNING] No tensor output from '{final_comp}'")
        print(f"Available: {[k for k in outputs.keys() if isinstance(outputs.get(k), torch.Tensor)]}")
        sys.exit(1)

    print(f"\n[Output] Final tensor: {final_output.shape}")
    print(f"[Output] Raw VAE output: min={final_output.min():.4f}, max={final_output.max():.4f}, mean={final_output.mean():.4f}")

    output_range = pkg.defaults.get("output_range")
    if output_range is None:
        output_cfg = get_output_processing(family)
        output_range = output_cfg.get("output_range", [-1.0, 1.0])
    min_val, max_val = output_range

    output_cfg = get_output_processing(family)
    batch_axis = output_cfg.get("batch_axis", 0)
    channel_axis = output_cfg.get("channel_axis", 1)
    valid_channels = output_cfg.get("valid_channels", [1, 3, 4])
    bit_depth = output_cfg.get("bit_depth", 8)
    layout = output_cfg.get("layout", "CHW")

    img = torch.select(final_output, batch_axis, 0).cpu().float()

    from neurobrix.core.module.output_processor import OutputProcessor
    processor = OutputProcessor.from_package(pkg)
    img = processor.process(img, output_range)

    img = img.clamp(0, 1)

    if layout == "CHW" and img.dim() == 3:
        actual_channel_axis = channel_axis - 1 if batch_axis < channel_axis else channel_axis
        if img.shape[actual_channel_axis] in valid_channels:
            img = img.permute(1, 2, 0)

    from PIL import Image
    import numpy as np

    if bit_depth == 8:
        img_np = (img.numpy() * 255).astype(np.uint8)
    elif bit_depth == 16:
        img_np = (img.numpy() * 65535).astype(np.uint16)
    else:
        img_np = (img.numpy() * 255).astype(np.uint8)

    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)

    Image.fromarray(img_np).save(output_path)

    print(f"\n{'='*70}")
    print(f"SAVED: {output_path}")
    print(f"{'='*70}")

    return 0
