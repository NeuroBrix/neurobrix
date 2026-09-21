"""
neurobrix run — Execute inference using NBX Engine.

DATA-DRIVEN DESIGN:
- CLI does NO business logic
- All args are mapped to global.* variables
- Family is read from manifest.json (set at trace/import time)
- The model (via variables.json) decides how to use them

ZERO HARDCODE: Defaults cascade from CLI > runtime/defaults.json > family config
"""

import os as _os_dbg
import sys
import json
import time
from pathlib import Path

from neurobrix import __version__
from neurobrix.cli.utils import find_model

from neurobrix.core.runtime_values import (MissingRuntimeValue,
                                          resolve as _resolve_runtime)


def _rt(name, args, container, family, **kw):
    """request -> container runtime/defaults.json -> family config -> refuse.

    One door for every runtime dimension and parameter. A literal standing in
    for one of these is a claim the engine never checked: `height`/`width` fell
    through to 1024 and the plan came out identical for a 448x448 request and a
    160x112 one (measured 2026-09-17).
    """
    return _resolve_runtime(name, request=args, container=container,
                            family=family, **kw)



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
    # Warm-boundary symmetry (doctrine: serving RPCs hide R30 gaps): the
    # resolved mode and speaker travel to the daemon exactly like the
    # cold path ships them to the flow. Mode resolves once, here — the
    # binary-output path block below reuses it.
    family = status.get("family")
    mode = None
    if family:
        from neurobrix.core.runtime.output_dispatch import resolve_mode
        try:
            mode = resolve_mode(family, args)
        except RuntimeError:
            mode = getattr(args, "mode", None)
    if mode is not None:
        kwargs["mode"] = mode
    if getattr(args, 'speaker', None) is not None:
        kwargs["speaker"] = args.speaker
    if args.seed is not None:
        kwargs["seed"] = args.seed

    # Audio models: pass audio_path to daemon
    if getattr(args, 'audio', None):
        kwargs["audio_path"] = args.audio

    # For binary-output families (image, video, audio-wav): pass output_path
    # so daemon saves the file. Text families (llm/vlm/multimodal-text/stt/
    # audio_llm) return text in the JSON response and don't need a server-side
    # save path.
    if family:
        from neurobrix.core.runtime.output_dispatch import (
            get_output_format,
            resolve_output_path,
        )
        try:
            fmt = get_output_format(family)
        except RuntimeError:
            fmt = "txt"
        if fmt != "txt":
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


def run_entry(args):
    """The dispatcher's entry: under `--explain-plan --json` the plan is the only
    thing on stdout, every human line of the run's preamble goes to stderr
    (`json_out`). `cmd_run` below is the run itself, unchanged and read as such
    by the tests that inspect its source."""
    from neurobrix.cli.json_out import human_lines_to_stderr
    with human_lines_to_stderr(bool(getattr(args, "json", False) and getattr(args, "explain_plan", False))):
        return cmd_run(args)


def cmd_run(args):
    """Generate output using NeuroBrix Runtime."""
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

    # Agent mode: the orchestration loop above inference has its own
    # daemon-first wiring (neurobrix.agent is engine-blind; adapters live
    # in cli/commands/agent.py).
    if getattr(args, "mode", None) == "agent":
        from neurobrix.cli.commands.agent import run_agent_mode
        sys.exit(run_agent_mode(args))

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

    # Mutually exclusive flags: --compiled / --sequential / --triton /
    # --triton-sequential. Default = compiled when no flag is passed.
    # Resolved BEFORE the container gates: capability gates (e.g. the
    # audio-mode engine gate) need the resolved engine name.
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
        # Triton mode: the backend for this hardware must actually be present.
        # Both gates below REFUSE with an install command instead of letting
        # the run die later inside the Triton driver — and neither of them
        # fetches anything on the user's behalf.
        from neurobrix.triton.cpu_backend import (
            TritonCPUNotInstalledError, ensure_triton_cpu_or_raise,
        )
        from neurobrix.triton.metal_backend import (
            TritonMetalNotInstalledError, ensure_triton_metal_or_raise,
        )
        try:
            # A census shadow (NBX_CENSUS=1, kernels/census.py) launches nothing: it
            # runs with NO visible device on purpose — the door that proves it cannot
            # touch a card — so the backend-presence gates below, which answer for a
            # process that WILL launch, do not apply to it.
            from neurobrix.kernels import census as _census
            if not _census.active():
                # Apple GPUs: upstream Triton has no Metal target, but an
                # out-of-tree backend exists. This replaced a hardcoded
                # "supported in a future version" message that was already
                # out of date.
                ensure_triton_metal_or_raise()
                # CPU-only profile: triton-cpu is a separate upstream package.
                # Previously TRITON_CPU_BACKEND was set without checking it was
                # installed, so the run died in the driver a step later.
                ensure_triton_cpu_or_raise()
        except (TritonMetalNotInstalledError, TritonCPUNotInstalledError) as exc:
            print(f"\n[ERROR] {exc}")
            return 1
        if getattr(args, 'triton_sequential', False):
            execution_mode = "triton_sequential"
        else:
            execution_mode = "triton"
    else:
        # default OR --compiled explicit
        execution_mode = "compiled"

    # 1. Load NBX Container
    print("\n[1/4] Loading NBX container...")
    container = NBXContainer.load(str(nbx_path))

    # DATA-DRIVEN: Validate inputs against family YAML inputs.required spec
    manifest = container.get_manifest() or {}

    # CHAIN OF CUSTODY. The runtime cache is keyed by model NAME, so a local
    # re-build silently replaces the graph of a published model under an
    # unchanged name — and nothing in the run said which container had
    # actually executed. That produced a wrong verdict on 2026-09-02: a probe
    # reported "fails at HEAD on the hub container" while running a build that
    # had overwritten it eight hours earlier. Naming the container on every
    # run is what makes that impossible to repeat silently.
    _built = manifest.get("created_at")
    if _built:
        print(f"Container: built {_built}  ({nbx_path})")

    family = manifest.get("family")
    if family is None:
        print(f"ERROR: 'family' missing in manifest for '{args.model}'.")
        sys.exit(1)

    from neurobrix.core.runtime.output_dispatch import (
        validate_required_inputs,
        resolve_mode,
        get_family_config,
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
        # Generative-image contract (P-OMNI-GEN model 2/3): a build that
        # declares topology.flow.image_gen serves --mode image through
        # the diffusion leg — the autoregressive-image coherence gate
        # below is for VQ-AR builds (Janus class), not this contract
        # (the speech-gate pattern: capability comes from the DECLARED
        # contract, never the model name).
        if mode == "image" and _topo.get("flow", {}).get("image_gen"):
            _mode_gen_type = None
        if _mode_gen_type and _build_gen_type and _mode_gen_type != _build_gen_type:
            _supported_mode = "image" if _build_gen_type == "autoregressive_image" else "text"
            print(
                f"\nERROR: This '{args.model}' build supports only --mode "
                f"{_supported_mode} (its trace generation_type is "
                f"'{_build_gen_type}'). Re-import a build traced for "
                f"--mode {mode} to use that mode."
            )
            sys.exit(1)
        # Speech capability gates — scoped to multimodal_strict families
        # (the generative-speech leg: builds where --mode audio routes
        # through topology.flow.speech). Audio-native families (tts, stt,
        # audio_llm) serve mode "audio" through their own flow handlers,
        # closed in both engines — these gates must never fire for them.
        _mm_strict = (
            get_family_config(family).get("modes", {}).get("multimodal_strict", False)
        )
        if _mm_strict and mode == "audio":
            # All four engines serve --mode audio: the compiled leg
            # (core/flow/speech.py) and its R33-pure triton mirror
            # (triton/flow/speech.py) consume the same contract and the
            # same graphs (P-OMNI-GEN §1, R30).
            # --mode audio requires the generative-speech contract in the
            # container (topology.flow.speech, emitted from the model's
            # declared registry contract). Builds without it refuse HERE.
            if not _topo.get("flow", {}).get("speech"):
                print(
                    f"\nERROR: This '{args.model}' build carries no "
                    f"generative-speech contract (topology.flow.speech). "
                    f"--mode audio requires a speech-declaring build "
                    f"(P-OMNI-GEN speech leg)."
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
    # An image request's size is the IMAGE's size. Without this the cascade
    # fell through to a cached or family default and Prism planned the SAME
    # memory whatever was asked of it — measured 2026-09-17 on hat-s-x4:
    #     --input-image apple_448.png      planned 278 MB
    #     --input-image apple_160x112.png  planned 278 MB
    # a 22.4x difference in pixels and not one byte of difference in the plan.
    # The consequence is not academic: the per-cell memory gate is fed that
    # number, so it could not refuse a cell that went on to hold 8408 MB live
    # and take the machine to 127 MB before the OS killed it.
    # An image request's size is the IMAGE's size. If the file cannot be read,
    # REFUSE: planning on a declared default for a request whose real size is
    # unknown is how the same 278 MB plan came out for 448x448 and 160x112.
    _img_hw = None
    _img_path = getattr(args, 'input_image', None)
    if _img_path and not getattr(args, 'height', None) and not getattr(args, 'width', None):
        from PIL import Image as _PILImage
        try:
            with _PILImage.open(_img_path) as _im:
                _img_hw = (_im.size[1], _im.size[0])       # PIL gives (W, H)
        except Exception as _e:                            # noqa: BLE001
            raise MissingRuntimeValue(
                f"the request names an input image the engine cannot read "
                f"({_img_path!r}: {type(_e).__name__}: {_e}), so its height and "
                f"width are unknown. Planning on a declared default for a "
                f"request of unknown size is what this refusal exists to "
                f"prevent — pass --height/--width, or give a readable image."
            ) from _e

    # OPTIONAL: a speech model, a TTS or an LLM has no spatial extent. Requiring
    # height/width of every model refused whisper, Kokoro and TinyLlama outright
    # — the same over-reach as demanding a VAE scale from a model with no VAE.
    # Where a spatial request exists the image supplies them, so they are never
    # silently absent for the models that need them.
    # The container's own output size — the SAME authority the executor renders at when
    # the request names none (`resolution.container_size`, 2026-09-21): a plan budgeted
    # at the VAE's trace extent while the flow decoded 81 frames at 480x832 asked 24.8 GB
    # at one conv against a 19.7 GB plan. A request-side fact still outranks it.
    from neurobrix.core.runtime.resolution.container_size import container_output_size
    # The topology as the executor reads it: the cache path's own file (the container
    # object answers an empty topology for an extracted directory).
    _topo_path = cache_path / "topology.json"
    _topo_components = (json.load(open(_topo_path)).get("components") or {}) if _topo_path.exists() else {}
    _cos = container_output_size(manifest, cached_defaults, _topo_components)
    height = _rt("height", args, cached_defaults, _fam_defaults,
                 extra=[("the input image's height", _img_hw[0] if _img_hw else None),
                        ("the container's own output height", _cos[0] if _cos else None)],
                 default=None)
    width = _rt("width", args, cached_defaults, _fam_defaults,
                extra=[("the input image's width", _img_hw[1] if _img_hw else None),
                       ("the container's own output width", _cos[1] if _cos else None)],
                default=None)
    # OPTIONAL by nature: a model without a VAE has no scale factor, a model
    # without a temporal axis has no compression. Absent is a legitimate answer
    # for these two, and only for these two.
    # The VAE scale in the container's own spellings (manifest, defaults, trace), the
    # same brick the executor reads: without it the plan's spatial symbols bind to nothing.
    from neurobrix.core.runtime.resolution.container_size import vae_scale_factor as _vsf
    vae_scale = _rt("vae_scale_factor", args, cached_defaults, _fam_defaults,
                    extra=[("the container's own VAE scale", _vsf(manifest, cached_defaults, _topo_components))],
                    default=None)
    num_frames = _rt("num_frames", args, cached_defaults, _fam_defaults, default=None)
    temporal_compression = _rt("temporal_compression_ratio", args, cached_defaults,
                               _fam_defaults, default=None)

    # The batch the flow will ACTUALLY run. `batch_size=2` stood here for every
    # model with the comment "CFG effectively doubles batch" — applied to
    # upscalers and speech models, which run no classifier-free guidance, and
    # overriding the batch_size: 1 that diffusion containers themselves declare.
    # CFG doubles the batch only where guidance is in force, and the engine
    # already treats `guidance_scale` as the declaration of that
    # (core/flow/autoregressive.py refuses without it).
    # A request carrying ONE image or ONE prompt is a batch of one. That is
    # read off the request, not assumed: it is consulted after the request's own
    # explicit --batch-size and after nothing else, so a container that declares
    # a different batch still wins over the derivation only when the request
    # names no single item.
    _single = 1 if (getattr(args, 'input_image', None)
                    or getattr(args, 'audio', None)
                    or getattr(args, 'prompt', None)) else None
    batch_size = _rt("batch_size", args, cached_defaults, _fam_defaults,
                     extra=[("the request's single input item", _single)],
                     why="It is the batch Prism plans for.")
    _guidance = _rt("guidance_scale", args, cached_defaults, _fam_defaults, default=None)
    if _guidance is not None and float(_guidance) > 1.0:
        batch_size *= 2

    # The dtype execution will resolve, not a literal. Every container on this
    # rack declares one (float32 for swin2SR, bfloat16 for TinyLlama, float16
    # for whisper), and "float16" stood here for all of them.
    # The manifest is container-declared data as much as runtime/defaults.json,
    # and three upscalers on this rack declare their dtype ONLY there.
    dtype = _rt("dtype", args, cached_defaults, _fam_defaults,
                extra=[("the container manifest's dtype", manifest.get("dtype"))],
                why="It sizes every tensor in the plan.")

    input_config = InputConfig(
        batch_size=batch_size,
        height=height,
        width=width,
        dtype=dtype,
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
            mode=execution_mode,
        )

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
    for comp_name, alloc in execution_plan.components.items():
        print(f"   {comp_name} → {alloc.device}")

    if getattr(args, "explain_plan", False):
        # The plan, and nothing after it: no weights load, no card is touched
        # beyond what the hardware profile read. What is printed is the plan
        # object the runtime would have received.
        from neurobrix.core.prism.solver import explain_plan, plan_record
        print("\n[plan] --explain-plan: the placement decision, read from the plan the runtime would receive\n")
        if getattr(args, "json", False):
            from neurobrix.cli.json_out import emit
            emit("explain-plan", {"model": getattr(args, "model", None), **plan_record(execution_plan)})
        else:
            print(explain_plan(execution_plan))
        return 0

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
    if num_frames is not None:
        # The frame count the CLI RESOLVED (request → container defaults → family), not only the
        # raw argument: a video flow reading `global.num_frames` for its image conditioning
        # refused every request that named none (Allegro-TI2V, CogVideoX-5b-I2V in the
        # census, 2026-09-21) while the container declared 88 and 49.
        inputs["global.num_frames"] = int(num_frames)
    if getattr(args, 'fps', None) is not None:
        inputs["global.fps"] = args.fps
    if getattr(args, 'input_image', None):
        # Image input routed through the shared CLI/daemon brick (mirror
        # of AudioInputProcessor; numpy DSP core shared with the triton
        # path, R34) — single source of truth, output_dispatch pattern.
        from neurobrix.core.module.vision.input_processor import (
            prepare_image_inputs,
        )
        inputs.update(prepare_image_inputs(
            pkg.topology, getattr(args, "model", None), args.input_image,
            cache_path, height=args.height, width=args.width,
            # The frame count cmd_run already resolved, NOT a second read of
            # the raw argument: without `--frames` that re-derivation was 0, so
            # `pad_to_num_frames` was 0 and the still image stayed a ONE-frame
            # clip for every model declaring `pad_image_to_num_frames` —
            # whatever its container declared (Allegro-TI2V 88, the Wan I2V
            # pair 81, VACE 81). Allegro-TI2V's VAE compresses time by 4 and
            # refused the extent-1 clip outright; the others simply conditioned
            # on one frame. height/width stay on the raw argument on purpose:
            # absent, the processor keeps the source image's own size.
            num_frames=int(num_frames or 0)))
        if _os_dbg.environ.get("NBX_DEBUG") == "1":
            _img = inputs.get("global.image")
            print(f"   [Inputs] image clip {tuple(getattr(_img, 'shape', ()))} from num_frames={num_frames!r} "
                  f"(request {getattr(args, 'num_frames', None)!r}, container {cached_defaults.get('num_frames')!r})", flush=True)
        # Upscaler metadata key, not a runtime input (the dedicated
        # `nbx upscale` path owns the exact-size crop on this side).
        inputs.pop("_upscale_orig_hw", None)

    if getattr(args, 'input_video', None):
        # Video understanding input — the video variant of the build's
        # declared image preprocessing (native_patch_grid →
        # native_patch_grid_video). Emits the vendor model_input_names
        # (pixel_values_videos / video_grid_thw) plus the per-video
        # M-RoPE temporal scale (video_second_per_grid). --fps overrides
        # the vendor sampling default when given.
        from neurobrix.core.module.vision.input_processor import (
            ImageInputProcessor as _VIP,
        )
        _vlm_blk = pkg.topology.get("flow", {}).get("vlm") or {}
        _vlm_in = _vlm_blk.get("input", {})
        if not _vlm_in.get("preprocessing"):
            raise RuntimeError(
                "ZERO FALLBACK: --input-video needs a build whose "
                "topology.flow.vlm declares an image preprocessing type "
                "(video understanding rides the vision tower).")
        _vid = _VIP.process(
            f"{_vlm_in['preprocessing']}_video", args.input_video,
            preprocessor_config=(_vlm_blk.get("preprocessing") or {}),
            fps=(float(args.fps) if getattr(args, 'fps', None) else None))
        for _k, _v in _vid.items():
            inputs[f"global.{_k}"] = _v

    # VACE control conditioning with no explicit control video: the all-generate
    # (unconditional / pure text→video) path. The vae_encoder encodes a zeros
    # control clip [1,3,num_frames,H,W]; the brick builds control_hidden_states
    # = cat([encode(0), encode(0), ones_mask]). Data-driven via the transformer's
    # vace_control_conditioning flag; only synthesized when global.image is absent.
    if "global.image" not in inputs:
        from neurobrix.core.runtime.registry_flags import get_component_flag as _gcf
        if _gcf(getattr(args, "model", None), "transformer",
                "vace_control_conditioning", default=None):
            _nf = int(getattr(args, "num_frames", 0) or 1)
            _h = int(args.height) if args.height else 480
            _w = int(args.width) if args.width else 832
            if execution_mode in ("triton", "triton_sequential"):
                # The Triton branch's container (R33: no torch in this process).
                import numpy as _np
                from neurobrix.kernels.nbx_tensor import NBXTensor as _NBXT
                inputs["global.image"] = _NBXT.from_numpy(
                    _np.zeros((1, 3, _nf, _h, _w), dtype=_np.float32))
            else:
                import torch as _torch
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
    if getattr(args, 'top_k', None) is not None:
        inputs["global.top_k"] = args.top_k
    if getattr(args, 'top_p', None) is not None:
        inputs["global.top_p"] = args.top_p
    if getattr(args, 'max_tokens', None) is not None:
        inputs["global.max_tokens"] = args.max_tokens
    if args.chat_mode is not None:
        inputs["global.chat_mode"] = args.chat_mode
    # Resolved output mode travels to the flow handlers (the speech leg
    # activates on global.mode == "audio" + topology.flow.speech; every
    # other handler ignores the key). Speaker preset rides along for
    # speech-capable builds.
    if mode is not None:
        inputs["global.mode"] = mode
    if getattr(args, 'speaker', None) is not None:
        inputs["global.speaker"] = args.speaker

    print(f"   CLI inputs: {list(inputs.keys())}")

    if args.seed is not None:
        inputs["global.seed"] = args.seed
        if execution_mode in ("compiled", "sequential"):
            # The ATen branch's RNG. The Triton flows seed their own
            # streams from global.seed (R33: a --triton run never loads torch).
            import torch
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
            stripped = value.strip()
            if stripped[:1] in ("[", "{"):
                # A list or an object, as JSON. The one runtime variable that
                # needs it is `global.input_token_ids`: the serving path sets
                # it programmatically, and a measurement that must land on an
                # EXACT context length — the head_dim cell of the protocol —
                # cannot get there through a text prompt, because the number
                # of tokens a sentence becomes is the tokenizer's to decide.
                import json as _json_set
                try:
                    value = _json_set.loads(stripped)
                except ValueError as exc:
                    raise SystemExit(
                        f"--set {key}: value starts with {stripped[:1]!r} so it "
                        f"is read as JSON, and it does not parse ({exc}). "
                        f"Quote it differently rather than have it silently "
                        f"become a string.")
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            inputs[key] = value
            shown = value if not isinstance(value, list) else (
                f"[{len(value)} values: {value[:4]}...]" if len(value) > 8 else value)
            print(f"   {key} = {shown}")

    print(f"   Total inputs: {len(inputs)}")

    # 5. Determine Execution Engine Mode
    # 6. Execute (engine resolved before the container gates, above)
    print("\n[Execute] Running pipeline...")
    print(f"   Engine: {execution_mode.upper()}")
    # Data-driven hardware capability surface for Triton kernel wrappers.
    # Set once per process from the resolved PrismProfile.
    #
    # GATED ON THE MODE, because the surface it configures belongs to the
    # Triton wrappers and the ATen path never touches them. Importing that
    # module pulls the kernel op modules, which carry real @triton.jit
    # decorators and cannot exist without a Triton wheel — and macOS has
    # none. Measured 2026-09-10 by blocking the module: a compiled run died
    # with `No module named 'triton'` AFTER Prism had chosen `single_gpu`
    # and the engine had printed `Engine: COMPILED`, so a fresh Mac could
    # not start the engine in ANY mode.
    if execution_mode != "compiled":
        from neurobrix.kernels.wrappers import set_hardware_profile
        set_hardware_profile(hw_profile)
    executor = RuntimeExecutor(pkg, execution_plan, mode=execution_mode)

    def _drain_device():
        # The wall clock below brackets the device work: each engine drains
        # its own device — the ATen branch through torch, the Triton branch
        # through the allocator's runtime (R33: no torch in that process).
        if execution_mode in ("compiled", "sequential"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            if DeviceAllocator.device_count() > 0:
                DeviceAllocator.device_synchronize()

    try:
        _drain_device()
        t_exec_start = time.time()
        outputs = executor.execute(inputs)
        _drain_device()
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

    from neurobrix.kernels import census as _census_out
    if _census_out.active():
        # A shadow run has no artefact to write: its product is the key record.
        print("\n[census] shadow run complete — keys recorded, no output written", flush=True)
        return 0
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
