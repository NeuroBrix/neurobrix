"""
NeuroBrix Runtime Executor

ORCHESTRATOR ONLY: This file orchestrates execution flow.
Flow logic is delegated to flow handlers in core/runtime/flow/.

ZERO SEMANTIC: Executes flow defined in topology.json mechanically.
ZERO HARDCODE: All bindings come from topology.json connections.
"""
import torch
import json
import gc
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from neurobrix.core.runtime.loader import RuntimePackage

logger = logging.getLogger(__name__)
from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
from neurobrix.core.runtime.factory import ExecutorFactory
from neurobrix.core.config import get_family_defaults, get_family_config, get_device_prefix
from neurobrix.core.strategies import get_strategy, StrategyContext, ExecutionStrategy
from neurobrix.core.module.scheduler.factory import SchedulerFactory
from neurobrix.nbx.cache import ensure_extracted

# Import modular components
from neurobrix.core.flow import FlowContext, get_flow_handler
from neurobrix.core.runtime.resolution.input_resolver import InputResolver
from neurobrix.core.runtime.resolution.input_synthesizer import InputSynthesizer
from neurobrix.core.runtime.resolution.output_extractor import OutputExtractor
from neurobrix.core.cfg.engine import CFGEngine
from neurobrix.core.module.tiling_engine import TilingEngine
from neurobrix.core.runtime.tensor_compat import is_tensor as _is_tensor



class RuntimeExecutor:
    """
    NeuroBrix Runtime Executor - Orchestrator Only

    RESPONSIBILITY: Setup and dispatch to flow handlers.
    DOES NOT: Contain flow logic, CFG logic, synthesis logic.

    Flow handlers (in flow/) contain the actual execution logic.
    Helper classes handle input resolution, synthesis, output extraction, and CFG.
    """

    def __init__(self, runtime_package: RuntimePackage, execution_plan: Any, mode: str = "compiled"):
        """
        Args:
            runtime_package: Loaded NBX package (immutable).
            execution_plan: Prism PipelineExecutionPlan (contains allocations).
            mode: Execution engine mode. One of:
                  - "compiled" (default): PyTorch fused (CompiledSequence + cuDNN)
                  - "sequential": PyTorch eager op-by-op
                  - "triton": Triton-pure compiled (TritonSequence)
                  - "triton_sequential": Triton-pure op-by-op (debug)
                  See CLAUDE.md "Execution Modes" section. The legacy value
                  "native" is deprecated — use "sequential".
        """
        self.pkg = runtime_package
        self.plan = execution_plan
        self.mode = mode
        self.executors: Dict[str, Any] = {}
        self.modules: Dict[str, Any] = {}
        self.variable_resolver: Optional[VariableResolver] = None

        # Strategy from Prism - initialized after executors are set up
        self.strategy: Optional[ExecutionStrategy] = None
        self._strategy_name = getattr(self.plan, 'strategy', 'single_gpu')

        # Path string for ExecutorFactory
        self._nbx_path_str = str(self.pkg.root_path)

        # Index connections from topology.json for O(1) lookup
        self._connections_index = self._index_connections()

        # ZERO HARDCODE: Get loop_id from variables contract
        self._loop_id = self._get_loop_id()

        # Helper modules (initialized in execute)
        self._input_resolver: Optional[InputResolver] = None
        self._input_synthesizer: Optional[InputSynthesizer] = None
        self._output_extractor: Optional[OutputExtractor] = None
        # Note: CFGEngine is created in _create_flow_handler where FlowContext is available

        # Universal tiling — DATA-DRIVEN per component from graph.json + profile.json
        self._component_tiling: Dict[str, TilingEngine] = {}
        # Cache of the auto-detected per-component TilingEngines (each parsed
        # from that component's STATIC graph.json/profile.json). Built once and
        # reused across warm-path requests so execute() does not re-read and
        # re-parse those files from disk on every call (P-COLD-WARM-PATH #1);
        # the parsed parameters are input-independent, only should_tile() /
        # tiled_execute() are per-request. None = not built yet.
        self._tiling_autodetect_cache: Optional[Dict[str, TilingEngine]] = None

        # Setup flag — allows InferenceEngine to call setup() once, then execute() many times
        self._is_setup = False

        # Persistent mode: when True, FlowContext propagates _persistent to GraphExecutors
        # so cleanup() preserves weights in VRAM. Set by serving layer for warm strategies.
        self._persistent_mode = False

    def _get_loop_id(self) -> str:
        """Get loop_id from variables.json contract. ZERO HARDCODE."""
        for var_name, var_def in self.pkg.variables.items():
            if not isinstance(var_def, dict):
                continue
            if var_def.get("role") == "loop_index":
                loop_id = var_def.get("resolver", {}).get("loop_id")
                if loop_id:
                    return loop_id
        return "default_loop"

    def _detect_flow_type(self) -> str:
        """DATA-DRIVEN flow type detection from topology.flow.type."""
        flow = self.pkg.topology.get("flow", {})
        if flow and "type" in flow:
            flow_type = flow["type"]
            # Legacy compatibility: forward_pass + generation.type override
            gen_type = flow.get("generation", {}).get("type", "")
            if gen_type == "encoder_decoder_audio":
                return "audio"
            # Modality-driven selection for multimodal understanding
            # builds (Qwen3-Omni lineage): a vlm-typed topology whose
            # generation block names a REAL component serves both
            # image+text (vlm flow) and pure-text requests — the latter
            # on the AR flow (KV-cached, the Stage-1-proven path) with
            # empty visual stubs. Request datum: the preprocessed image
            # input's presence. Topologies whose generation block names
            # a non-component (GLM's legacy 'language_model' alias) keep
            # the plain vlm routing — behavior unchanged.
            if flow_type == "vlm" and self.variable_resolver is not None:
                _gen_lm = flow.get("generation", {}).get("lm_component")
                _img_var = (flow.get("vlm", {}).get("input", {})
                            .get("image_variable", "global.pixel_values"))
                _resolved = self.variable_resolver.resolved
                _has_modal = (
                    _resolved.get(_img_var) is not None
                    or _resolved.get("global.pixel_values_videos") is not None
                    or _resolved.get("global.audio_path") is not None)
                if (_gen_lm in (self.pkg.topology.get("components") or {})
                        and not _has_modal):
                    return "autoregressive_generation"
            return flow_type

        raise RuntimeError(
            "ZERO FALLBACK: Cannot detect flow_type from topology.\n"
            "Expected: topology.flow.type\n"
            "Model data may use old schema. Re-trace and re-build."
        )

    def _index_connections(self) -> Dict[str, Dict[str, List[str]]]:
        """Index topology.json connections by target component."""
        index: Dict[str, Dict[str, List[str]]] = {}
        connections = self.pkg.topology.get("connections", [])

        # Known component names for multi-dot resolution (e.g., "model.encoder")
        known_components = set(self.pkg.topology.get("components", {}).keys())

        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")

            if not to_port or "." not in to_port:
                continue

            # Match against known component names to handle dots in names
            to_comp, to_input = self._split_port(to_port, known_components)

            if to_comp not in index:
                index[to_comp] = {}
            if to_input not in index[to_comp]:
                index[to_comp][to_input] = []

            index[to_comp][to_input].append(from_port)

        # Forward pass inference for models without connections
        # Skip for flow types that manage their own data routing
        if not connections:
            flow = self.pkg.topology.get("flow", {})
            flow_type = flow.get("type", "")
            order = flow.get("order", [])
            if order and flow_type not in ("tts_llm",):
                for i in range(len(order) - 1):
                    from_comp = order[i]
                    to_comp = order[i + 1]
                    if to_comp not in index:
                        index[to_comp] = {}
                    index[to_comp]["input"] = [f"{from_comp}.output_0"]
                logger.debug(f"Inferred {len(order) - 1} connections from flow.order")

        # Sort sources for state variables
        loop_def = self.pkg.topology.get("flow", {}).get("loop", {})
        state_variable = loop_def.get("state_variable", "")
        state_input = loop_def.get("state_input", "")

        for comp_name, inputs in index.items():
            for input_name, sources in inputs.items():
                if input_name == state_input and state_variable in sources:
                    index[comp_name][input_name] = sorted(
                        sources, key=lambda s: (0 if s == state_variable else 1, s)
                    )
                else:
                    index[comp_name][input_name] = sorted(
                        sources, key=lambda s: (1 if s.startswith("global.") else 0, s)
                    )

        return index

    @staticmethod
    def _split_port(port: str, known_components: set):
        """Split a connection port into (component_name, input/output_name).

        Handles multi-dot component names (e.g., "model.encoder.output_0")
        by matching against known component names longest-first.
        """
        # Try known components longest-first
        for comp in sorted(known_components, key=len, reverse=True):
            if port.startswith(comp + "."):
                return comp, port[len(comp) + 1:]
        # Fallback: split on first dot
        parts = port.split(".", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def setup(self) -> None:
        """
        One-time setup: modules, executors, strategy. Idempotent.

        Called explicitly by InferenceEngine (serving layer) for warm reuse,
        or implicitly by execute() for cold-start compatibility.
        """
        if self._is_setup:
            return
        self._optimize_cpu_threading()
        self._setup_modules()
        self._setup_executors()
        self._init_strategy()
        self._is_setup = True

    def _optimize_cpu_threading(self) -> None:
        """Auto-configure CPU thread count for optimal performance.

        When running on CPU (no GPU or zero3 offload), thread count
        matters significantly. Set OMP_NUM_THREADS to physical cores
        if not already set by the user.
        """
        import os
        # Only set if user hasn't explicitly configured
        if os.environ.get("OMP_NUM_THREADS"):
            return

        import torch
        # Check if we're running CPU-only or zero3
        strategy = getattr(self.plan, 'strategy', '') if self.plan else ''
        is_cpu_mode = strategy == "zero3" or not torch.cuda.is_available()

        if is_cpu_mode:
            physical_cores = os.cpu_count()
            if physical_cores:
                # Use physical cores (not hyperthreads) for compute
                # Hyperthreads hurt BLAS performance
                optimal = max(1, physical_cores // 2)
                os.environ["OMP_NUM_THREADS"] = str(optimal)
                torch.set_num_threads(optimal)

                # Check for BLAS library
                blas_info = torch.__config__.show()
                has_mkl = "mkl" in blas_info.lower()
                has_openblas = "openblas" in blas_info.lower()
                if not has_mkl and not has_openblas:
                    import logging
                    logging.getLogger(__name__).warning(
                        "CPU mode: No MKL or OpenBLAS detected. "
                        "Install torch with MKL for 2-4x faster CPU inference."
                    )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution entry point.

        1. Setup modules & executors (idempotent — skipped if already done)
        2. Initialize variables and helpers
        3. Dispatch to appropriate flow handler
        """
        # 1. Setup Phase (no-op if already called by serving layer)
        self.setup()

        # 2. Variable Resolution Phase
        merged_defaults = self._prepare_defaults(inputs)
        self._init_variable_resolver(inputs, merged_defaults)
        self._set_runtime_resolution_on_executors(merged_defaults)

        # 3. Initialize helper modules
        self._init_helpers()

        # 4. Create FlowContext and dispatch to handler
        flow_type = self._detect_flow_type()
        logger.debug(f"Flow type: {flow_type}")

        assert self.variable_resolver is not None, "variable_resolver must be initialized"
        assert self.strategy is not None, "strategy must be initialized"

        ctx = FlowContext(
            pkg=self.pkg,
            plan=self.plan,
            variable_resolver=self.variable_resolver,
            executors=self.executors,
            modules=self.modules,
            strategy=self.strategy,
            connections_index=self._connections_index,
            loop_id=self._loop_id,
            nbx_path_str=self._nbx_path_str,
            mode=self.mode,
            primary_device=self._get_primary_device(),
            persistent_mode=self._persistent_mode,
        )

        # Get and execute flow handler
        handler = self._create_flow_handler(flow_type, ctx)
        return handler.execute()

    def _prepare_defaults(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare merged defaults from family config, pkg defaults, and user inputs."""
        # ZERO FALLBACK: family MUST be in manifest
        family = self.pkg.manifest.get("family")
        if family is None:
            raise RuntimeError(
                f"ZERO FALLBACK: 'family' missing in manifest.\n"
                f"Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )
        family_defaults = get_family_defaults(family)

        # Cascade generation-type-specific defaults (e.g. image.yml diffusion.defaults)
        # Detect generation type from manifest or topology flow type
        generation_type = self.pkg.manifest.get("generation_type")
        if generation_type is None:
            flow_type = self.pkg.topology.get("flow", {}).get("type", "")
            if flow_type == "iterative_process":
                generation_type = "diffusion"
            elif flow_type == "autoregressive":
                generation_type = "autoregressive"

        merged_defaults = dict(family_defaults)
        if generation_type:
            try:
                family_cfg = get_family_config(family)
                gen_defaults = family_cfg.get(generation_type, {}).get("defaults", {})
                merged_defaults.update(gen_defaults)
            except FileNotFoundError:
                pass
        merged_defaults.update(self.pkg.defaults)

        for key, value in inputs.items():
            default_key = key.replace("global.", "") if key.startswith("global.") else key
            merged_defaults[default_key] = value

        # Compute dynamic latent dimensions
        comp_configs = {name: data for name, data in self.pkg.components.items()}
        merged_defaults = self._inject_dynamic_latent_dimensions(merged_defaults, comp_configs)

        return merged_defaults

    def _init_variable_resolver(self, inputs: Dict[str, Any], merged_defaults: Dict[str, Any]) -> None:
        """Initialize VariableResolver with all context."""
        comp_configs = {name: data for name, data in self.pkg.components.items()}

        self.variable_resolver = VariableResolver(
            variables_contract=self.pkg.variables,
            runtime_defaults=merged_defaults,
            component_configs=comp_configs,
            modules=self.modules,
            loop_state={},
            device=self._get_primary_device()
        )

        # Inject user inputs
        for name, value in inputs.items():
            self.variable_resolver.set(name, value)

        # Inject generation-type defaults as global.* variables (fallback if not user-set)
        # e.g. num_inference_steps from image.yml diffusion.defaults → global.num_inference_steps
        _global_default_keys = ("num_inference_steps", "guidance_scale", "height", "width",
                                "temperature", "top_p", "top_k", "cfg_weight", "negative_prompt")
        for key in _global_default_keys:
            global_key = f"global.{key}"
            if global_key not in self.variable_resolver.resolved and key in merged_defaults:
                self.variable_resolver.set(global_key, merged_defaults[key])

        # Register component handlers
        for comp_name, executor in self.executors.items():
            self.variable_resolver.register_module_handler(
                comp_name, lambda args, e=executor: e.run(args)
            )

    def _init_helpers(self) -> None:
        """Initialize helper modules (resolver, synthesizer, extractor, cfg)."""
        assert self.variable_resolver is not None, "variable_resolver must be initialized before _init_helpers"

        self._input_resolver = InputResolver(
            variable_resolver=self.variable_resolver,
            connections_index=self._connections_index,
            topology=self.pkg.topology,
            loop_id=self._loop_id,
            executors=self.executors,
        )

        self._input_synthesizer = InputSynthesizer(
            topology=self.pkg.topology,
            variable_resolver=self.variable_resolver,
            plan=self.plan,
            modules=self.modules,
            executors=self.executors
        )

        self._output_extractor = OutputExtractor(
            topology=self.pkg.topology,
            variable_resolver=self.variable_resolver
        )

        # Universal tiling — DATA-DRIVEN per component. The auto-detect reads
        # each component's static graph.json/profile.json from disk; parse them
        # ONCE and cache the resulting engines so the warm serving path does not
        # re-read/re-parse every component per request (P-COLD-WARM-PATH #1 —
        # ~100ms-to-seconds/request for nothing). The Prism-plan override and
        # op-level registration below run per request (in-memory plan, cheap)
        # and keep their prior semantics untouched.
        if self._tiling_autodetect_cache is None:
            cache: Dict[str, TilingEngine] = {}
            components_dir = self.pkg.cache_path / "components"
            for comp_name in self.pkg.topology.get("components", {}):
                try:
                    engine = TilingEngine.from_component_config(
                        graph_path=components_dir / comp_name / "graph.json",
                        profile_path=components_dir / comp_name / "profile.json",
                    )
                    if engine is not None:
                        cache[comp_name] = engine
                except (FileNotFoundError, ValueError) as e:
                    logger.debug(f"Tiling not available for {comp_name}: {e}")
            self._tiling_autodetect_cache = cache
        self._component_tiling = dict(self._tiling_autodetect_cache)

        # Prism-driven component-level tiling — overrides the auto-detect above.
        # Emitted when Prism kept a spatial-overflow component (e.g. a high-res
        # video VAE) on GPU by tiling its decode instead of host offload
        # (solver Strategy 3.5). The auto-detect from_component_config declines
        # symbolic / spatial-adaptive graphs, so this is how a symbolic VAE
        # gets a TilingEngine with the Prism-chosen tile_size.
        for comp_name, spec in (getattr(self.plan, 'component_tiling', None) or {}).items():
            self._component_tiling[comp_name] = TilingEngine(
                trace_size=spec["trace_size"],
                overlap=spec["overlap"],
                scale_factor=spec["scale_factor"],
                window_alignment=spec.get("window_alignment", 1),
                tile_size=spec["tile_size"],
                t_tile=spec.get("t_tile"),
                t_overlap=spec.get("t_overlap", 0),
                t_scale=spec.get("t_scale", 1),
                # D2-ENCODER: downscale direction (VAE encoder) — absent
                # from decode specs, so defaults keep them byte-identical.
                downscale=spec.get("downscale", False),
                t_axis_out=spec.get("t_axis_out", 2),
            )
            logger.info(
                f"[TilingEngine] {comp_name}: Prism component-tiling "
                f"tile_size={spec['tile_size']} scale={spec['scale_factor']}"
                + (f" t_tile={spec['t_tile']} t_scale={spec['t_scale']}"
                   if spec.get("t_tile") else "")
                + (" direction=encode(downscale)"
                   if spec.get("downscale") else "")
                + " (kept on GPU instead of host offload)"
            )

        # Op-level tiling — wires per-op_uid interceptors on the component's
        # GraphExecutor so upsample→conv fusion pairs stream band-by-band
        # without materializing the OOM intermediate. Plan emitted by Prism
        # when single-op overflow is detected (e.g. Sana 4Kpx VAE).
        op_tiling_plans = getattr(self.plan, 'runtime_op_tiling', None) or {}
        if op_tiling_plans:
            from neurobrix.core.module.tiling_engine import OpLevelTilingEngine
            for comp_name, op_plan in op_tiling_plans.items():
                comp_executor = self.executors.get(comp_name)
                if comp_executor is None:
                    logger.debug(f"Op-level tiling skipped: no executor for '{comp_name}'")
                    continue
                graph_exec = getattr(comp_executor, 'graph_executor', None) or comp_executor
                if not hasattr(graph_exec, 'register_op_uid_interceptors'):
                    logger.warning(
                        f"Op-level tiling skipped for '{comp_name}': "
                        f"graph executor missing register_op_uid_interceptors"
                    )
                    continue
                engine = OpLevelTilingEngine.from_op_level_constraint(op_plan)
                if engine is None:
                    continue
                count = engine.register_into_graph_executor(graph_exec)
                logger.info(
                    f"[OpLevelTiling] {comp_name}: registered {count} op_uid "
                    f"interceptors ({len(op_plan.fusion_pairs)} fusion pairs)"
                )

        # Note: CFGEngine is created in _create_flow_handler where FlowContext is available

    def _create_flow_handler(self, flow_type: str, ctx: FlowContext):
        """Create appropriate flow handler based on flow type."""
        assert self._output_extractor is not None, "output_extractor must be initialized"
        assert self._input_resolver is not None, "input_resolver must be initialized"
        assert self._input_synthesizer is not None, "input_synthesizer must be initialized"

        if flow_type == "iterative_process":
            if ctx.mode in ("triton", "triton_sequential"):
                # Triton mode: use TritonCFGEngine (zero torch dependency)
                from neurobrix.triton.cfg.engine import TritonCFGEngine
                cfg_engine = TritonCFGEngine.from_topology(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    extract_primary_output_fn=self._output_extractor.extract_primary_output
                )
                from neurobrix.triton.flow.iterative_process import TritonIterativeProcessHandler
                return TritonIterativeProcessHandler(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    extract_primary_output_fn=self._output_extractor.extract_primary_output,
                    cfg_engine=cfg_engine,
                    output_extractor=self._output_extractor
                )
            # Native mode: use CFGEngine (torch-based)
            cfg_engine = CFGEngine.from_topology(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                extract_primary_output_fn=self._output_extractor.extract_primary_output
            )
            from neurobrix.core.flow.iterative_process import IterativeProcessHandler
            return IterativeProcessHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                extract_primary_output_fn=self._output_extractor.extract_primary_output,
                cfg_engine=cfg_engine,
                output_extractor=self._output_extractor
            )
        elif flow_type == "static_graph":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.static_graph import TritonStaticGraphHandler
                return TritonStaticGraphHandler(
                    ctx=ctx,
                    execute_component_fn=self._execute_component
                )
            from neurobrix.core.flow.static_graph import StaticGraphHandler
            return StaticGraphHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component
            )
        elif flow_type == "forward_pass":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.forward_pass import TritonForwardPassHandler
                return TritonForwardPassHandler(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights
                )
            from neurobrix.core.flow.forward_pass import ForwardPassHandler
            return ForwardPassHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights
            )
        elif flow_type == "autoregressive_generation":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.autoregressive import TritonAutoregressiveHandler
                return TritonAutoregressiveHandler(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                    input_resolver=self._input_resolver,
                    input_synthesizer=self._input_synthesizer,
                    output_extractor=self._output_extractor
                )
            from neurobrix.core.flow.autoregressive import AutoregressiveHandler
            return AutoregressiveHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
                input_resolver=self._input_resolver,
                input_synthesizer=self._input_synthesizer,
                output_extractor=self._output_extractor
            )
        elif flow_type == "audio":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.audio import TritonAudioEngine
                return TritonAudioEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.audio import AudioEngine
            return AudioEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "rnnt":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.rnnt import TritonRNNTEngine
                return TritonRNNTEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.rnnt import RNNTEngine
            return RNNTEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "encoder_decoder":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.encoder_decoder import TritonEncoderDecoderEngine
                return TritonEncoderDecoderEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.encoder_decoder import EncoderDecoderEngine
            return EncoderDecoderEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "audio_llm":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.audio_llm import TritonAudioLLMEngine
                return TritonAudioLLMEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.audio_llm import AudioLLMEngine
            return AudioLLMEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "vlm":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.vlm import TritonVLMEngine
                return TritonVLMEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.vlm import VLMEngine
            return VLMEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "dual_ar":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.dual_ar import TritonDualAREngine
                return TritonDualAREngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.dual_ar import DualAREngine
            return DualAREngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "tts_llm":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.tts_llm import TritonTTSLLMEngine
                return TritonTTSLLMEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.tts_llm import TTSLLMEngine
            return TTSLLMEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "next_token_diffusion":
            if ctx.mode in ("triton", "triton_sequential"):
                from neurobrix.triton.flow.next_token_diffusion import TritonNextTokenDiffusionEngine
                return TritonNextTokenDiffusionEngine(
                    ctx=ctx,
                    execute_component_fn=self._execute_component,
                    resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                    ensure_weights_fn=self._ensure_weights_loaded,
                    unload_weights_fn=self._unload_component_weights,
                )
            from neurobrix.core.flow.next_token_diffusion import NextTokenDiffusionEngine
            return NextTokenDiffusionEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )

        raise RuntimeError(
            f"ZERO FALLBACK: Unsupported flow type '{flow_type}'.\n"
            f"Supported: iterative_process, static_graph, forward_pass, "
            f"autoregressive_generation, audio, rnnt, encoder_decoder, audio_llm, "
            f"dual_ar, tts_llm, next_token_diffusion"
        )

    # ========== SETUP METHODS ==========

    def _setup_modules(self) -> None:
        """Instantiate auxiliary modules from pkg.modules."""
        for mod_name, mod_data in self.pkg.modules.items():
            mod_type = mod_data.get("type")
            config = mod_data.get("config", {})

            if mod_type == "scheduler":
                # Two totally separate scheduler implementations; the orchestrator
                # (this shared entry point) picks by mode. Triton gets the
                # zero-torch NBXTensor scheduler; PyTorch gets the torch one.
                if self.mode in ("triton", "triton_sequential"):
                    from neurobrix.triton.scheduler.factory import TritonSchedulerFactory
                    self.modules[mod_name] = TritonSchedulerFactory.create(config)
                else:
                    self.modules[mod_name] = SchedulerFactory.create(config)
            elif mod_type == "tokenizer":
                self._setup_tokenizer(mod_name, mod_data)
            else:
                self.modules[mod_name] = config

    def _setup_tokenizer(self, mod_name: str, mod_data: Dict) -> None:
        """Setup tokenizer module with max_length detection."""
        module_path = mod_data.get("path", f"modules/{mod_name}").rstrip("/")
        max_length = self._detect_tokenizer_max_length(mod_name)

        if max_length is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Cannot determine tokenizer max_length for '{mod_name}'.\n"
                "topology.json must contain shapes.input_ids for text component."
            )

        self.modules[mod_name] = self._load_tokenizer(module_path, max_length)

    def _detect_tokenizer_max_length(self, mod_name: str) -> Optional[int]:
        """Detect tokenizer max_length from topology or config."""
        topology_components = self.pkg.topology.get("components", {})

        # Map tokenizer to corresponding encoder
        if mod_name == "tokenizer":
            corresponding_encoder = "text_encoder"
        elif mod_name.startswith("tokenizer_"):
            suffix = mod_name[len("tokenizer_"):]
            corresponding_encoder = f"text_encoder_{suffix}"
        else:
            corresponding_encoder = None

        # Try from corresponding encoder
        if corresponding_encoder and corresponding_encoder in topology_components:
            comp_info = topology_components[corresponding_encoder]
            shapes = comp_info.get("shapes", {})
            if "input_ids" in shapes:
                max_length = shapes["input_ids"][1]
                return max_length

        # Fallback: search all components
        for comp_name, comp_info in topology_components.items():
            shapes = comp_info.get("shapes", {})
            if "input_ids" in shapes:
                max_length = shapes["input_ids"][1]
                return max_length

        # Fallback: tokenizer_config.json
        cache_path = ensure_extracted(Path(self._nbx_path_str))
        module_path = self.pkg.modules.get(mod_name, {}).get("path", f"modules/{mod_name}")
        tokenizer_config_path = cache_path / module_path / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            try:
                with open(tokenizer_config_path) as f:
                    tokenizer_config = json.load(f)
                max_length = tokenizer_config.get("model_max_length")
                if max_length is not None:
                    return max_length
            except Exception:
                pass

        # Fallback: defaults.json max_tokens (for audio/TTS models without text_encoder)
        max_tokens = self.pkg.defaults.get("max_tokens")
        if max_tokens is not None:
            return max_tokens

        # Fallback: max_position_embeddings from topology
        for comp_name, comp_info in topology_components.items():
            ev = comp_info.get("extracted_values", {})
            mpe = ev.get("max_position_embeddings")
            if mpe is not None:
                return mpe

        # Final fallback: reasonable default for TTS/audio models
        flow_type = self.pkg.topology.get("flow", {}).get("type", "")
        if flow_type in ("audio", "rnnt", "encoder_decoder", "audio_llm", "dual_ar", "tts_llm"):
            return 2048

        # ZERO FALLBACK: Crash if we cannot determine max_length
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine max_length for tokenizer '{mod_name}'.\n"
            "Searched:\n"
            "  1. topology.components.{encoder}.shapes.input_ids[1]\n"
            "  2. modules/{mod_name}/tokenizer_config.json model_max_length\n"
            "  3. defaults.json max_tokens\n"
            "  4. topology.components.*.extracted_values.max_position_embeddings\n"
            "All sources returned None."
        )

    def _setup_executors(self) -> None:
        """Instantiate Neural Executors via Factory."""
        topology_components = self.pkg.topology.get("components", {})

        # Non-neural component types that never need executors
        NON_NEURAL_TYPES = {"module", "scheduler"}

        for comp_name, comp_info in topology_components.items():
            comp_type = comp_info.get("type", "")
            if comp_type in NON_NEURAL_TYPES:
                continue

            if comp_name in self.modules:
                continue

            allocation = self._get_allocation(comp_name)
            if allocation is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: No allocation for component '{comp_name}'.\n"
                    f"Prism plan must provide allocation for all neural components."
                )

            dag = self._load_graph(comp_name)
            executor = ExecutorFactory.create(
                component=comp_name,
                allocation=allocation,
                nbx_path=self._nbx_path_str,
                dag=dag,
                mode=self.mode
            )
            self.executors[comp_name] = executor

    def _get_allocation(self, comp_name: str) -> Optional[Any]:
        """Get allocation from Prism plan."""
        if hasattr(self.plan, 'get_allocation'):
            return self.plan.get_allocation(comp_name)
        elif hasattr(self.plan, 'components'):
            return self.plan.components.get(comp_name)
        return None

    def _init_strategy(self) -> None:
        """Initialize execution strategy from Prism's allocation decision."""
        allocations = {}
        for comp_name in self.executors:
            alloc = self._get_allocation(comp_name)
            if alloc is not None:
                if hasattr(alloc, 'device'):
                    device = str(alloc.device)
                    # FIX: Use correct field name 'shard_map' (not 'shard_allocations')
                    shard_map = getattr(alloc, 'shard_map', {})
                    # FIX: Propagate dtype from allocation
                    dtype = getattr(alloc, 'dtype', None)
                    # FIX: Get strategy from allocation
                    strategy = getattr(alloc, 'strategy', 'single_gpu')
                elif isinstance(alloc, tuple) and len(alloc) >= 2:
                    device, shard_map = alloc[0], alloc[1]
                    dtype = alloc[2] if len(alloc) > 2 else None
                    strategy = alloc[3] if len(alloc) > 3 else 'single_gpu'
                else:
                    device = str(alloc)
                    shard_map = {}
                    dtype = None
                    strategy = 'single_gpu'
                # Store all allocation info including dtype and strategy
                allocations[comp_name] = {
                    'device': device,
                    'shard_map': shard_map,
                    'dtype': dtype,
                    'strategy': strategy,
                }

        # Get loading_mode from Prism plan (DATA-DRIVEN)
        loading_mode = getattr(self.plan, 'loading_mode', 'lazy')

        context = StrategyContext(
            strategy_name=self._strategy_name,
            allocations=allocations,
            component_executors=self.executors,
            variable_resolver=self.variable_resolver,
            topology=self.pkg.topology,
            runtime_package=self.pkg,
            loading_mode=loading_mode,
            mode=self.mode,
        )

        self.strategy = get_strategy(self._strategy_name, context)
        logger.debug(f"Strategy: {self._strategy_name}")

    # ========== COMPONENT EXECUTION ==========

    def _execute_component(
        self,
        comp_name: str,
        phase: str = "default",
        loop_timestep: Optional[torch.Tensor] = None
    ) -> Optional[Any]:
        """Execute a single component. ZERO HARDCODE."""
        assert self._input_resolver is not None, "input_resolver must be initialized"
        assert self._input_synthesizer is not None, "input_synthesizer must be initialized"
        assert self._output_extractor is not None, "output_extractor must be initialized"

        if comp_name not in self.executors:
            raise RuntimeError(
                f"ZERO FALLBACK: Component '{comp_name}' not in executors.\n"
                f"Available: {list(self.executors.keys())}"
            )

        # Update loop state if timestep provided
        if loop_timestep is not None and self.variable_resolver:
            if hasattr(loop_timestep, 'unsqueeze') and loop_timestep.dim() == 0:
                loop_timestep = loop_timestep.unsqueeze(0)
            self.variable_resolver.loop_state[self._loop_id] = loop_timestep

        # Lazy load weights
        self._ensure_weights_loaded(comp_name)
        # Get component handler (DATA-DRIVEN component-specific behavior)
        executor = self.executors[comp_name]
        handler = getattr(executor, '_component_handler', None)

        # Resolve, synthesize, and transform inputs
        comp_inputs = self._input_resolver.resolve_component_inputs(comp_name)

        # Apply input transformations for post_loop phase
        if phase == "post_loop":
            # Step 1: Replace inputs with state variable (always needed for post_loop VAE)
            comp_inputs = self._replace_with_state_variable(comp_name, comp_inputs)

            # Step 2: Apply handler-specific transformations (e.g., scaling_factor)
            if handler is not None:
                comp_inputs = handler.transform_inputs(comp_inputs, phase)

        comp_inputs = self._input_synthesizer.remap_inputs_to_graph(comp_name, comp_inputs)
        comp_inputs = self._input_synthesizer.synthesize_missing_inputs(comp_name, comp_inputs)
        comp_inputs = self._input_synthesizer.apply_shape_transforms(comp_name, comp_inputs)

        # Execute: Check for tiling first, then TP, then normal
        output = None

        # UNIVERSAL TILING: Any component with a TilingEngine gets tiled
        # when its spatial input exceeds trace size
        tiling = self._component_tiling.get(comp_name)
        if output is None and tiling is not None:
            spatial_input = self._find_spatial_input(comp_inputs)
            if spatial_input is not None and tiling.should_tile(spatial_input):
                input_name = next(iter(comp_inputs.keys()))

                def execute_tile(tile):
                    result = executor.run({input_name: tile})
                    if isinstance(result, dict):
                        result = next(iter(result.values()))
                    return result

                output = tiling.tiled_execute(spatial_input, execute_tile)

        # Standard execution if tiling didn't happen
        if output is None:
            if self._is_tp_component(comp_name):
                assert self.strategy is not None, "strategy must be initialized for TP components"
                output = self.strategy.execute_component(comp_name, phase, comp_inputs)
            elif self._is_zero3_component(comp_name) and self.strategy is not None:
                # Zero3 components: delegate to strategy for pinned memory + GPU transfer
                output = self.strategy.execute_component(comp_name, phase, comp_inputs)
            elif self._is_hybrid_strategy() and self.strategy is not None:
                # Hybrid placement strategies (lazy_sequential with mixed
                # CPU/GPU, cpu_execution, any strategy where different
                # components live on different devices) MUST go through
                # `strategy.execute_component` so the strategy can transfer
                # inputs to the destination component's device. Otherwise
                # inputs produced on cuda:0 by one component would arrive
                # on the consumer's CPU dispatcher unchanged and ATen would
                # either hang in implicit transfer or raise a device-
                # mismatch error. P-RUNTIME-HYBRID-DEVICE-DISPATCH
                # 2026-05-12.
                output = self.strategy.execute_component(comp_name, phase, comp_inputs)
            else:
                output = executor.run(comp_inputs)

        # Store outputs
        self._output_extractor.store_component_outputs(comp_name, output)
        return output

    def _is_tp_component(self, comp_name: str) -> bool:
        """Check if component uses Tensor Parallel allocation."""
        if hasattr(self.plan, 'components'):
            alloc = self.plan.components.get(comp_name)
            if alloc:
                device = getattr(alloc, 'device', '')
                return device.startswith('tp:')

        if hasattr(self.plan, 'allocations'):
            for alloc in self.plan.allocations:
                if alloc.name == comp_name:
                    device = getattr(alloc, 'device', '')
                    return device.startswith('tp:')

        return False

    def _is_zero3_component(self, comp_name: str) -> bool:
        """Check if component has zero3 strategy (plan-level or per-component)."""
        # Plan-level zero3
        if self._strategy_name == "zero3":
            return True
        # Per-component zero3 (within lazy_sequential)
        if hasattr(self.plan, 'components'):
            alloc = self.plan.components.get(comp_name)
            if alloc:
                return getattr(alloc, 'strategy', '') == 'zero3'
        return False

    def _is_hybrid_strategy(self) -> bool:
        """Check whether the active strategy needs per-component device
        transfer of inputs (lazy_sequential with hybrid CPU+GPU
        placement, cpu_execution, etc.). Strategies where every
        component lives on the same device (single_gpu, component
        placement on a single GPU class, ...) do not need this — the
        producer's output is already on the consumer's device.

        Heuristic: a strategy is "hybrid" if it can route different
        components to devices with different families (`cpu` vs
        `cuda:*`). `lazy_sequential` is the canonical case after
        `_place_component` Strategy 4 (CPU fallback) lands. Also
        triggers for `cpu_execution` so its `prepare_inputs` runs
        consistently — host-resident producer outputs are still
        torch.Tensors that need explicit `.to('cpu')` if they
        crossed a graph executor boundary.

        P-RUNTIME-HYBRID-DEVICE-DISPATCH 2026-05-12.
        """
        if self._strategy_name in ("lazy_sequential", "cpu_execution"):
            return True
        # Generic detection: if allocations route any component to "cpu"
        # AND any other to "cuda:*" / "hip:*" / "xpu:*", we're hybrid.
        if hasattr(self.plan, 'components'):
            seen_cpu = False
            seen_gpu = False
            for alloc in self.plan.components.values():
                dev = getattr(alloc, 'device', '')
                if dev == 'cpu' or dev.startswith('cpu'):
                    seen_cpu = True
                elif dev.startswith(('cuda', 'hip', 'xpu')):
                    seen_gpu = True
            if seen_cpu and seen_gpu:
                return True
        return False

    # ========== TILING HELPERS ==========

    def _find_spatial_input(self, comp_inputs: Dict[str, Any]) -> Optional[Any]:
        """Find the first spatial tensor in component inputs for tiling.

        Accepts 4D images [B, C, H, W] and 5D video [B, C, T, H, W] — the
        TilingEngine tiles the trailing (H, W) pair for both. Returns either
        torch.Tensor or NBXTensor; both expose .dim() and are usable
        downstream by the tiling engine.
        """
        for value in comp_inputs.values():
            if _is_tensor(value) and value.dim() in (4, 5):
                return value
        return None

    # ========== WEIGHT MANAGEMENT ==========

    def _ensure_weights_loaded(self, comp_name: str) -> None:
        """Lazily load weights for a component if not already loaded."""
        executor = self.executors.get(comp_name)
        if executor is None:
            return

        if getattr(executor, '_weights_loaded', False):
            return

        params = getattr(executor, '_weight_loading_params', None)
        if params is None:
            return

        nbx_path = params["nbx_path"]
        component = params["component"]
        shard_map = params.get("shard_map", {})

        if shard_map:
            executor.load_weights(nbx_path, component, shard_map)
        else:
            executor.load_weights(nbx_path, component)

        executor._weights_loaded = True

        # Zero3 needs to install its per-executor hooks (pipeline callback
        # + post-run teardown) BEFORE any executor.run() call. Flow
        # handlers that bypass strategy.execute_component (autoregressive
        # LLM prefill) still funnel through here for weight loading, so
        # this is the natural install point. Other strategies ignore —
        # the method is zero3-specific by design.
        if self._is_zero3_component(comp_name) and self.strategy is not None:
            install_fn = getattr(self.strategy, 'install_for_executor', None)
            if install_fn is not None:
                install_fn(comp_name, executor)

    def _unload_component_weights(self, comp_name: str) -> None:
        """Unload weights for a component to free GPU memory."""
        executor = self.executors.get(comp_name)
        if executor is None:
            return

        if not getattr(executor, '_weights_loaded', False):
            return

        if hasattr(executor, 'unload_weights'):
            executor.unload_weights()
            executor._weights_loaded = False

    # ========== UTILITY METHODS ==========

    def _get_primary_device(self) -> str:
        """Returns the primary device. ZERO FALLBACK."""
        if self.executors:
            first_exec = next(iter(self.executors.values()))
            if hasattr(first_exec, "device"):
                return str(first_exec.device)

        if hasattr(self.plan, 'primary_device'):
            return str(self.plan.primary_device)

        raise RuntimeError(
            "ZERO FALLBACK: Cannot determine primary device.\n"
            "No executors instantiated and execution_plan has no 'primary_device'."
        )

    def _load_graph(self, component_name: str) -> Dict[str, Any]:
        """Load graph.json for a component from cache."""
        cache_path = ensure_extracted(Path(self._nbx_path_str))
        graph_path = cache_path / "components" / component_name / "graph.json"

        if not graph_path.exists():
            raise RuntimeError(
                f"ZERO FALLBACK: graph.json not found for '{component_name}'.\n"
                f"Expected path: {graph_path}\n"
                f"Model graph invalid. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

        with open(graph_path) as f:
            return json.load(f)

    def _load_tokenizer(self, module_path: str, max_length: int):
        """Load tokenizer from extracted NBX cache."""
        from neurobrix.core.module.tokenizer.sp_tokenizer import load_tokenizer_from_path

        cache_path = ensure_extracted(Path(self._nbx_path_str))
        tokenizer_dir = cache_path / module_path

        if not tokenizer_dir.exists():
            raise RuntimeError(
                f"ZERO FALLBACK: Tokenizer directory not found.\n"
                f"Expected: {tokenizer_dir}\n"
                f"NBX path: {self._nbx_path_str}"
            )

        return load_tokenizer_from_path(tokenizer_dir, max_length)

    def _set_runtime_resolution_on_executors(self, merged_defaults: Dict[str, Any]) -> None:
        """Set runtime resolution on all executors for pos_embed scaling."""
        # Autoregressive models don't use resolution-dependent pos_embed
        flow_type = self.pkg.topology.get("flow", {}).get("type", "")
        if flow_type == "autoregressive_generation":
            return

        height = merged_defaults.get("height")
        width = merged_defaults.get("width")

        if height is None or width is None:
            return

        height, width = int(height), int(width)
        logger.debug(f"Setting runtime resolution {height}x{width} on all executors")

        for comp_name, executor in self.executors.items():
            if hasattr(executor, 'set_runtime_resolution'):
                executor.set_runtime_resolution(height, width)

    def _inject_dynamic_latent_dimensions(
        self,
        merged_defaults: Dict[str, Any],
        comp_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute and inject dynamic latent dimensions from runtime resolution."""
        # Autoregressive models don't use VAE latent space — skip entirely
        flow_type = self.pkg.topology.get("flow", {}).get("type", "")
        if flow_type == "autoregressive_generation":
            return merged_defaults

        height = merged_defaults.get("height")
        width = merged_defaults.get("width")

        if height is None or width is None:
            return merged_defaults

        height, width = int(height), int(width)
        vae_scale_factor = self._get_vae_scale_factor(comp_configs)

        if vae_scale_factor is None:
            logger.warning("Cannot determine VAE scale factor, using trace-time latent dims")
            return merged_defaults

        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        # NOTE: Resolution binning removed. Models with symbolic shapes (like Sana with
        # Linear Attention) support dynamic resolution. The transformer computes pos_embed
        # dynamically based on runtime latent dims, not trace-time sample_size.
        #
        # If a model truly requires fixed resolution, it should be enforced at trace time
        # and validated here, not silently upscaled.

        merged_defaults["latent_height"] = latent_height
        merged_defaults["latent_width"] = latent_width

        # Video models: compute latent_frames from num_frames and temporal_compression_ratio
        num_frames = merged_defaults.get("num_frames")
        temporal_cr = merged_defaults.get("temporal_compression_ratio")
        if num_frames is not None and temporal_cr is not None:
            num_frames = int(num_frames)
            temporal_cr = int(temporal_cr)
            latent_frames = (num_frames - 1) // temporal_cr + 1
            merged_defaults["latent_frames"] = latent_frames
            logger.debug(f"Video latent frames: ({num_frames}-1)//{temporal_cr}+1 = {latent_frames}")

        logger.debug(f"Dynamic latent dims: {height}x{width} / {vae_scale_factor} = {latent_height}x{latent_width}")
        return merged_defaults

    def _get_vae_scale_factor(self, comp_configs: Dict[str, Any]) -> Optional[int]:
        """Determine VAE spatial compression factor."""
        manifest_scale = self.pkg.manifest.get("vae_scale_factor")
        if manifest_scale is not None:
            return int(manifest_scale)

        transformer_data = comp_configs.get("transformer", {})
        transformer_attrs = transformer_data.get("attributes", {})
        trace_latent_extent = transformer_attrs.get("state_extent_0")

        if trace_latent_extent:
            trace_pixel_res = self.pkg.manifest.get("trace_resolution")
            if trace_pixel_res:
                scale = int(trace_pixel_res) // int(trace_latent_extent)
                return scale

            state_channels = transformer_attrs.get("state_channels", 4)
            if state_channels >= 32:
                return 32
            return 8

        return None

    def _replace_with_state_variable(self, comp_name: str, comp_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace tensor inputs with the final state variable for post_loop components.

        For VAE decoding after the denoising loop, the input should be the final
        latent tensor from the state variable, not whatever the input_resolver returned.

        Args:
            comp_name: Component name
            comp_inputs: Resolved inputs

        Returns:
            Inputs with tensor values replaced by state variable
        """
        assert self.variable_resolver is not None, "variable_resolver must be initialized"

        loop_def = self.pkg.topology.get("flow", {}).get("loop", {})
        state_variable = loop_def.get("state_variable")

        if not state_variable:
            return comp_inputs

        final_latents = self.variable_resolver.get(state_variable)
        if final_latents is None:
            raise RuntimeError(
                f"ZERO FALLBACK: state_variable '{state_variable}' is None after loop.\n"
                "The denoising loop did not produce final latents."
            )

        # Replace all tensor inputs with the final latents
        fixed_inputs = {}
        for input_name, value in comp_inputs.items():
            if _is_tensor(value):
                fixed_inputs[input_name] = final_latents
            else:
                fixed_inputs[input_name] = value

        return fixed_inputs

    def get_final_output(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Get the final tensor output from execution outputs.

        Searches for the primary output tensor in order:
        1. vae.last_output (image generation)
        2. decoder.last_output (any decoder)
        3. First tensor in outputs dict

        Args:
            outputs: Dict from execute() - resolved variables

        Returns:
            Final tensor output, or None if not found
        """
        def _is_tensor(v):
            return isinstance(v, torch.Tensor) or hasattr(v, 'data_ptr')

        def _to_torch(v):
            if isinstance(v, torch.Tensor):
                return v
            if hasattr(v, 'data_ptr'):
                from neurobrix.kernels.nbx_tensor import nbx_to_torch
                return nbx_to_torch(v)
            return v

        # Try known output keys (order: AR image → diffusion → generic)
        for key in [
            "output_image",               # Autoregressive image (Janus, LlamaGen)
            "global.output_image",         # Fully-qualified AR image
            "vae.last_output",             # Diffusion VAE output
            "decoder.last_output",         # Generic decoder
            "final_output",                # Generic fallback
        ]:
            if key in outputs and _is_tensor(outputs[key]):
                return _to_torch(outputs[key])

        # Get post_loop component from topology
        flow = self.pkg.topology.get("flow", {})
        post_loop = flow.get("post_loop", [])
        if post_loop:
            final_comp = post_loop[-1]
            for key in [f"{final_comp}.last_output", f"{final_comp}.output_0"]:
                if key in outputs and _is_tensor(outputs[key]):
                    return _to_torch(outputs[key])

        # forward_pass flow (no loop): the final output is produced by
        # the LAST component in `flow.order`. Without this branch the
        # generic "first tensor" fallback below could return an
        # intermediate trunk output (e.g. an upscaler's feature trunk
        # at input resolution) instead of the head's upscaled image.
        # DATA-DRIVEN (R34): the component name comes from the
        # container topology, never hardcoded.
        if flow.get("type") == "forward_pass":
            order = flow.get("order", [])
            if order:
                final_comp = order[-1]
                for key in (f"{final_comp}.last_output",
                            f"{final_comp}.output_0"):
                    if key in outputs and _is_tensor(outputs[key]):
                        return _to_torch(outputs[key])

        # Fallback: find first tensor in outputs
        for key, value in outputs.items():
            if _is_tensor(value) and (value.dim() if hasattr(value, 'dim') else value.ndim) >= 3:
                return _to_torch(value)

        return None

    def get_final_component(self) -> str:
        """
        Get the name of the final component in execution.

        Returns:
            Component name (e.g., "vae")
        """
        flow = self.pkg.topology.get("flow", {})
        post_loop = flow.get("post_loop", [])
        if post_loop:
            return post_loop[-1]

        # Fallback to flow order
        flow = self.pkg.topology.get("flow", {})
        order = flow.get("order", [])
        if order:
            return order[-1]

        return "unknown"
