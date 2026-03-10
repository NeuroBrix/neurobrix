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
            mode: Execution engine mode: "compiled" (default), "native", "pytorch", or "triton".
        """
        self.pkg = runtime_package
        self.plan = execution_plan
        self.mode = mode  # "native", "pytorch", or "triton"
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
        if not connections:
            flow = self.pkg.topology.get("flow", {})
            order = flow.get("order", [])
            if order:
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
        self._setup_modules()
        self._setup_executors()
        self._init_strategy()
        self._is_setup = True

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
            loop_id=self._loop_id
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

        # Universal tiling — DATA-DRIVEN per component
        self._component_tiling = {}
        components_dir = self.pkg.cache_path / "components"
        for comp_name in self.pkg.topology.get("components", {}):
            try:
                engine = TilingEngine.from_component_config(
                    graph_path=components_dir / comp_name / "graph.json",
                    profile_path=components_dir / comp_name / "profile.json",
                )
                if engine is not None:
                    self._component_tiling[comp_name] = engine
            except (FileNotFoundError, ValueError) as e:
                logger.debug(f"Tiling not available for {comp_name}: {e}")

        # Note: CFGEngine is created in _create_flow_handler where FlowContext is available

    def _create_flow_handler(self, flow_type: str, ctx: FlowContext):
        """Create appropriate flow handler based on flow type."""
        assert self._output_extractor is not None, "output_extractor must be initialized"
        assert self._input_resolver is not None, "input_resolver must be initialized"
        assert self._input_synthesizer is not None, "input_synthesizer must be initialized"

        if flow_type == "iterative_process":
            from neurobrix.core.flow.iterative_process import IterativeProcessHandler
            # Create CFGEngine from topology (DATA-DRIVEN detection)
            cfg_engine = CFGEngine.from_topology(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                extract_primary_output_fn=self._output_extractor.extract_primary_output
            )
            return IterativeProcessHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                extract_primary_output_fn=self._output_extractor.extract_primary_output,
                cfg_engine=cfg_engine,
                output_extractor=self._output_extractor
            )
        elif flow_type == "static_graph":
            from neurobrix.core.flow.static_graph import StaticGraphHandler
            return StaticGraphHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component
            )
        elif flow_type == "forward_pass":
            from neurobrix.core.flow.forward_pass import ForwardPassHandler
            return ForwardPassHandler(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights
            )
        elif flow_type == "autoregressive_generation":
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
            from neurobrix.core.flow.audio import AudioEngine
            return AudioEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )
        elif flow_type == "rnnt":
            from neurobrix.core.flow.rnnt import RNNTEngine
            return RNNTEngine(
                ctx=ctx,
                execute_component_fn=self._execute_component,
                resolve_inputs_fn=self._input_resolver.resolve_component_inputs,
                ensure_weights_fn=self._ensure_weights_loaded,
                unload_weights_fn=self._unload_component_weights,
            )

        raise RuntimeError(
            f"ZERO FALLBACK: Unsupported flow type '{flow_type}'.\n"
            f"Supported: iterative_process, static_graph, forward_pass, "
            f"autoregressive_generation, audio, rnnt"
        )

    # ========== SETUP METHODS ==========

    def _setup_modules(self) -> None:
        """Instantiate auxiliary modules from pkg.modules."""
        for mod_name, mod_data in self.pkg.modules.items():
            mod_type = mod_data.get("type")
            config = mod_data.get("config", {})

            if mod_type == "scheduler":
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
        if flow_type in ("audio", "rnnt"):
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
        if tiling is not None:
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

    # ========== TILING HELPERS ==========

    def _find_spatial_input(self, comp_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Find the first 4D spatial tensor in component inputs for tiling."""
        for value in comp_inputs.values():
            if isinstance(value, torch.Tensor) and value.dim() == 4:
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
            if isinstance(value, torch.Tensor):
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
        # Try known output keys (order: AR image → diffusion → generic)
        for key in [
            "output_image",               # Autoregressive image (Janus, LlamaGen)
            "global.output_image",         # Fully-qualified AR image
            "vae.last_output",             # Diffusion VAE output
            "decoder.last_output",         # Generic decoder
            "final_output",                # Generic fallback
        ]:
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                return outputs[key]

        # Get post_loop component from topology
        flow = self.pkg.topology.get("flow", {})
        post_loop = flow.get("post_loop", [])
        if post_loop:
            final_comp = post_loop[-1]
            for key in [f"{final_comp}.last_output", f"{final_comp}.output_0"]:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    return outputs[key]

        # Fallback: find first tensor in outputs
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                return value

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
