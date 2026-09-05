from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch draws through torch; the Triton flows carry their own stream
    import torch
from typing import Dict, Any, List, Optional, Union

def to_engine_container(value: Any, mode: str) -> Any:
    """Arrays (the tokenizer engine's output, the media processors' output)
    become the executing engine's tensors: torch.from_numpy on the ATen
    branch, NBXTensor.from_numpy on the Triton branch (R33: that branch
    never imports torch). Nested dicts are walked; anything else passes."""
    import numpy as np
    if isinstance(value, dict):
        return {k: to_engine_container(v, mode) for k, v in value.items()}
    if not isinstance(value, np.ndarray):
        return value
    if mode in ("triton", "triton_sequential"):
        from neurobrix.kernels.nbx_tensor import NBXTensor
        return NBXTensor.from_numpy(np.ascontiguousarray(value))
    import torch
    return torch.from_numpy(np.ascontiguousarray(value))


class VariableResolver:
    """
    NeuroBrix Variable Resolver v0.1
    Mechanical implementation of the NBX v0.1 Variables Contract.
    ZERO SEMANTIC: No knowledge of model types or domains.
    """

    def __init__(
        self,
        variables_contract: Dict[str, Any],
        runtime_defaults: Dict[str, Any],
        component_configs: Dict[str, Dict[str, Any]],
        modules: Dict[str, Any],
        loop_state: Dict[str, Any],
        device: str = "cuda",
        mode: str = "compiled",
    ):
        self.contract = variables_contract
        # The executing engine: module outputs (arrays from the tokenizer
        # engine) enter its container here — torch on the ATen branch,
        # NBXTensor on the Triton branch (R33).
        self.mode = mode
        self.defaults = runtime_defaults
        self.component_configs = component_configs
        self.modules = modules
        self.loop_state = loop_state
        self.device = device
        self.resolved: Dict[str, Any] = {}
        self.module_handlers: Dict[str, Any] = {}
        self._sampling_generator: Any = None

    def sampling_generator(self):
        """Run-scoped sampling RNG stream (vendor semantics), or None if seedless.

        Diffusers pipelines thread ONE `torch.Generator(device).manual_seed(seed)`
        from `prepare_latents` (init draw) through every `scheduler.step()`
        (ancestral / SDE / eta-variance draws). Creating a fresh generator per
        draw — or letting scheduler noise fall back to the GLOBAL RNG seeded
        with the SAME seed — re-emits the init sequence: the step-0 ancestral
        noise comes out BIT-IDENTICAL to the init latent, a deterministic
        correlated injection along the state's own direction instead of
        independent noise (Allegro 20-step degeneracy / 100-step saturation
        root cause). One lazily-created, run-scoped generator restores the
        vendor stream exactly: init = draw 1, step-k ancestral = draw k+2.
        """
        if self._sampling_generator is None:
            seed = self.defaults.get("seed")
            if seed is None:
                return None
            import torch
            self._sampling_generator = torch.Generator(
                device=self.device).manual_seed(int(seed))
        return self._sampling_generator

    def to_engine_container(self, value: Any) -> Any:
        """Module outputs enter the executing engine's container (see the
        module-level ``to_engine_container``)."""
        return to_engine_container(value, self.mode)

    def _allocate_nbx(self, name: str, init_type: str, shape: tuple, dtype_str: str,
                      resolver_node: Dict[str, Any]):
        """The Triton branch's `allocate`: NBXTensor end-to-end. The random
        inits are drawn by the house random kernels from the branch's own
        run-scoped seeded stream (`kernels/rng_stream`, armed by the flow
        from the same data-driven seed) — never torch's generator (R33).
        The engines' noise streams are therefore distinct by construction;
        each is reproducible per (seed, draw order)."""
        import os
        import numpy as np
        from neurobrix.kernels.nbx_tensor import NBXTensor, parse_dtype
        from neurobrix.kernels import wrappers as W
        dt = parse_dtype(str(dtype_str).replace("torch.", ""))
        dev = self.device if isinstance(self.device, str) else "cuda"
        if init_type == "empty":
            return NBXTensor.empty(shape, dtype=dt, device=dev)
        if init_type == "zeros":
            return NBXTensor.zeros(shape, dtype=dt, device=dev)
        if init_type == "ones":
            return NBXTensor.ones(shape, dtype=dt, device=dev)
        if init_type == "randn":
            # Diagnostic, read-only (default off): NBX_FIXED_LATENT=<path.npy>
            # loads the initial noise from an array of the same shape so the
            # two engines (or a vendor pipeline) start from a bit-identical
            # latent for an op-by-op divergence diff.
            fixed = os.environ.get("NBX_FIXED_LATENT")
            if fixed and fixed.endswith(".npy") and os.path.exists(fixed):
                arr = np.load(fixed)
                if tuple(arr.shape) == tuple(shape):
                    return NBXTensor.from_numpy(np.ascontiguousarray(arr)).to(dt)
            return W.randn_wrapper(list(shape), dtype=dt, device=dev)
        if init_type == "uniform":
            return W.rand_wrapper(list(shape), dtype=dt, device=dev)
        if init_type == "full":
            fill_value = resolver_node.get("fill_value")
            if fill_value is None:
                raise RuntimeError(f"Variable '{name}' uses init='full' but no 'fill_value' specified")
            np_dt = np.float32 if str(dt).endswith("bfloat16") else np.dtype(str(dt).split(".")[-1].rstrip("_"))
            return NBXTensor.from_numpy(np.full(tuple(shape), fill_value, dtype=np_dt)).to(dt)
        raise RuntimeError(
            f"Variable '{name}' has unknown init type '{init_type}'.\n"
            f"Supported: empty, zeros, ones, randn, uniform, full"
        )

    def register_module_handler(self, module_name: str, handler: Any):
        """Registers a custom handler for a module (e.g. neural component executor)."""
        self.module_handlers[module_name] = handler
        self.modules[module_name] = handler

    def set(self, var_name: str, value: Any):
        """Manually sets a resolved variable value (e.g. user input)."""
        self.resolved[var_name] = value

    def get(self, var_name: str, default: Any = None) -> Any:
        """Retrieves a resolved variable, resolving it if necessary.

        Args:
            var_name: Variable name (e.g., "global.height")
            default: Default value if variable not found (None raises KeyError)
        """
        if var_name in self.resolved:
            return self.resolved[var_name]

        if var_name in self.contract:
            self.resolved[var_name] = self._resolve_single(var_name, self.contract[var_name])
            return self.resolved[var_name]

        if default is not None:
            return default

        # SPRINT 0 - R0.3: Enhanced error messages for debugging
        # Show available variables to help identify typos or missing setup
        available_resolved = list(self.resolved.keys())[:15]  # Limit to avoid spam
        available_contract = list(self.contract.keys())[:15]

        raise KeyError(
            f"Variable '{var_name}' not found.\n"
            f"  Resolved variables ({len(self.resolved)} total): {available_resolved}\n"
            f"  Contract variables ({len(self.contract)} total): {available_contract}\n"
            f"  Tip: Check if variable was set via vr.set() or defined in runtime/variables.json"
        )

    def _resolve_pointer(self, pointer: Union[str, int, float]) -> Any:
        """
        Resolves a NBX v0.1 pointer (without $ prefix).

        Formats supported:
        - "runtime.batch_size" → self.defaults["batch_size"]
        - "component.transformer.attributes.state_channels" → nested lookup
        - Literal int/float → returned as-is
        """
        # Literal numeric
        if not isinstance(pointer, str):
            return pointer

        # Try parsing as int
        try:
            return int(pointer)
        except ValueError:
            pass

        # Try parsing as float
        try:
            return float(pointer)
        except ValueError:
            pass

        parts = pointer.split(".")
        root = parts[0]

        if root == "runtime":
            # "runtime.batch_size" → defaults["batch_size"]
            if len(parts) < 2:
                raise RuntimeError(f"Malformed runtime pointer: {pointer}")
            key = parts[1]
            if key not in self.defaults:
                raise RuntimeError(f"Key '{key}' not found in runtime/defaults.json")
            return self.defaults[key]

        if root == "component":
            # "component.transformer.attributes.state_channels"
            # parts = [component, transformer, attributes, state_channels]
            if len(parts) < 4:
                raise RuntimeError(f"Malformed component pointer (need 4 parts): {pointer}")
            comp_name = parts[1]
            section = parts[2]  # "attributes"
            key = parts[3]      # "state_channels"

            if comp_name not in self.component_configs:
                raise RuntimeError(f"Component '{comp_name}' not found in configs")
            comp_cfg = self.component_configs[comp_name]

            if section not in comp_cfg:
                raise RuntimeError(f"Section '{section}' not found in component '{comp_name}'")
            if key not in comp_cfg[section]:
                raise RuntimeError(f"Key '{key}' not found in '{comp_name}.{section}'")

            return comp_cfg[section][key]

        raise RuntimeError(f"Unknown pointer root '{root}' in: {pointer}")

    def _resolve_value(self, val_spec: Any) -> Any:
        """Resolves a value reference ($global.*, $loop_state.*) or returns literal."""
        if not isinstance(val_spec, str) or not val_spec.startswith("$"):
            return val_spec

        parts = val_spec[1:].split(".")
        root = parts[0]

        if root == "global":
            var_name = ".".join(parts)
            if var_name not in self.resolved:
                if var_name not in self.contract:
                    raise RuntimeError(f"Variable reference '{var_name}' not found in contract")
                self.resolved[var_name] = self._resolve_single(var_name, self.contract[var_name])
            return self.resolved[var_name]

        if root == "loop_state":
            if len(parts) < 2:
                raise RuntimeError(f"Malformed loop_state pointer: {val_spec}")
            key = ".".join(parts[1:])
            if key not in self.loop_state:
                raise RuntimeError(f"Key '{key}' not found in loop_state")
            return self.loop_state[key]

        if root == "defaults":
            if len(parts) < 2:
                raise RuntimeError(f"Malformed defaults pointer: {val_spec}")
            key = parts[1]
            if key not in self.defaults:
                raise RuntimeError(f"Key '{key}' not found in runtime/defaults.json")
            return self.defaults[key]

        return val_spec

    def _resolve_single(self, name: str, spec: Dict[str, Any]) -> Any:
        """Executes the specific resolution method defined in the contract."""
        resolver_node = spec.get("resolver")
        if not resolver_node:
            raise RuntimeError(f"Variable '{name}' has no 'resolver' definition")

        method = resolver_node.get("method")

        if method == "allocate":
            # UNIVERSAL TENSOR ALLOCATION
            # NBX v0.1: Supports multiple initialization strategies via "init" field
            #
            # Supported init types:
            #   - "empty"  : Uninitialized memory (fast, for intermediate buffers)
            #   - "zeros"  : Zero-filled (for accumulators, padding, RNN initial states)
            #   - "ones"   : One-filled (for multiplicative masks, attention)
            #   - "randn"  : Gaussian noise N(0,1) (for diffusion latents, VAE sampling)
            #   - "uniform": Uniform [0,1) (for dropout masks, random sampling)
            #   - "full"   : Constant value via "fill_value" param
            #
            # This is ZERO HARDCODE: init type is declared in NBX contract, not assumed.

            shape_source = resolver_node.get("shape_source", {})
            if not shape_source:
                raise RuntimeError(f"Variable '{name}' has allocate but no shape_source")

            # Sort axes by name (axis_0, axis_1, axis_2, axis_3)
            sorted_axes = sorted(shape_source.keys())
            shape = []
            for axis_key in sorted_axes:
                pointer = shape_source[axis_key]
                val = self._resolve_pointer(pointer)
                if not isinstance(val, (int, float)):
                    raise RuntimeError(f"Shape axis '{axis_key}' resolved to non-numeric: {val}")
                shape.append(int(val))

            # dtype from defaults (ZERO FALLBACK: must be declared)
            dtype_str = self.defaults.get("dtype")
            if dtype_str is None:
                raise RuntimeError(
                    f"Variable '{name}' requires dtype but 'dtype' not found in runtime/defaults.json"
                )
            shape_tuple = tuple(shape)

            # Get initialization type (ZERO FALLBACK: must be declared)
            init_type = resolver_node.get("init")
            if init_type is None:
                raise RuntimeError(
                    f"Variable '{name}' uses 'allocate' method but 'init' type not specified.\n"
                    f"Supported: empty, zeros, ones, randn, uniform, full"
                )

            if self.mode in ("triton", "triton_sequential"):
                return self._allocate_nbx(name, init_type, shape_tuple, dtype_str, resolver_node)

            import torch  # the ATen branch's allocation and generator
            if not hasattr(torch, dtype_str):
                raise RuntimeError(f"Invalid torch dtype: {dtype_str}")
            dtype = getattr(torch, dtype_str)

            if init_type == "empty":
                return torch.empty(shape_tuple, dtype=dtype, device=self.device)

            elif init_type == "zeros":
                return torch.zeros(shape_tuple, dtype=dtype, device=self.device)

            elif init_type == "ones":
                return torch.ones(shape_tuple, dtype=dtype, device=self.device)

            elif init_type == "randn":
                # Diagnostic, read-only (default off): NBX_FIXED_LATENT=<path>
                # loads the initial noise latent from a file (matching shape) so
                # NeuroBrix and a vendor pipeline can be run on a BIT-IDENTICAL
                # initial latent for a same-input op-by-op divergence diff.
                import os as _os_fl
                _fl = _os_fl.environ.get("NBX_FIXED_LATENT")
                if _fl and _os_fl.path.exists(_fl):
                    _ft = torch.load(_fl, map_location=self.device, weights_only=True)
                    if tuple(_ft.shape) == tuple(shape_tuple):
                        return _ft.to(dtype=dtype, device=self.device)
                # Gaussian noise from the run-scoped sampling stream (vendor
                # semantics: ONE seeded generator drives init + every later
                # stochastic scheduler draw, in consumption order).
                generator = self.sampling_generator()
                if generator is not None:
                    return torch.randn(shape_tuple, dtype=dtype, device=self.device, generator=generator)
                return torch.randn(shape_tuple, dtype=dtype, device=self.device)

            elif init_type == "uniform":
                # Uniform [0, 1) from the run-scoped sampling stream
                generator = self.sampling_generator()
                if generator is not None:
                    return torch.rand(shape_tuple, dtype=dtype, device=self.device, generator=generator)
                return torch.rand(shape_tuple, dtype=dtype, device=self.device)

            elif init_type == "full":
                fill_value = resolver_node.get("fill_value")
                if fill_value is None:
                    raise RuntimeError(f"Variable '{name}' uses init='full' but no 'fill_value' specified")
                return torch.full(shape_tuple, fill_value, dtype=dtype, device=self.device)

            else:
                raise RuntimeError(
                    f"Variable '{name}' has unknown init type '{init_type}'.\n"
                    f"Supported: empty, zeros, ones, randn, uniform, full"
                )

        if method == "module_output":
            # NBX v0.1: module and output_key are at resolver level
            module_name = resolver_node.get("module")
            if not module_name:
                raise RuntimeError(f"Variable '{name}' has module_output but no module specified")

            if module_name not in self.modules:
                raise RuntimeError(f"Module '{module_name}' required by '{name}' not found in active modules")

            module_instance = self.modules[module_name]

            # Bind module inputs (if any)
            inputs = {}
            for k, v in resolver_node.get("inputs", {}).items():
                inputs[k] = self._resolve_value(v)

            # Execute module as a black box
            try:
                outputs = module_instance(**inputs)
            except Exception as e:
                raise RuntimeError(f"Failed to execute module '{module_name}' for variable '{name}': {str(e)}")
            outputs = self.to_engine_container(outputs)

            # Extract specific key if defined
            output_key = resolver_node.get("output_key")
            if output_key:
                if not isinstance(outputs, dict):
                    raise RuntimeError(f"Module '{module_name}' did not return a dict, cannot extract key '{output_key}'")
                if output_key not in outputs:
                    raise RuntimeError(f"Key '{output_key}' missing from module '{module_name}' output")
                return outputs[output_key]

            return outputs

        if method == "loop_state":
            # NBX v0.1: uses "loop_id" as key (ZERO FALLBACK, ZERO HARDCODE)
            loop_id = resolver_node.get("loop_id")
            if not loop_id:
                raise RuntimeError(f"Method 'loop_state' for '{name}' requires 'loop_id'")

            # loop_state is indexed by loop_id, not by a hardcoded key like "timestep"
            if loop_id not in self.loop_state:
                raise RuntimeError(
                    f"Loop '{loop_id}' not found in loop_state for variable '{name}'. "
                    f"Available loops: {list(self.loop_state.keys())}. "
                    f"Is the execution loop running?"
                )

            return self.loop_state[loop_id]

        if method == "alias":
            # NBX v0.1: alias to another variable
            target = resolver_node.get("target")
            if not target:
                raise RuntimeError(f"Method 'alias' for '{name}' requires 'target'")
            return self.get(target)

        raise RuntimeError(f"Unsupported resolver method '{method}' for variable '{name}'")

    def resolve_all(self) -> Dict[str, Any]:
        """Resolves all variables defined in the contract and returns the global store."""
        for var_name, var_spec in self.contract.items():
            # Skip internal metadata keys
            if var_name.startswith("_"):
                continue
            if var_name not in self.resolved:
                self.resolved[var_name] = self._resolve_single(var_name, var_spec)
        return self.resolved
