"""
Input Synthesizer

ZERO HARDCODE: All synthesis rules come from topology.synthesis.
ZERO SEMANTIC: No knowledge of "resolution", "aspect_ratio" - only synthesis methods.

Creates missing inputs based on declared synthesis rules in topology.
Also handles input remapping and shape transformations.
"""

import torch
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
    from neurobrix.core.runtime.graph_executor import GraphExecutor


def get_device_prefix(vendor: str, arch: str) -> str:
    """Get device prefix for vendor."""
    if vendor == "nvidia":
        return "cuda"
    elif vendor == "amd":
        return "hip"
    elif vendor == "apple":
        return "mps"
    return "cpu"


class InputSynthesizer:
    """
    Synthesizes missing inputs using rules from topology.synthesis.

    This class is responsible for:
    1. Synthesizing missing inputs (from_dimensions, compute_ratio, zeros, ones)
    2. Remapping topology input names to graph input names
    3. Applying shape transformations (sequence slicing)

    ZERO HARDCODE: All synthesis rules from topology.json.
    ZERO SEMANTIC: No domain knowledge - just applies declared rules.
    """

    def __init__(
        self,
        topology: Dict[str, Any],
        variable_resolver: 'VariableResolver',
        plan: Any,
        modules: Dict[str, Any],
        executors: Dict[str, 'GraphExecutor']
    ):
        """
        Initialize input synthesizer.

        Args:
            topology: Full topology dict with synthesis rules
            variable_resolver: VariableResolver for accessing defaults and globals
            plan: Prism plan with device allocation info
            modules: Loaded modules (scheduler, tokenizer, etc.)
            executors: Component executors for graph info
        """
        self._topology = topology
        self._variable_resolver = variable_resolver
        self._plan = plan
        self._modules = modules
        self._executors = executors

    def synthesize_missing_inputs(
        self,
        comp_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize missing inputs using rules from topology.synthesis.

        UNIVERSAL: Handles any model type based on declared synthesis rules.

        Methods:
        - from_dimensions: Create tensor from global.height/width
        - compute_ratio: Create ratio tensor from height/width
        - zeros: Create zero tensor
        - ones: Create ones tensor

        NESTED DICT SUPPORT:
        Input names with dots (e.g., "added_cond_kwargs.resolution") create nested dicts:
        inputs["added_cond_kwargs"]["resolution"] = tensor

        Args:
            comp_name: Component name
            inputs: Current resolved inputs

        Returns:
            Inputs with synthesized values added
        """
        # Get synthesis rules for this component
        synthesis_rules = self._topology.get("synthesis", {}).get(comp_name, {})
        if not synthesis_rules:
            return inputs

        # Get batch size from existing inputs
        batch_size = 1
        for val in inputs.values():
            if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                batch_size = val.shape[0]
                break

        # Get device and dtype from existing inputs, with data-driven fallback
        # ZERO HARDCODE: device from vendor config, dtype from topology
        device = None
        dtype = None
        for val in inputs.values():
            if isinstance(val, torch.Tensor):
                device = val.device
                dtype = val.dtype
                break

        # Fallback to plan/config if no tensors in inputs
        if device is None:
            # Get vendor/arch from plan and derive device prefix
            vendor = getattr(self._plan, 'vendor', 'nvidia')
            arch = getattr(self._plan, 'architecture', 'volta')
            device_prefix = get_device_prefix(vendor, arch)
            device_index = getattr(self._plan, 'primary_device_index', 0)
            device = f"{device_prefix}:{device_index}"

        if dtype is None:
            # Get dtype from topology extracted_values or defaults
            dominant_dtype = self._topology.get("extracted_values", {}).get("_global", {}).get("dominant_dtype", "float16")
            dtype = getattr(torch, dominant_dtype, torch.float16)

        def _set_nested(d: Dict, key_path: str, value: Any) -> None:
            """Set value in nested dict using dot notation path."""
            parts = key_path.split(".")
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value

        def _has_nested(d: Dict, key_path: str) -> bool:
            """Check if nested key exists in dict."""
            parts = key_path.split(".")
            for part in parts:
                if not isinstance(d, dict) or part not in d:
                    return False
                d = d[part]
            return True

        # Apply synthesis rules
        for input_name, rule in synthesis_rules.items():
            # Check if already provided (handles both flat and nested keys)
            if input_name in inputs or _has_nested(inputs, input_name):
                continue  # Already provided

            method = rule.get("method")

            if method == "from_dimensions":
                height, width = self._get_dimensions()
                if height is None or width is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: Cannot synthesize '{input_name}' for {comp_name}.\n"
                        f"Required 'global.height' and 'global.width' but got: "
                        f"height={height}, width={width}\n"
                        "Provide height/width via --set or runtime/defaults.json"
                    )

                # Create [batch, 2] tensor with [[height, width], ...]
                tensor = torch.tensor([[height, width]], device=device, dtype=dtype).repeat(batch_size, 1)
                _set_nested(inputs, input_name, tensor)

            elif method == "compute_ratio":
                height, width = self._get_dimensions()
                if height is None or width is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: Cannot synthesize '{input_name}' for {comp_name}.\n"
                        f"Required 'global.height' and 'global.width' but got: "
                        f"height={height}, width={width}\n"
                        "Provide height/width via --set or runtime/defaults.json"
                    )

                ratio = height / width if width > 0 else 1.0
                # Create [batch, 1] tensor
                tensor = torch.tensor([[ratio]], device=device, dtype=dtype).repeat(batch_size, 1)
                _set_nested(inputs, input_name, tensor)

            elif method == "zeros":
                # Create zero tensor with specified shape
                shape = rule.get("shape", [batch_size, 1])
                resolved_shape = self._resolve_shape(shape, batch_size)
                tensor = torch.zeros(resolved_shape, device=device, dtype=dtype)
                _set_nested(inputs, input_name, tensor)

            elif method == "ones":
                # Create ones tensor with specified shape (for attention masks)
                shape = rule.get("shape", [batch_size, 1])
                resolved_shape = self._resolve_shape(shape, batch_size)
                tensor = torch.ones(resolved_shape, device=device, dtype=dtype)
                _set_nested(inputs, input_name, tensor)

            elif method == "latent_image_ids":
                # Create Flux-style latent image position IDs
                # Shape: [latent_h * latent_w, 3] where dim 0 = batch idx, dim 1 = row, dim 2 = col
                # DATA-DRIVEN: latent dimensions from topology shapes
                latent_h = rule.get("latent_height")
                latent_w = rule.get("latent_width")
                if latent_h is None or latent_w is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: latent_image_ids synthesis for '{input_name}' requires "
                        f"'latent_height' and 'latent_width' in rule."
                    )
                ids = torch.zeros(latent_h, latent_w, 3)
                ids[..., 1] = ids[..., 1] + torch.arange(latent_h)[:, None]
                ids[..., 2] = ids[..., 2] + torch.arange(latent_w)[None, :]
                tensor = ids.reshape(-1, 3).to(device=device)
                rule_dtype = rule.get("dtype")
                if rule_dtype:
                    tensor = tensor.to(getattr(torch, rule_dtype, torch.float32))
                _set_nested(inputs, input_name, tensor)

            else:
                pass  # Unknown synthesis method - skip silently

        return inputs

    def _get_dimensions(self) -> tuple:
        """
        Get height and width from merged defaults and user overrides.

        Returns:
            Tuple of (height, width), may contain None if not set
        """
        # ZERO FALLBACK: Get height/width from merged defaults (family + pkg + user)
        height = self._variable_resolver.defaults.get("height")
        width = self._variable_resolver.defaults.get("width")

        # Check if user provided override via variable resolver
        try:
            user_height = self._variable_resolver.get("global.height")
            if user_height is not None:
                height = user_height
        except KeyError:
            pass  # Not in variables, use defaults

        try:
            user_width = self._variable_resolver.get("global.width")
            if user_width is not None:
                width = user_width
        except KeyError:
            pass  # Not in variables, use defaults

        return height, width

    def _resolve_shape(self, shape: List, batch_size: int) -> List[int]:
        """
        Resolve shape with symbolic batch dimension.

        Args:
            shape: Shape with potentially symbolic dims
            batch_size: Resolved batch size

        Returns:
            Fully resolved integer shape
        """
        resolved_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim == -1:
                resolved_shape.append(batch_size)
            else:
                resolved_shape.append(dim)
        return resolved_shape

    def remap_inputs_to_graph(
        self,
        comp_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Remap topology input names to graph input names.

        ZERO HEURISTIC: Uses explicit matching or input_aliases from topology.
        No positional guessing.

        Strategy:
        1. If names match exactly -> use as-is
        2. If single input on both sides -> direct map (unambiguous)
        3. Check input_aliases in topology.interface
        4. Otherwise -> CRASH (no silent positional mapping)

        Args:
            comp_name: Component name
            inputs: Inputs with topology names

        Returns:
            Inputs with graph names
        """
        if comp_name not in self._executors:
            return inputs

        executor = self._executors[comp_name]
        if not hasattr(executor, '_dag'):
            return inputs

        # Extract graph input names from DAG
        dag = executor._dag
        if dag is None:
            return inputs
        graph_input_names = []
        for tensor_id, tensor_info in dag.get("tensors", {}).items():
            if tensor_info.get("is_input"):
                input_name = tensor_info.get("input_name")
                if input_name:
                    graph_input_names.append(input_name)

        if not graph_input_names:
            return inputs

        # Check if all topology inputs match graph inputs
        topology_names = set(inputs.keys())
        graph_names = set(graph_input_names)

        if topology_names == graph_names:
            # Perfect match - no remapping needed
            return inputs

        # Single input case: direct map (unambiguous)
        if len(graph_input_names) == 1 and len(inputs) == 1:
            graph_name = graph_input_names[0]
            topology_name = next(iter(inputs.keys()))
            if graph_name != topology_name:
                return {graph_name: inputs[topology_name]}

        # Multi-input case: use explicit matching and input_aliases
        comp_info = self._topology.get("components", {}).get(comp_name, {})
        aliases = comp_info.get("interface", {}).get("input_aliases", {})

        remapped = {}
        unmatched_topology = []

        for topo_name, value in inputs.items():
            if topo_name in graph_names:
                # Direct match
                remapped[topo_name] = value
            elif topo_name in aliases:
                # Use declared alias
                graph_name = aliases[topo_name]
                if graph_name in graph_names:
                    remapped[graph_name] = value
                else:
                    raise RuntimeError(
                        f"ZERO FALLBACK: Alias '{topo_name}' -> '{graph_name}' invalid.\n"
                        f"'{graph_name}' not in graph inputs: {graph_input_names}"
                    )
            else:
                unmatched_topology.append(topo_name)

        # Check for unmatched inputs
        unmatched_graph = [g for g in graph_input_names if g not in remapped]

        if unmatched_topology and unmatched_graph:
            raise RuntimeError(
                f"ZERO FALLBACK: Input name mismatch for '{comp_name}'.\n"
                f"Topology provides: {unmatched_topology}\n"
                f"Graph expects: {unmatched_graph}\n"
                f"Available aliases: {aliases}\n"
                "Fix: Add 'input_aliases' to topology.interface or fix input names."
            )

        return remapped

    def apply_shape_transforms(
        self,
        comp_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Auto-slice sequence dimensions when input shapes don't match expected shapes.

        UNIVERSAL: Checks topology.components[comp].shapes for expected dimensions.
        If input tensor is longer in a sequence dimension, slices to expected length.

        This handles cases like Sana where:
        - text_encoder outputs [B, 506, 2304]
        - transformer expects [B, 300, 2304]
        - Pipeline normally slices, but we need to do it here

        PADDING_SIDE AWARE: For LEFT padding (Gemma/Sana), slices from END [-n:]
        because content is at the end of the sequence. For RIGHT padding, slices
        from START [:n] because content is at the beginning.

        Args:
            comp_name: Component name
            inputs: Current inputs

        Returns:
            Inputs with shape transformations applied
        """
        comp_config = self._topology.get("components", {}).get(comp_name, {})
        expected_shapes = comp_config.get("shapes", {})

        # Detect padding side from tokenizer module (if present)
        padding_side = "right"  # Default
        tokenizer = self._modules.get("tokenizer")
        if tokenizer and hasattr(tokenizer, '_padding_side'):
            padding_side = tokenizer._padding_side

        if not expected_shapes:
            return inputs

        # Skip resolution-dependent inputs for loop components.
        # The state variable and spatial inputs have dynamic shapes that depend
        # on image resolution, not tokenization. Symbolic shapes handle them.
        flow = self._topology.get("flow", {})
        loop_cfg = flow.get("loop", {})
        loop_comps = loop_cfg.get("components", [])
        state_input = loop_cfg.get("state_input")
        skip_inputs: set = set()
        if comp_name in loop_comps and state_input:
            skip_inputs.add(state_input)
            # Also skip spatial position IDs (resolution-dependent)
            for inp_name in expected_shapes:
                if inp_name.endswith("_ids") and inp_name != "input_ids":
                    skip_inputs.add(inp_name)

        for input_name, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                continue

            if input_name in skip_inputs:
                continue

            expected_shape = expected_shapes.get(input_name)
            if not expected_shape:
                continue

            actual_shape = list(tensor.shape)

            # Skip if shapes match
            if actual_shape == expected_shape:
                continue

            # Check if we need to slice a sequence dimension
            if len(actual_shape) != len(expected_shape):
                continue

            # Check if this looks like a sequence tensor needing slicing
            if len(actual_shape) >= 2:
                seq_dim_idx = 1  # Typically the sequence dimension
                actual_seq = actual_shape[seq_dim_idx]
                expected_seq = expected_shape[seq_dim_idx]

                # Only slice if actual is longer and other dims match
                needs_slice = (
                    actual_seq > expected_seq and
                    actual_shape[seq_dim_idx + 1:] == expected_shape[seq_dim_idx + 1:]
                )

                if needs_slice:
                    # Check for Sana-style complex_human_instruction pattern
                    tokenizer_vals = self._topology.get("extracted_values", {}).get("tokenizer", {})
                    complex_human_instruction = tokenizer_vals.get("complex_human_instruction")

                    if complex_human_instruction and "encoder_hidden_states" in input_name:
                        # SANA PATTERN: select_index = [0] + list(range(-max_length + 1, 0))
                        # This takes token 0 (BOS) + last (n-1) tokens which includes the actual content
                        # when using left-padding (Gemma tokenizer)
                        select_index = [0] + list(range(-expected_seq + 1, 0))
                        sliced = tensor[:, select_index]
                    elif padding_side == "left":
                        # LEFT padding: content at end, take LAST n tokens
                        sliced = tensor[:, -expected_seq:]
                    else:
                        # RIGHT padding: content at start, take FIRST n tokens
                        sliced = tensor[:, :expected_seq]
                    inputs[input_name] = sliced

                    # Also slice attention mask if present
                    mask_name = input_name.replace("hidden_states", "attention_mask")
                    if mask_name in inputs and mask_name != input_name:
                        mask = inputs[mask_name]
                        if isinstance(mask, torch.Tensor) and len(mask.shape) >= 2:
                            if mask.shape[1] > expected_seq:
                                if complex_human_instruction and "encoder" in mask_name:
                                    # Use same select_index pattern as hidden_states
                                    select_index = [0] + list(range(-expected_seq + 1, 0))
                                    inputs[mask_name] = mask[:, select_index]
                                elif padding_side == "left":
                                    inputs[mask_name] = mask[:, -expected_seq:]
                                else:
                                    inputs[mask_name] = mask[:, :expected_seq]

        return inputs
