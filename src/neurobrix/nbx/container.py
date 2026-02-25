"""
NBX Container - .nbx File Format Handler

Creates and reads .nbx container files (ZIP-based format).
Follows NBX Format Specification v1.0.

Category is derived from config.json (_class_name or architectures).
"""

import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, BinaryIO, Union
from datetime import datetime
from io import BytesIO
from dataclasses import dataclass, field

from .cache import ensure_extracted

try:
    import safetensors
    from safetensors.numpy import save_file as save_safetensors
    from safetensors.numpy import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .spec import NBX_VERSION, NBX_MAGIC


def get_category_from_config(config: Dict[str, Any]) -> str:
    """
    Get category directly from config.json.

    ZERO HARDCODE: Category IS the value from config, not a mapping.
    Priority: _class_name > architectures[0] > model_type
    """
    # _class_name (diffusers) - USE DIRECTLY
    class_name = config.get("_class_name")
    if class_name:
        return class_name

    # architectures (transformers) - USE DIRECTLY
    archs = config.get("architectures", [])
    if archs:
        return archs[0]

    # model_type (transformers fallback)
    model_type = config.get("model_type")
    if model_type:
        return model_type

    return "unknown"


# =============================================================================
# COMPONENT DATA STRUCTURE
# =============================================================================

@dataclass
class ComponentData:
    """
    Data for a single component in the NBX container.

    Category derived from config.json (_class_name or architectures).
    """
    name: str
    category: str  # Derived from config.json
    config: Dict[str, Any]  # Original config.json
    graph: Optional[Dict[str, Any]] = None  # graph.json (TensorDAG) for neural components
    weight_paths: List[str] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None  # profile.json with trace-time values
    attributes: Optional[Dict[str, Any]] = None  # runtime.json attributes (state_channels, etc.)

    @property
    def is_neural(self) -> bool:
        """
        True if this is a neural component (has graph OR has weights).

        For pure LLMs using UniversalLMExecutor, components have weights but no graph.
        They should still be allocated by Prism.
        """
        return self.graph is not None or (self.weight_paths is not None and len(self.weight_paths) > 0)

    def get_dominant_dtype(self) -> str:
        """
        Get dominant dtype for this component.

        Priority:
        1. profile.hints.dtype (ACTUAL weight dtype from trace hints)
        2. config.torch_dtype (HuggingFace original - actual weight dtype)
        3. profile.dtype (trace-time dtype - may differ from weights if hardware converted)
        4. float32 (fallback)

        CRITICAL: profile.dtype is trace-time dtype (after hardware conversion).
        For bf16 models on V100, trace runs as fp16 but weights remain bf16.
        We need the ACTUAL weight dtype to choose correct conversion:
        - bf16 → fp32 (same exponent range, no overflow)
        - bf16 → fp16 would cause precision loss
        """
        # PRIORITY 1: hints.dtype from profile (ACTUAL weight dtype)
        # This is the dtype of weights BEFORE any trace-time conversion
        if self.profile:
            hints = self.profile.get("hints", {})
            hints_dtype = hints.get("dtype")
            if hints_dtype:
                return hints_dtype

        # PRIORITY 2: torch_dtype from HF config (original model dtype)
        torch_dtype = self.config.get("torch_dtype")
        if torch_dtype:
            return torch_dtype

        # PRIORITY 3: profile.dtype (trace-time, may differ from weights)
        if self.profile:
            profile_dtype = self.profile.get("dtype")
            if profile_dtype:
                return profile_dtype

        # Default
        return "float32"


class NBXContainer:
    """
    Handle .nbx container files (ZIP-based format).

    The .nbx format is a ZIP archive containing:
    - manifest.json: Metadata about the model
    - components/{name}/graph.json: TensorDAG for neural components
    - components/{name}/weights/: SafeTensors weight files
    - components/{name}/config.json: Component configuration
    - tokenizer/: Tokenizer files (optional)

    Example - Creating:
        container = NBXContainer()
        container.set_manifest({...})
        container.add_weights("transformer", {"weight": array, "bias": array})
        container.save("model.nbx")

    Example - Reading:
        container = NBXContainer.load("model.nbx")
        for comp in container.get_neural_components():
            dag = comp.graph  # TensorDAG dict, already loaded
    """

    def __init__(self):
        """Initialize an empty NBX container."""
        self._manifest: Optional[Dict] = None
        self._graph: Optional[Dict] = None  # Model graph (TensorDAG)
        self._config: Optional[Dict] = None
        self._weights: Dict[str, Dict[str, Any]] = {}
        self._files: Dict[str, bytes] = {}  # Additional files
        self._source_path: Optional[Path] = None
        self._cache_path: Optional[Path] = None  # Extracted cache path (~/.neurobrix/cache/<name>)
        self._weight_paths: Dict[str, List[str]] = {}  # component -> [paths in zip]

        # Component-based storage (UNIVERSAL DETECTION)
        self._components: Dict[str, ComponentData] = {}

    @property
    def cache_path(self) -> Optional[Path]:
        """Get the cache path where .nbx was extracted (~/.neurobrix/cache/<name>)."""
        return self._cache_path

    def get_shard_sizes(self) -> Dict[str, Dict[str, int]]:
        """
        Get actual shard sizes from NBX archive.
        
        ZERO HARDCODE: Reads real file sizes, no assumptions.
        
        Returns:
            {component_name: {shard_path: size_bytes}}
        """
        import zipfile
        from pathlib import Path

        shard_sizes: Dict[str, Dict[str, int]] = {}
        source = Path(self._source_path)

        # Handle both ZIP files and extracted directories
        if source.is_dir():
            # Already extracted - scan filesystem
            for shard_file in source.rglob("*.safetensors"):
                rel_path = str(shard_file.relative_to(source))
                parts = rel_path.split('/')
                # Format: components/transformer/weights/shard_XXX.safetensors
                if len(parts) >= 3 and parts[0] == 'components':
                    comp_name = parts[1]
                    if comp_name not in shard_sizes:
                        shard_sizes[comp_name] = {}
                    shard_sizes[comp_name][rel_path] = shard_file.stat().st_size
        else:
            # ZIP file - read from archive
            with zipfile.ZipFile(str(self._source_path), 'r') as z:
                for info in z.infolist():
                    if '.safetensors' in info.filename:
                        parts = info.filename.split('/')
                        # Format: components/transformer/weights/block_XXXX.safetensors
                        if len(parts) >= 3 and parts[0] == 'components':
                            comp_name = parts[1]
                            if comp_name not in shard_sizes:
                                shard_sizes[comp_name] = {}
                            # ZERO HARDCODE: Real size from ZIP metadata
                            shard_sizes[comp_name][info.filename] = info.file_size

        return shard_sizes

    # =========================================================================
    # MANIFEST
    # =========================================================================

    def set_manifest(
        self,
        model_name: str,
        framework_source: str = "pytorch",
        model_type: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Set the manifest metadata.

        Args:
            model_name: Name of the model
            framework_source: Source framework (pytorch, tensorflow, onnx, etc.)
            model_type: Type of model (diffusion, llm, vision, etc.)
            description: Human-readable description
            **kwargs: Additional metadata fields
        """
        self._manifest = {
            "format_version": NBX_VERSION,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "framework_source": framework_source,
        }

        if model_type:
            self._manifest["model_type"] = model_type
        if description:
            self._manifest["description"] = description

        # Add any additional fields
        self._manifest.update(kwargs)

    def get_manifest(self) -> Optional[Dict]:
        """Get the manifest metadata."""
        return self._manifest

    # =========================================================================
    # GRAPH (TensorDAG)
    # =========================================================================

    def set_graph(self, graph: Dict[str, Any]) -> None:
        """
        Set the model execution graph (TensorDAG).

        Args:
            graph: TensorDAG dictionary with ops, tensors, execution_order
        """
        self._graph = graph

    def get_graph(self) -> Optional[Dict]:
        """Get the model execution graph (TensorDAG)."""
        return self._graph

    # Legacy aliases for backward compatibility
    def set_topology(self, topology: Dict[str, Any]) -> None:
        """DEPRECATED: Use set_graph() instead."""
        self._graph = topology

    def get_topology(self) -> Optional[Dict]:
        """DEPRECATED: Use get_graph() instead."""
        return self._graph

    # =========================================================================
    # CONFIG
    # =========================================================================

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the model configuration.

        Args:
            config: Model configuration dictionary
        """
        self._config = config

    def get_config(self) -> Optional[Dict]:
        """Get the model configuration."""
        return self._config

    # =========================================================================
    # WEIGHTS
    # =========================================================================

    def add_weights(
        self,
        component_name: str,
        weights: Dict[str, "np.ndarray"],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add weights for a component.

        Args:
            component_name: Name of the component (e.g., "unet", "text_encoder")
            weights: Dictionary mapping weight names to numpy arrays
            metadata: Optional metadata for the weights
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for weights handling")

        self._weights[component_name] = {
            "tensors": weights,
            "metadata": metadata or {},
        }

    def get_weights(self, component_name: str) -> Optional[Dict[str, "np.ndarray"]]:
        """
        Get weights for a component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary mapping weight names to numpy arrays
        """
        if component_name in self._weights:
            return self._weights[component_name].get("tensors")
        return None

    def list_weight_components(self) -> List[str]:
        """List all component names that have weights."""
        return list(self._weights.keys())

    # =========================================================================
    # ADDITIONAL FILES
    # =========================================================================

    def add_file(self, path: str, content: Union[bytes, str]) -> None:
        """
        Add an arbitrary file to the container.

        Args:
            path: Path within the container (e.g., "tokenizer/vocab.json")
            content: File content (bytes or string)
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._files[path] = content

    def get_file(self, path: str) -> Optional[bytes]:
        """
        Get a file from the container.

        Args:
            path: Path within the container

        Returns:
            File content as bytes, or None if not found
        """
        return self._files.get(path)

    def list_files(self) -> List[str]:
        """List all additional files in the container."""
        return list(self._files.keys())

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, output_path: str, compression: int = zipfile.ZIP_DEFLATED) -> None:
        """
        Save the container to a .nbx file.

        Args:
            output_path: Path for the output .nbx file
            compression: ZIP compression method
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, "w", compression=compression) as zf:
            # Write manifest
            if self._manifest:
                zf.writestr(
                    "manifest.json",
                    json.dumps(self._manifest, indent=2)
                )

            # NOTE: Root-level graph.json NOT saved here
            # Graphs are stored per-component in components/{name}/graph.json
            # This is handled by the importer (nbx_builder.py)

            # Write config
            if self._config:
                zf.writestr(
                    "config.json",
                    json.dumps(self._config, indent=2)
                )

            # Write weights
            if SAFETENSORS_AVAILABLE:
                for component_name, weight_data in self._weights.items():
                    tensors = weight_data.get("tensors", {})
                    if tensors:
                        # Save to bytes buffer
                        buffer = BytesIO()
                        save_safetensors(tensors, buffer)
                        buffer.seek(0)

                        # Write to ZIP
                        zf.writestr(
                            f"weights/{component_name}.safetensors",
                            buffer.read()
                        )

            # Write additional files
            for path, content in self._files.items():
                zf.writestr(path, content)

        # Update source path
        self._source_path = output_path

    @classmethod
    def load(cls, nbx_path: str) -> "NBXContainer":
        """
        Load a container from cache (extracts .nbx on first run if needed).

        CACHE ARCHITECTURE:
        - .nbx file is ONLY for transport/packaging
        - Runtime ALWAYS reads from ~/.neurobrix/cache/<model_name>/
        - ensure_extracted() handles extraction on first run

        Args:
            nbx_path: Path to the .nbx file

        Returns:
            NBXContainer instance
        """
        nbx_path = Path(nbx_path)
        if not nbx_path.exists():
            raise FileNotFoundError(f"NBX file not found: {nbx_path}")

        # Extract to cache (fast if already cached)
        cache_path = ensure_extracted(nbx_path)

        container = cls()
        container._source_path = nbx_path
        container._cache_path = cache_path  # Store cache path for runtime access

        # Read manifest from cache
        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                container._manifest = json.load(f)

        # Read config from cache
        config_path = cache_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                container._config = json.load(f)

        # Index weight file paths (lazy loading - weights loaded by executor)
        components_dir = cache_path / "components"
        if components_dir.exists():
            for comp_dir in components_dir.iterdir():
                if comp_dir.is_dir():
                    weights_dir = comp_dir / "weights"
                    if weights_dir.exists():
                        comp_name = comp_dir.name
                        container._weight_paths[comp_name] = []
                        for weight_file in weights_dir.glob("*.safetensors"):
                            rel_path = str(weight_file.relative_to(cache_path))
                            container._weight_paths[comp_name].append(rel_path)

        # Load auxiliary files (tokenizer vocab, etc.) from cache
        for file_path in cache_path.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(cache_path))
            # Skip weights (lazy loaded by executor)
            if ".safetensors" in rel_path:
                continue
            # Skip already processed files
            if rel_path in ("manifest.json", "config.json"):
                continue
            # Skip component-level files (handled by _load_components_from_cache)
            if rel_path.startswith("components/"):
                parts = rel_path.split("/")
                if len(parts) == 3 and parts[2] in ("config.json", "detection.json", "graph.json"):
                    continue
            # Load auxiliary files
            with open(file_path, "rb") as f:
                container._files[rel_path] = f.read()

        # Load components from cache
        container._load_components_from_cache(cache_path)

        return container

    def _load_components(self, zf: zipfile.ZipFile) -> None:
        """
        Load all components.

        ZERO HARDCODE: Category from config.json (_class_name).
        """
        # Find all components from manifest or by scanning for config.json
        component_names = set()

        # From manifest
        if self._manifest:
            components = self._manifest.get("components", {})
            component_names.update(components.keys())

        # By scanning for config files
        for name in zf.namelist():
            # Match components/{name}/*config.json
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "components" and "config.json" in parts[2]:
                comp_name = parts[1]
                component_names.add(comp_name)

        # Load each component
        for comp_name in component_names:
            self._load_single_component(zf, comp_name)

    def _load_single_component(self, zf: zipfile.ZipFile, comp_name: str) -> None:
        """Load a single component with all its files."""
        prefix = f"components/{comp_name}/"
        file_list = zf.namelist()

        # Load config (try config.json, then {name}_config.json, then any *config.json)
        config = {}
        config_candidates = [
            f"{prefix}config.json",
            f"{prefix}{comp_name}_config.json",
            f"{prefix}scheduler_config.json",
            f"{prefix}tokenizer_config.json"
        ]

        # Also scan specifically for this component if needed
        for f in file_list:
            if f.startswith(prefix) and "config.json" in f:
                if f not in config_candidates:
                    config_candidates.append(f)

        found_config = False
        for config_path in config_candidates:
            if config_path in file_list:
                try:
                    config = json.loads(zf.read(config_path).decode("utf-8"))
                    found_config = True
                    break
                except Exception:
                    continue

        # Category from config.json (ZERO HARDCODE)
        category = get_category_from_config(config)

        # Load graph.json (neural components only) - TensorDAG format
        graph_path = f"{prefix}graph.json"
        if graph_path in file_list:
            graph = json.loads(zf.read(graph_path).decode("utf-8"))
        else:
            graph = None

        # Load profile.json (trace-time values - SOURCE OF TRUTH for dtype)
        profile_path = f"{prefix}profile.json"
        if profile_path in file_list:
            profile = json.loads(zf.read(profile_path).decode("utf-8"))
        else:
            profile = None

        # NOTE: attributes are loaded from cache only (runtime.json)
        # Per CLAUDE.md: Runtime ALWAYS reads from ~/.neurobrix/cache/<name>/

        # Find weight paths
        weight_paths = [
            n for n in file_list
            if n.startswith(f"{prefix}weights/") and n.endswith(".safetensors")
        ]

        # Create ComponentData (attributes=None here, loaded from cache later)
        self._components[comp_name] = ComponentData(
            name=comp_name,
            category=category,
            config=config,
            graph=graph,
            weight_paths=weight_paths,
            profile=profile,
            attributes=None,
        )

    def _load_components_from_cache(self, cache_path: Path) -> None:
        """
        Load all components from cache directory.

        CACHE ARCHITECTURE: Runtime reads from ~/.neurobrix/cache/<name>/, NOT from .nbx.
        This method iterates the cache components directory structure.

        ZERO HARDCODE: Category from config.json (_class_name).
        """
        components_dir = cache_path / "components"
        if not components_dir.exists():
            return

        # Find all components by scanning directories
        for comp_dir in components_dir.iterdir():
            if not comp_dir.is_dir():
                continue

            comp_name = comp_dir.name
            self._load_single_component_from_cache(cache_path, comp_name)

    def _load_single_component_from_cache(self, cache_path: Path, comp_name: str) -> None:
        """
        Load a single component from cache directory.

        CACHE ARCHITECTURE: Reads from ~/.neurobrix/cache/<name>/components/<comp_name>/
        """
        comp_dir = cache_path / "components" / comp_name

        # Load config (try config.json, then {name}_config.json, then any *config.json)
        config = {}
        config_candidates = [
            comp_dir / "config.json",
            comp_dir / f"{comp_name}_config.json",
            comp_dir / "scheduler_config.json",
            comp_dir / "tokenizer_config.json"
        ]

        # Also scan for any *config*.json files
        for f in comp_dir.glob("*config*.json"):
            if f not in config_candidates:
                config_candidates.append(f)

        for config_path in config_candidates:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    break
                except Exception:
                    continue

        # Category from config.json (ZERO HARDCODE)
        category = get_category_from_config(config)

        # Load graph.json (neural components only) - TensorDAG format
        graph_path = comp_dir / "graph.json"
        if graph_path.exists():
            with open(graph_path) as f:
                graph = json.load(f)
        else:
            graph = None

        # Load profile.json (trace-time values - SOURCE OF TRUTH for dtype)
        profile_path = comp_dir / "profile.json"
        if profile_path.exists():
            with open(profile_path) as f:
                profile = json.load(f)
        else:
            profile = None

        # Load runtime.json attributes (ZERO HARDCODE - used by SmartSolver)
        runtime_path = comp_dir / "runtime.json"
        attributes = None
        if runtime_path.exists():
            with open(runtime_path) as f:
                runtime_data = json.load(f)
                attributes = runtime_data.get("attributes", {})

        # Find weight paths (relative to cache_path for consistency)
        weights_dir = comp_dir / "weights"
        weight_paths = []
        if weights_dir.exists():
            for weight_file in weights_dir.glob("*.safetensors"):
                rel_path = str(weight_file.relative_to(cache_path))
                weight_paths.append(rel_path)

        # Create ComponentData
        self._components[comp_name] = ComponentData(
            name=comp_name,
            category=category,
            config=config,
            graph=graph,
            weight_paths=weight_paths,
            profile=profile,
            attributes=attributes,
        )

    # =========================================================================
    # COMPONENT ACCESS (UNIVERSAL DETECTION)
    # =========================================================================

    def get_component(self, name: str) -> ComponentData:
        """
        Get a component by name.

        ZERO HARDCODE: Use this to get input_spec instead of hardcoding shapes.

        Args:
            name: Component name (e.g., "transformer", "vae")

        Returns:
            ComponentData with detection info and input_spec
        """
        if name not in self._components:
            raise KeyError(f"Component not found: {name}. Available: {list(self._components.keys())}")
        return self._components[name]

    def get_neural_components(self) -> List[ComponentData]:
        """
        Get all neural components (those with graph.json).

        Returns:
            List of ComponentData for neural components
        """
        return [c for c in self._components.values() if c.is_neural]

    def get_input_spec(self, comp_name: str) -> Dict[str, Any]:
        """
        Get input_spec for a component from detection.json.

        ZERO HARDCODE: Use this instead of hardcoding shapes/dtypes!

        Args:
            comp_name: Component name

        Returns:
            input_spec dict with "inputs", "dynamic_axes", etc.

        Example:
            input_spec = container.get_input_spec("transformer")
            hidden_shape = input_spec["inputs"]["hidden_states"]["shape"]
            hidden_dtype = input_spec["inputs"]["hidden_states"]["dtype"]
        """
        if comp_name not in self._components:
            raise KeyError(f"Component not found: {comp_name}")
        return self._components[comp_name].input_spec

    def list_components(self) -> List[str]:
        """List all component names."""
        return list(self._components.keys())

    def get_component_categories(self) -> Dict[str, str]:
        """
        Get category for each component.

        Returns:
            Dict mapping component name to detected category
        """
        return {name: comp.category for name, comp in self._components.items()}

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(self) -> List[str]:
        """
        Validate the container structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check manifest
        if not self._manifest:
            errors.append("Missing manifest")
        else:
            required = ["format_version", "model_name", "created_at", "framework_source"]
            for key in required:
                if key not in self._manifest:
                    errors.append(f"Manifest missing required field: {key}")

        # Check components have graphs
        neural_comps = self.get_neural_components()
        for comp in neural_comps:
            if comp.graph:
                # Validate TensorDAG format
                if comp.graph.get("format") != "tensor_dag":
                    errors.append(f"Component '{comp.name}' graph is not tensor_dag format")

        return errors

    def compute_checksum(self) -> str:
        """
        Compute SHA256 checksum of the container contents.

        Returns:
            Hex-encoded SHA256 hash
        """
        hasher = hashlib.sha256()

        # Hash manifest
        if self._manifest:
            hasher.update(json.dumps(self._manifest, sort_keys=True).encode())

        # Hash component graphs
        for comp in sorted(self._components.values(), key=lambda c: c.name):
            if comp.graph:
                hasher.update(json.dumps(comp.graph, sort_keys=True).encode())

        # Hash weight shapes (not values - too slow)
        for name in sorted(self._weights.keys()):
            weight_data = self._weights[name]
            tensors = weight_data.get("tensors", {})
            for tensor_name in sorted(tensors.keys()):
                tensor = tensors[tensor_name]
                hasher.update(f"{name}/{tensor_name}:{tensor.shape}".encode())

        return hasher.hexdigest()

    # =========================================================================
    # INFO
    # =========================================================================

    def info(self) -> Dict[str, Any]:
        """
        Get summary information about the container.

        Returns:
            Dictionary with container summary
        """
        info = {
            "source_path": str(self._source_path) if self._source_path else None,
            "has_manifest": self._manifest is not None,
            "has_config": self._config is not None,
            "weight_components": list(self._weights.keys()),
            "additional_files": list(self._files.keys()),
            "neural_components": [c.name for c in self.get_neural_components()],
        }

        if self._manifest:
            info["model_name"] = self._manifest.get("model_name")
            info["framework_source"] = self._manifest.get("framework_source")
            info["format_version"] = self._manifest.get("format_version")

        # Count ops across all component graphs
        total_ops = 0
        for comp in self._components.values():
            if comp.graph:
                total_ops += len(comp.graph.get("ops", {}))
        info["total_ops"] = total_ops

        # Count total parameters
        total_params = 0
        if NUMPY_AVAILABLE:
            for weight_data in self._weights.values():
                tensors = weight_data.get("tensors", {})
                for tensor in tensors.values():
                    total_params += tensor.size
        info["total_parameters"] = total_params

        return info

    def __repr__(self) -> str:
        """String representation."""
        name = self._manifest.get("model_name", "unnamed") if self._manifest else "unnamed"
        return f"NBXContainer(model={name}, weights={list(self._weights.keys())})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# REMOVED: create_nbx_from_onnx - ONNX support removed from system
# NBX containers are now created directly from PyTorch models via trace
