"""
Weight Loader for NeuroBrix Executor.

Loads weights from .nbx files directly to GPU.
Supports sharded weights (multiple blocks).

ZERO HARDCODE: All paths from shard_map, no assumptions.
ZERO FALLBACK: Missing file = explicit error, not silent skip.
ZERO CPU BOTTLENECK: Uses cache + direct GPU loading.

PERFORMANCE OPTIMIZATION:
  OLD: ZIP → temp file → CPU RAM → GPU = slow, ~0.33 GB/s
  NEW: Cache → CPU mmap → pinned DMA (non-blocking) → batch sync = fast

NBX Support:
- Uses weights_index.json as SOURCE OF TRUTH for tensor names
- Tensors stored with NeuroTax normalized keys
- Semantic metadata available for optimization hints
"""

import os
import json
import zipfile
import warnings
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    safetensors_load_file = None  # type: ignore

# SmartLoader for pinned memory DMA transfers
try:
    from neurobrix.core.io.memory import get_io_config, PinnedMemoryManager
    HAS_SMART_IO = True
except ImportError:
    HAS_SMART_IO = False

# Import cache system
try:
    from neurobrix.nbx.cache import ensure_extracted, get_cache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False
    ensure_extracted = None  # type: ignore
    get_cache = None  # type: ignore


# Paths patterns to search for weights_index.json
WEIGHTS_INDEX_PATTERNS = [
    "components/{component}/weights_index.json",
    "{component}/weights_index.json",
    "weights/{component}/weights_index.json",
]

# PARALLEL LOADING: Number of concurrent shard loads
# safetensors loading is I/O-bound, threading helps significantly
# Default: 8 workers (optimal for NVMe, matches system.yml io.num_workers)
PARALLEL_SHARD_WORKERS = int(os.environ.get("NBX_IO_WORKERS", "8"))

# Centralized dtype conversion — single source of truth
from neurobrix.core.dtype.converter import safe_dtype_convert


class WeightLoader:
    """
    Loads model weights from .nbx container.

    ZERO HARDCODE: Device assignments from Prism shard_map.
    ZERO FALLBACK: Missing files raise explicit errors.

    NBX: Uses weights_index.json as source of truth for tensor names.

    Usage:
        loader = WeightLoader("/path/to/model.nbx")
        weights = loader.load_component("transformer", device="cuda:0")
        loader.close()
    """

    def __init__(self, nbx_path: str, use_cache: bool = True):
        """
        Initialize loader with .nbx file path.

        Args:
            nbx_path: Path to .nbx file
            use_cache: Use extracted cache for fast loading (default: True)

        Raises:
            FileNotFoundError: If nbx_path does not exist
        """
        self.nbx_path = Path(nbx_path)
        if not self.nbx_path.exists():
            raise FileNotFoundError(
                f"NBX container not found: {nbx_path}\n"
                f"Verify the model was imported correctly."
            )

        self._zip: Optional[zipfile.ZipFile] = None
        self._manifest: Optional[Dict] = None
        self._file_list_cache: Optional[List[str]] = None
        self._weights_index_cache: Dict[str, Optional[Dict]] = {}  # component → index

        # Cache system for fast loading
        self.use_cache = use_cache and HAS_CACHE
        self._cache_path: Optional[Path] = None

        if self.use_cache:
            # Extract NBX to cache (fast if already cached)
            if HAS_CACHE and ensure_extracted is not None:
                self._cache_path = ensure_extracted(self.nbx_path)

    def open(self) -> None:
        """Open the .nbx container (ZIP file or extracted directory)."""
        if self.nbx_path.is_dir():
            # Extracted cache directory — no ZIP needed
            self._zip = None
            self._cache_path = self.nbx_path
            self.use_cache = True
            self._file_list_cache = [
                str(f.relative_to(self.nbx_path))
                for f in self.nbx_path.rglob("*")
                if f.is_file()
            ]
            manifest_file = self.nbx_path / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    self._manifest = json.load(f)
            else:
                self._manifest = {"components": {}}
            return

        self._zip = zipfile.ZipFile(self.nbx_path, 'r')
        self._file_list_cache = self._zip.namelist()

        # Load manifest
        try:
            manifest_data = self._zip.read("manifest.json")
            self._manifest = json.loads(manifest_data)
        except KeyError:
            # No manifest - structure must be inferred
            self._manifest = {"components": {}}

    def close(self) -> None:
        """Close the container."""
        if self._zip:
            self._zip.close()
            self._zip = None
        self._file_list_cache = None

    def _get_file_list(self) -> List[str]:
        """Get cached file list from ZIP or cache directory."""
        if self._file_list_cache is None:
            if self.use_cache and self._cache_path:
                # Get file list from cache directory
                self._file_list_cache = [
                    str(f.relative_to(self._cache_path))
                    for f in self._cache_path.rglob("*")
                    if f.is_file()
                ]
            else:
                # Get file list from ZIP
                if self._zip is None:
                    self.open()
                assert self._zip is not None
                self._file_list_cache = self._zip.namelist()
        return self._file_list_cache

    def _find_weight_files(self, component_name: str) -> List[str]:
        """
        Find all weight files for a component.

        ZERO HARDCODE: Searches multiple path patterns.

        Args:
            component_name: Component name

        Returns:
            List of weight file paths in ZIP
        """
        file_list = self._get_file_list()

        # Path patterns to try (ordered by priority)
        patterns = [
            f"components/{component_name}/weights/",
            f"{component_name}/weights/",
            f"weights/{component_name}/",
        ]

        for pattern in patterns:
            found = [
                name for name in file_list
                if name.startswith(pattern) and
                (name.endswith(".safetensors") or name.endswith(".bin"))
            ]
            if found:
                return sorted(found)

        return []

    # ════════════════════════════════════════════════════════════════════════
    # NBX: weights_index.json support
    # ════════════════════════════════════════════════════════════════════════

    def _find_weights_index_path(self, component_name: str) -> Optional[str]:
        """
        Find weights_index.json path for a component.

        ZERO HARDCODE: Tries multiple path patterns from WEIGHTS_INDEX_PATTERNS.

        Args:
            component_name: Component name

        Returns:
            Path in ZIP if found, None otherwise
        """
        file_list = self._get_file_list()

        for pattern in WEIGHTS_INDEX_PATTERNS:
            path = pattern.format(component=component_name)
            if path in file_list:
                return path

        return None

    def _load_weights_index(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Load weights_index.json for a component.

        NBX Format:
        {
            "version": "0.1.0",
            "tensors": {
                "block.0.attn.query.weight": {
                    "shard": "shard_000.safetensors",
                    "dtype": "bfloat16",
                    "shape": [3072, 3072],
                    "size_bytes": 18874368,
                    "semantic": {...},
                    "hints": {...}
                }
            }
        }

        Args:
            component_name: Component name

        Returns:
            Parsed weights_index dict or None if not found
        """
        # Check cache first
        if component_name in self._weights_index_cache:
            return self._weights_index_cache[component_name]

        index_path = self._find_weights_index_path(component_name)
        if not index_path:
            self._weights_index_cache[component_name] = None
            return None

        try:
            # FAST PATH: Load from cache directory
            if self.use_cache and self._cache_path:
                cache_file = self._cache_path / index_path
                if cache_file.exists():
                    with open(cache_file) as f:
                        index = json.load(f)
                    self._weights_index_cache[component_name] = index
                    return index

            # SLOW PATH: Load from ZIP
            if not self._zip:
                self.open()
            assert self._zip is not None
            data = self._zip.read(index_path)
            index = json.loads(data)
            self._weights_index_cache[component_name] = index
            return index
        except (KeyError, json.JSONDecodeError, FileNotFoundError) as e:
            warnings.warn(f"[WeightLoader] Failed to load weights_index.json: {e}")
            self._weights_index_cache[component_name] = None
            return None

    def _load_neurotax_map(self, component_name: str) -> Optional[Dict[str, str]]:
        """
        Load neurotax_map.json and build reverse mapping.

        neurotax_map.json format:
        {
            "mappings": {
                "hf_name": "neurotax_name",
                ...
            }
        }

        Returns:
            Dict of HF_name → NeuroTax_name (forward mapping)
            or None if not found
        """
        # Path patterns for neurotax_map.json
        patterns = [
            f"components/{component_name}/neurotax_map.json",
            f"{component_name}/neurotax_map.json",
        ]

        for pattern in patterns:
            if pattern in self._get_file_list():
                try:
                    # FAST PATH: Load from cache
                    if self.use_cache and self._cache_path:
                        cache_file = self._cache_path / pattern
                        if cache_file.exists():
                            with open(cache_file) as f:
                                data = json.load(f)
                            if "mappings" in data:
                                return data["mappings"]

                    # SLOW PATH: Load from ZIP
                    if not self._zip:
                        self.open()
                    assert self._zip is not None
                    data = json.loads(self._zip.read(pattern))
                    if "mappings" in data:
                        return data["mappings"]
                except (KeyError, json.JSONDecodeError, FileNotFoundError):
                    pass

        return None

    def _build_weight_aliases(
        self,
        weights: Dict[str, torch.Tensor],
        neurotax_map: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """
        Add HuggingFace aliases to weights dict.

        Graph.json uses HF names in parent_module, but weights are stored
        with NeuroTax names. This creates aliases so both lookups work.

        Args:
            weights: Dict with NeuroTax keys
            neurotax_map: HF_name → NeuroTax_name mapping

        Returns:
            weights dict with both HF and NeuroTax keys
        """
        # Build reverse map: NeuroTax → HF
        reverse_map = {v: k for k, v in neurotax_map.items()}

        aliases_added = 0
        for neurotax_name, tensor in list(weights.items()):
            hf_name = reverse_map.get(neurotax_name)
            if hf_name and hf_name not in weights:
                weights[hf_name] = tensor
                aliases_added += 1

        return weights

    def load_component(
        self,
        component_name: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load all weights for a component to a single device.

        ARCHITECTURE:
        - Weights stored as fp32 (preserves precision)
        - Prism determines optimal dtype for hardware
        - Weights converted to Prism dtype during load

        Args:
            component_name: Name of component (e.g., "transformer", "vae")
            device: Target device (e.g., "cuda:0", "hip:0", "xpu:0")
            dtype: Target dtype from Prism (e.g., torch.float16, torch.bfloat16)

        Returns:
            Dict of weight name -> tensor (with Prism-specified dtype)

        Raises:
            RuntimeError: If no weight files found
        """
        if not self._zip:
            self.open()

        weight_files = self._find_weight_files(component_name)

        if not weight_files:
            raise RuntimeError(
                f"No weight files found for component '{component_name}'.\n"
                f"Expected path: components/{component_name}/weights/*.safetensors\n"
                f"Available in container: {self._get_file_list()[:10]}..."
            )

        weights = {}

        # PARALLEL LOADING: Load multiple shards concurrently
        if len(weight_files) > 1 and PARALLEL_SHARD_WORKERS > 1:
            def load_shard(weight_file):
                """Load a single shard (runs in thread)."""
                shard_name = weight_file.split("/")[-1]
                return shard_name, self._load_weight_file(weight_file, device, dtype)

            with ThreadPoolExecutor(max_workers=PARALLEL_SHARD_WORKERS) as executor:
                futures = {executor.submit(load_shard, wf): wf for wf in weight_files}
                for future in as_completed(futures):
                    shard_name, block_weights = future.result()
                    weights.update(block_weights)
        else:
            # Sequential fallback for single shard
            for weight_file in weight_files:
                block_weights = self._load_weight_file(weight_file, device, dtype)
                weights.update(block_weights)

        # Batch sync: ensure all GPU transfers complete before returning
        if device.startswith("cuda") or device.startswith("hip"):
            torch.cuda.synchronize()

        # NBX: Add HF aliases if neurotax_map exists
        neurotax_map = self._load_neurotax_map(component_name)
        if neurotax_map:
            weights = self._build_weight_aliases(weights, neurotax_map)

        return weights

    def _is_fgp_shard_map(self, shard_map: Dict[str, str]) -> bool:
        """
        Detect if shard_map is FGP-style (key patterns) or file-style (file paths).

        FGP shard_map format: {"block.0.*": "cuda:0", "non_block": "cuda:1", ...}
        File shard_map format: {"shard_000.safetensors": "cuda:0", ...}
        """
        if not shard_map:
            return False

        # Check first few keys
        sample_keys = list(shard_map.keys())[:3]
        for key in sample_keys:
            # File-style: has file extension
            if key.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                return False
            # FGP-style: NOT a file path → it's a weight key pattern
            # Matches: "block.0.*", "non_block", "encoder.block.0.*", etc.
            if not ('/' in key or '\\' in key):
                return True

        return False

    def _match_fgp_key(self, weight_key: str, shard_map: Dict[str, str]) -> Optional[str]:
        """
        Match a weight key to the correct device using FGP shard_map.

        FGP shard_map contains actual weight keys mapped to devices:
        {"block.0.cross_attn.key.weight": "cuda:0", "time_embed.proj.bias": "cuda:0", ...}

        Args:
            weight_key: Weight key like "block.0.cross_attn.key.weight"
            shard_map: FGP key map {weight_key: device}

        Returns:
            Device string for this weight, or None if key not in this stage's map
        """
        # Direct lookup - FGP shard_map contains actual weight keys
        return shard_map.get(weight_key)

    def _load_component_fgp(
        self,
        component_name: str,
        shard_map: Dict[str, str],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights using FGP (Fine-Grained Pipeline) mode with DIRECT GPU loading.

        ZERO CPU INTERMEDIATE: For each weight file, determine which device its
        contents belong to, and load directly to that GPU using safetensors mmap.
        This avoids 64GB CPU memory peak on MoE models.

        Strategy for mixed-device files:
        - If all keys in a file go to same device -> load directly to GPU
        - If keys go to different devices -> load to CPU, route per-key

        Args:
            component_name: Component name
            shard_map: FGP key map {"model.layers.0.self_attn.q_proj.weight": "cuda:0", ...}
            dtype: Target dtype from Prism

        Returns:
            Dict of weight name -> tensor (on correct device, with Prism dtype)
        """
        import gc

        file_list = self._get_file_list()

        # Find all weight files for this component
        weight_files = []
        weight_paths = [
            f"components/{component_name}/weights/",
            f"{component_name}/weights/",
            f"weights/{component_name}/",
        ]

        for f in file_list:
            if f.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                for prefix in weight_paths:
                    if f.startswith(prefix):
                        weight_files.append(f)
                        break

        if not weight_files:
            raise FileNotFoundError(
                f"No weight files found for component '{component_name}'.\n"
                f"Searched paths: {weight_paths}\n"
                f"Available files: {file_list[:10]}..."
            )

        # Collect unique devices + determine default device for non-block weights
        devices = set(shard_map.values())
        default_device = sorted(devices)[0] if devices else "cuda:0"

        # ── Phase 1: Pre-read metadata to classify files ──
        # Build {file → {key → device}} map and classify as direct/mixed/unknown
        file_meta = {}  # weight_file → ("direct", target_device) | ("mixed", {key: device}) | ("unknown",)
        for weight_file in sorted(weight_files):
            file_keys = None
            if self.use_cache and self._cache_path:
                cache_file = self._cache_path / weight_file
                if cache_file.exists() and str(cache_file).endswith('.safetensors'):
                    try:
                        from safetensors import safe_open
                        with safe_open(str(cache_file), framework="pt") as f:
                            file_keys = list(f.keys())
                    except Exception:
                        pass

            if file_keys:
                key_devices = {}
                for key in file_keys:
                    target = self._match_fgp_key(key, shard_map)
                    key_devices[key] = target if target else default_device
                unique_devices = set(key_devices.values())
                if len(unique_devices) == 1:
                    file_meta[weight_file] = ("direct", list(unique_devices)[0])
                else:
                    file_meta[weight_file] = ("mixed", key_devices)
            else:
                file_meta[weight_file] = ("unknown",)

        # ── Phase 2: Parallel file loading ──
        weights = {}

        def _load_one_fgp_file(weight_file):
            """Load a single FGP file (runs in thread). Returns list of (key, tensor) pairs."""
            shard_name = weight_file.split("/")[-1]
            meta = file_meta[weight_file]
            results = []

            if meta[0] == "direct":
                target_device = meta[1]
                block_weights = self._load_weight_file(weight_file, target_device, dtype)
                for key, tensor in block_weights.items():
                    results.append((key, tensor))

            elif meta[0] == "mixed":
                key_devices = meta[1]
                block_weights = self._load_weight_file(weight_file, "cpu", None)
                for key, tensor in list(block_weights.items()):
                    target_device = key_devices.get(key, default_device)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = safe_dtype_convert(tensor, dtype)
                    tensor = tensor.to(target_device)
                    results.append((key, tensor))
                    del block_weights[key]
                del block_weights

            else:
                # unknown - CPU load + route
                block_weights = self._load_weight_file(weight_file, "cpu", None)
                for key, tensor in list(block_weights.items()):
                    target_device = self._match_fgp_key(key, shard_map) or default_device
                    if dtype is not None and tensor.is_floating_point():
                        tensor = safe_dtype_convert(tensor, dtype)
                    tensor = tensor.to(target_device)
                    results.append((key, tensor))
                    del block_weights[key]
                del block_weights

            return results

        num_workers = min(PARALLEL_SHARD_WORKERS, len(weight_files))
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_load_one_fgp_file, wf): wf for wf in sorted(weight_files)}
                for future in as_completed(futures):
                    for key, tensor in future.result():
                        weights[key] = tensor
        else:
            for wf in sorted(weight_files):
                for key, tensor in _load_one_fgp_file(wf):
                    weights[key] = tensor

        gc.collect()

        # NBX: Add HF aliases if neurotax_map exists
        neurotax_map = self._load_neurotax_map(component_name)
        if neurotax_map:
            weights = self._build_weight_aliases(weights, neurotax_map)

        return weights

    def load_component_with_shard_map(
        self,
        component_name: str,
        shard_map: Dict[str, str],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights respecting Prism shard_map.

        CRITICAL: Loads ONLY files present in shard_map (standard mode).
        PipelineExecutor passes partial shard_maps (one stage at a time).

        FGP MODE: When shard_map contains key patterns (block.N.*),
        loads all files and routes per-key based on pattern matching.

        ARCHITECTURE (ZERO HARDCODE):
        - Weights stored as fp32 in NBX (preserves precision)
        - Prism determines optimal dtype from hardware profile (supports_dtypes)
        - Weights converted to Prism dtype during load

        ZERO FALLBACK: File not found = explicit error with details.

        Args:
            component_name: Name of component (for error messages)
            shard_map: Prism shard_map {zip_path: device} - ONLY these loaded
            dtype: Target dtype from Prism (embedded from hardware profile)

        Returns:
            Dict of weight name -> tensor (on correct device, with Prism dtype)

        Raises:
            RuntimeError: If shard_map is empty
            FileNotFoundError: If any shard file not found in container
        """
        if not self._zip:
            self.open()

        # === ZERO FALLBACK: Empty shard_map is an error ===
        if not shard_map:
            raise RuntimeError(
                f"Empty shard_map for component '{component_name}'.\n"
                f"Prism must provide device assignments for all shards.\n"
                f"Check that Prism allocation completed successfully."
            )

        # === FGP MODE: Detect key-pattern shard_map ===
        if self._is_fgp_shard_map(shard_map):
            return self._load_component_fgp(component_name, shard_map, dtype)

        # === STANDARD MODE: File-path shard_map ===
        weights = {}
        file_list = self._get_file_list()

        # Resolve all paths first
        shard_items = []
        for shard_path, device in sorted(shard_map.items()):
            actual_path = self._resolve_shard_path(shard_path, component_name, file_list)
            shard_items.append((actual_path, device))

        # === CRITICAL: Load ONLY files in shard_map ===
        # PipelineExecutor passes partial maps (one stage's shards only)
        if len(shard_items) > 1 and PARALLEL_SHARD_WORKERS > 1:
            def load_shard(item):
                """Load a single shard (runs in thread)."""
                actual_path, device = item
                shard_name = actual_path.split("/")[-1]
                return shard_name, device, self._load_weight_file(actual_path, device, dtype)

            with ThreadPoolExecutor(max_workers=PARALLEL_SHARD_WORKERS) as executor:
                futures = {executor.submit(load_shard, item): item for item in shard_items}
                for future in as_completed(futures):
                    shard_name, device, block_weights = future.result()
                    weights.update(block_weights)
        else:
            # Sequential fallback for single shard
            for actual_path, device in shard_items:
                block_weights = self._load_weight_file(actual_path, device, dtype)
                weights.update(block_weights)

        # Batch sync: ensure all GPU transfers complete before returning
        devices_used = set(d for _, d in shard_items)
        if any(d.startswith("cuda") or d.startswith("hip") for d in devices_used):
            torch.cuda.synchronize()

        # NBX: Add HF aliases if neurotax_map exists
        neurotax_map = self._load_neurotax_map(component_name)
        if neurotax_map:
            weights = self._build_weight_aliases(weights, neurotax_map)

        return weights

    def _resolve_shard_path(
        self,
        shard_path: str,
        component_name: str,
        file_list: List[str],
    ) -> str:
        """
        Resolve shard path to actual path in container.

        ZERO HARDCODE: Tries multiple path formats from shard_path.
        ZERO FALLBACK: Not found = explicit error with suggestions.

        Args:
            shard_path: Path from shard_map (may be full or partial)
            component_name: Component name for building alternate paths
            file_list: List of files in container

        Returns:
            Actual path in container

        Raises:
            FileNotFoundError: If shard not found in container
        """
        # Try exact path first
        if shard_path in file_list:
            return shard_path

        # Extract filename from path
        shard_filename = shard_path.split("/")[-1]

        # Build alternate paths to try
        alternate_paths = [
            f"components/{component_name}/weights/{shard_filename}",
            f"{component_name}/weights/{shard_filename}",
            f"weights/{component_name}/{shard_filename}",
            f"weights/{shard_filename}",
        ]

        for alt_path in alternate_paths:
            if alt_path in file_list:
                return alt_path

        # === ZERO FALLBACK: Explicit error with helpful info ===
        # Find similar files for suggestion
        similar_files = [
            f for f in file_list 
            if shard_filename in f or component_name in f
        ][:5]

        raise FileNotFoundError(
            f"Shard not found in NBX container:\n"
            f"  Requested: {shard_path}\n"
            f"  Component: {component_name}\n"
            f"  Tried paths:\n"
            f"    - {shard_path}\n" +
            "\n".join(f"    - {p}" for p in alternate_paths) +
            f"\n  Similar files in container: {similar_files}\n"
            f"  Verify Prism shard_map matches NBX structure."
        )

    def _load_weight_file(
        self,
        zip_path: str,
        device: str,
        dtype: Optional[torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        """
        Load a single weight file from the container.

        ARCHITECTURE (ZERO HARDCODE):
        - Format detected from file extension
        - dtype from Prism (embedded from hardware profile)

        Args:
            zip_path: Path within ZIP container
            device: Target device
            dtype: Target dtype from Prism

        Returns:
            Dict of weight name -> tensor (with Prism dtype)

        Raises:
            RuntimeError: If file format not supported
        """
        if zip_path.endswith(".safetensors"):
            return self._load_safetensor_block(zip_path, device, dtype)
        elif zip_path.endswith(".bin"):
            return self._load_pytorch_block(zip_path, device, dtype)
        else:
            # === ZERO FALLBACK: Unknown format is error ===
            raise RuntimeError(
                f"Unsupported weight file format: {zip_path}\n"
                f"Supported formats: .safetensors, .bin"
            )

    def _load_safetensor_block(
        self,
        zip_path: str,
        device: str,
        dtype: Optional[torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        """
        Load a single safetensor block with high-performance pinned memory DMA.

        ARCHITECTURE (ZERO HARDCODE):
        - Weights stored in original dtype in NBX
        - Prism determines target dtype from hardware profile (supports_dtypes)
        - Convert dtype on CPU (fast) then DMA transfer to GPU (16GB/s)

        OPTIMIZATION (CPU Conversion + Pinned Memory DMA):
        - Load to CPU first (mmap, fast)
        - Convert dtype on CPU (uses AVX/SSE, much faster than GPU for bf16->fp32)
        - Pin memory for PCIe DMA transfer (16GB/s vs 8GB/s regular copy)
        - Non-blocking transfer to GPU

        This approach is 4-5x faster for bf16 models on V100 (no native bf16).
        """
        if not HAS_SAFETENSORS:
            raise ImportError(
                "safetensors library required for weight loading.\n"
                "Install with: pip install safetensors"
            )

        # Determine if we're loading to GPU
        is_cuda = device.startswith("cuda") or device.startswith("hip")

        # FAST PATH: Load from cache (no ZIP extraction needed)
        if self.use_cache and self._cache_path:
            cache_file = self._cache_path / zip_path
            if cache_file.exists():
                return self._load_with_pinned_dma(str(cache_file), device, dtype, is_cuda)

        # FALLBACK PATH: Extract from ZIP to temp file
        if self._zip is None:
            self.open()
        assert self._zip is not None

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp.write(self._zip.read(zip_path))
            tmp_path = tmp.name

        try:
            return self._load_with_pinned_dma(tmp_path, device, dtype, is_cuda)
        finally:
            os.unlink(tmp_path)

    def _load_with_pinned_dma(
        self,
        file_path: str,
        device: str,
        dtype: Optional[torch.dtype],
        is_cuda: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights using CPU mmap + pinned memory DMA transfer.

        Architecture:
        1. CPU load via safetensors mmap (single large read, ~10 GB/s)
        2. CPU dtype conversion (AVX/SSE vectorized, fast for bf16→fp16)
        3. Pinned memory non-blocking DMA to GPU (~12 GB/s PCIe Gen3)
        4. Synchronization handled by caller (batch sync after all shards)

        This outperforms safetensors direct GPU loading because:
        - Single mmap read vs many small per-tensor PCIe transfers
        - CPU AVX conversion faster than V100 GPU bf16→fp16
        - Non-blocking DMA enables overlap between shards
        """
        if not HAS_SAFETENSORS or safetensors_load_file is None:
            raise ImportError("safetensors library is required")

        # Step 1: CPU load via mmap (single large read)
        weights = safetensors_load_file(file_path, device="cpu")

        # Step 2: CPU dtype conversion (AVX/SSE, fast for bf16→fp16)
        if dtype is not None:
            weights = self._convert_weights_dtype(weights, dtype)

        # Step 3: Non-blocking DMA to GPU (caller handles sync)
        if is_cuda:
            weights = self._transfer_with_pinned_memory(weights, device)

        return weights

    def _transfer_with_pinned_memory(
        self,
        weights: Dict[str, torch.Tensor],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer CPU tensors to GPU using pinned memory for maximum throughput.

        Pinned memory enables DMA (Direct Memory Access) transfers at ~16GB/s
        compared to ~8GB/s for regular CPU->GPU copies.
        """
        result = {}
        for name, tensor in weights.items():
            if tensor.is_cuda:
                # Already on GPU (shouldn't happen, but handle gracefully)
                result[name] = tensor
            else:
                # Pin memory and transfer with non-blocking DMA
                pinned = tensor.pin_memory()
                result[name] = pinned.to(device, non_blocking=True)

        # NOTE: No synchronize here — caller handles batch sync after all shards loaded.
        # Per-shard sync was the main bottleneck (prevented overlapped transfers).
        return result

    def _load_pytorch_block(
        self,
        zip_path: str,
        device: str,
        dtype: Optional[torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        """
        Load a PyTorch .bin file with high-performance pinned memory DMA.

        ARCHITECTURE (ZERO HARDCODE):
        - Weights stored in original dtype in NBX
        - Prism determines target dtype from hardware profile (supports_dtypes)
        - Convert dtype on CPU (fast) then DMA transfer to GPU (16GB/s)

        OPTIMIZATION (CPU Conversion + Pinned Memory DMA):
        Same approach as safetensors - load to CPU, convert, pin, transfer.
        """
        # Determine if we're loading to GPU
        is_cuda = device.startswith("cuda") or device.startswith("hip")

        # FAST PATH: Load from cache
        if self.use_cache and self._cache_path:
            cache_file = self._cache_path / zip_path
            if cache_file.exists():
                return self._load_pytorch_with_pinned_dma(str(cache_file), device, dtype, is_cuda)

        # FALLBACK PATH: Extract from ZIP
        if self._zip is None:
            self.open()
        assert self._zip is not None

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(self._zip.read(zip_path))
            tmp_path = tmp.name

        try:
            return self._load_pytorch_with_pinned_dma(tmp_path, device, dtype, is_cuda)
        finally:
            os.unlink(tmp_path)

    def _load_pytorch_with_pinned_dma(
        self,
        file_path: str,
        device: str,
        dtype: Optional[torch.dtype],
        is_cuda: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Load PyTorch weights using CPU conversion + pinned memory DMA transfer.
        """
        # Step 1: Load to CPU
        weights = torch.load(file_path, map_location="cpu", weights_only=True)

        # Step 2: Convert dtype on CPU (fast - uses AVX/SSE vectorization)
        if dtype is not None:
            weights = self._convert_weights_dtype(weights, dtype)

        # Step 3: Transfer to GPU with pinned memory DMA
        if is_cuda:
            weights = self._transfer_with_pinned_memory(weights, device)

        return weights

    def _convert_weights_dtype(
        self,
        weights: Dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert weights to target dtype (from Prism hardware profile).

        ARCHITECTURE (ZERO HARDCODE):
        - Only floating-point tensors are converted
        - Integer tensors (indices, attention masks) are preserved
        - bfloat16 → float16 uses safe clamping to avoid overflow
        """
        # Early exit: check if first floating-point tensor already matches
        for t in weights.values():
            if t.is_floating_point():
                if t.dtype == dtype:
                    return weights  # No conversion needed
                break  # Needs conversion

        converted = {}
        for name, tensor in weights.items():
            if tensor.is_floating_point() and tensor.dtype != dtype:
                converted[name] = safe_dtype_convert(tensor, dtype)
            else:
                converted[name] = tensor
        return converted

    def get_graph(self, component_name: str) -> Optional[Dict]:
        """
        Load graph.json (TensorDAG) for a component.

        NOTE: Prefer using NBXContainer.get_neural_components() instead.
        The graph is already loaded when NBXContainer.load() is called.

        Args:
            component_name: Component name

        Returns:
            TensorDAG dict or None if not found
        """
        if not self._zip:
            self.open()

        patterns = [
            f"components/{component_name}/graph.json",
            f"{component_name}/graph.json",
        ]

        for pattern in patterns:
            try:
                assert self._zip is not None
                data = self._zip.read(pattern)
                return json.loads(data)
            except KeyError:
                continue

        return None

    def list_files(self) -> List[str]:
        """List all files in the container."""
        return self._get_file_list()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
