# core/nbx/fast_loader.py
"""
Fast NBX Loader using cache + direct GPU loading.

Uses safetensors.torch.load_file(path, device="cuda") for zero-copy GPU load.

PERFORMANCE COMPARISON:
  OLD METHOD (ZIP):
    zipfile.read() -> bytes -> safetensors.load() -> CPU tensors -> .to(cuda)
    ~108 seconds for 21GB model

  NEW METHOD (DIRECT):
    safetensors.load_file(path, device="cuda") -> GPU tensors directly
    ~30 seconds estimated (3.5x faster)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from safetensors.torch import load_file as load_safetensors

from .cache import ensure_extracted

# Parallel loading workers (matches system.yml io.num_workers)
NBX_LOADER_WORKERS = 8


@dataclass
class LoadStats:
    """Loading statistics for performance tracking."""
    extraction_time: float = 0.0
    metadata_time: float = 0.0
    weights_time: float = 0.0
    total_time: float = 0.0
    weights_size_gb: float = 0.0
    throughput_gbps: float = 0.0
    shard_times: List[float] = field(default_factory=list)


class FastNBXLoader:
    """
    Fast NBX loader using extracted cache.

    Key optimization: safetensors.load_file(path, device="cuda")
    loads directly to GPU without CPU intermediate copy.

    Usage:
        loader = FastNBXLoader("model.nbx", device="cuda:0")
        weights = loader.load_weights("transformer")
        # Or load everything
        data = loader.load_all()
    """

    def __init__(self, nbx_path: Path, device: str = "cuda:0"):
        """
        Initialize fast loader.

        Args:
            nbx_path: Path to .nbx file
            device: Target device for weight loading (e.g., "cuda:0")
        """
        self.nbx_path = Path(nbx_path)
        self.device = device
        self.stats = LoadStats()

        # Ensure extracted (fast if already cached)
        t0 = time.time()
        self.cache_path = ensure_extracted(self.nbx_path)
        self.stats.extraction_time = time.time() - t0

    def load_json(self, relative_path: str) -> Dict[str, Any]:
        """Load any JSON file from cache."""
        path = self.cache_path / relative_path
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def load_manifest(self) -> Dict[str, Any]:
        """Load manifest.json."""
        return self.load_json("manifest.json")

    def load_topology(self) -> Dict[str, Any]:
        """Load topology.json (pipeline definition)."""
        # Try both names
        try:
            return self.load_json("topology.json")
        except FileNotFoundError:
            return self.load_json("pipeline.json")

    def load_runtime(self) -> Dict[str, Dict[str, Any]]:
        """Load all runtime/*.json files."""
        runtime = {}
        runtime_dir = self.cache_path / "runtime"
        if runtime_dir.exists():
            for f in runtime_dir.glob("*.json"):
                runtime[f.stem] = json.loads(f.read_text())
        return runtime

    def load_component_config(self, component: str) -> Dict[str, Any]:
        """Load component's runtime.json config."""
        config_path = self.cache_path / "components" / component / "runtime.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def load_component_graph(self, component: str) -> Dict[str, Any]:
        """Load component's graph.json (TensorDAG)."""
        return self.load_json(f"components/{component}/graph.json")

    def find_weight_files(self, component: str) -> List[Path]:
        """Find all weight files for a component."""
        # Primary location
        weights_dir = self.cache_path / "components" / component / "weights"

        if not weights_dir.exists():
            # Try alternate structures
            alt_paths = [
                self.cache_path / "weights" / component,
                self.cache_path / "weights",
            ]
            for alt in alt_paths:
                if alt.exists() and list(alt.glob("*.safetensors")):
                    weights_dir = alt
                    break

        if not weights_dir.exists():
            raise FileNotFoundError(
                f"Weights directory not found for component '{component}'.\n"
                f"Searched: {weights_dir}"
            )

        files = sorted(weights_dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(
                f"No .safetensors files found in {weights_dir}"
            )

        return files

    def load_weights(
        self,
        component: str,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load all weights for a component directly to GPU (PARALLEL).

        This is the KEY OPTIMIZATION:
        safetensors.load_file(path, device="cuda") uses mmap to load
        tensors directly to GPU memory without CPU intermediate copy.

        Args:
            component: Component name (e.g., "transformer", "vae")
            dtype: Optional dtype conversion after loading
            verbose: Print progress

        Returns:
            Dict of weight name -> tensor on GPU
        """
        weight_files = self.find_weight_files(component)

        if verbose:
            print(f"[FastLoader] Loading {len(weight_files)} shards -> {self.device}")

        all_weights = {}
        total_bytes = 0
        t0 = time.time()

        def load_shard(wf):
            """Load a single shard (for parallel execution)."""
            shard_weights = load_safetensors(str(wf), device=self.device)
            shard_bytes = sum(t.numel() * t.element_size() for t in shard_weights.values())
            if dtype is not None:
                shard_weights = {k: v.to(dtype) for k, v in shard_weights.items()}
            return wf.name, shard_weights, shard_bytes

        if len(weight_files) > 1:
            # PARALLEL LOADING
            if verbose:
                print(f"[FastLoader] Parallel loading with {NBX_LOADER_WORKERS} workers...")

            with ThreadPoolExecutor(max_workers=NBX_LOADER_WORKERS) as executor:
                futures = {executor.submit(load_shard, wf): wf for wf in weight_files}
                for future in as_completed(futures):
                    shard_name, shard_weights, shard_bytes = future.result()
                    all_weights.update(shard_weights)
                    total_bytes += shard_bytes
                    if verbose:
                        print(f"[FastLoader] {shard_name}: {shard_bytes/1e9:.2f}GB")
        else:
            # Single shard - no parallelism needed
            for wf in weight_files:
                shard_name, shard_weights, shard_bytes = load_shard(wf)
                all_weights.update(shard_weights)
                total_bytes += shard_bytes

        self.stats.weights_time = time.time() - t0
        self.stats.weights_size_gb = total_bytes / 1e9
        self.stats.throughput_gbps = (
            self.stats.weights_size_gb / max(self.stats.weights_time, 0.001)
        )

        if verbose:
            print(
                f"[FastLoader] Total: {self.stats.weights_size_gb:.2f}GB in "
                f"{self.stats.weights_time:.2f}s ({self.stats.throughput_gbps:.2f} GB/s)"
            )

        return all_weights

    def load_all_components(
        self,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load weights for all neural components.

        Returns:
            Dict of component_name -> weights dict
        """
        manifest = self.load_manifest()
        components = manifest.get("components", [])

        # Filter to neural components only
        topology = self.load_topology()
        neural_components = [
            name for name, info in topology.get("components", {}).items()
            if info.get("type") == "neural"
        ]

        all_weights = {}
        for comp in neural_components:
            if verbose:
                print(f"\n[FastLoader] === {comp} ===")
            try:
                all_weights[comp] = self.load_weights(comp, dtype=dtype, verbose=verbose)
            except FileNotFoundError as e:
                if verbose:
                    print(f"[FastLoader] Skipping {comp}: {e}")

        return all_weights

    def load_all(self) -> Dict[str, Any]:
        """
        Load complete model (metadata + weights).

        Returns:
            Dict with 'manifest', 'topology', 'runtime', 'weights', 'stats'
        """
        t_total = time.time()

        # Load metadata
        t0 = time.time()
        manifest = self.load_manifest()
        topology = self.load_topology()
        runtime = self.load_runtime()
        self.stats.metadata_time = time.time() - t0

        # Load weights for all components
        weights = self.load_all_components()

        self.stats.total_time = time.time() - t_total

        print(f"\n[FastLoader] === LOAD STATS ===")
        cached = self.stats.extraction_time < 0.1
        print(f"  Extraction: {self.stats.extraction_time:.2f}s (cached={cached})")
        print(f"  Metadata:   {self.stats.metadata_time:.2f}s")
        print(f"  Weights:    {self.stats.weights_time:.2f}s ({self.stats.weights_size_gb:.2f}GB)")
        print(f"  Throughput: {self.stats.throughput_gbps:.2f} GB/s")
        print(f"  TOTAL:      {self.stats.total_time:.2f}s")

        return {
            "manifest": manifest,
            "topology": topology,
            "runtime": runtime,
            "weights": weights,
            "stats": self.stats,
            "cache_path": self.cache_path,
        }

    def get_component_list(self) -> List[str]:
        """Get list of available components in cache."""
        components_dir = self.cache_path / "components"
        if not components_dir.exists():
            return []
        return [d.name for d in components_dir.iterdir() if d.is_dir()]


def fast_load_nbx(nbx_path: str, device: str = "cuda:0") -> Dict[str, Any]:
    """
    Convenience function for fast NBX loading.

    Args:
        nbx_path: Path to .nbx file
        device: Target GPU device

    Returns:
        Dict with manifest, topology, runtime, weights, stats
    """
    loader = FastNBXLoader(nbx_path, device)
    return loader.load_all()


def fast_load_weights(
    nbx_path: str,
    component: str,
    device: str = "cuda:0",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load just weights for a component.

    Args:
        nbx_path: Path to .nbx file
        component: Component name
        device: Target GPU device
        dtype: Optional dtype conversion

    Returns:
        Dict of weight name -> tensor
    """
    loader = FastNBXLoader(nbx_path, device)
    return loader.load_weights(component, dtype=dtype)
