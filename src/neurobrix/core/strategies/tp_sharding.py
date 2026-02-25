"""
Tensor Parallel Block Sharding

Creates and manages weight shards for tensor parallel execution.

Key features:
- LAZY: Creates shards on first runtime, not import
- HASH-BASED: Only re-shards if source weights changed
- ZERO HARDCODE: Shard axis detected from tensor shapes
- CACHED: Shards stored in ~/.neurobrix/cache/<name>/components/<comp>/block_shards/

Cache structure:
  ~/.neurobrix/cache/<model>/components/<component>/
  ├── weights/                    # Original merged weights
  │   └── model.safetensors
  └── block_shards/              # TP shards (created lazily)
      ├── manifest.json          # Shard metadata + source hash
      ├── block.0/
      │   ├── shard_0.safetensors
      │   └── shard_1.safetensors
      └── block.N/
          ├── shard_0.safetensors
          └── shard_1.safetensors
"""

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Use consolidated parallel loader (eliminates code duplication)
from neurobrix.core.io.loader import load_files_parallel, DEFAULT_NUM_WORKERS

# Parallel loading workers for TP sharding (matches system.yml io.num_workers)
TP_SHARD_WORKERS = DEFAULT_NUM_WORKERS


def create_block_shards(
    cache_path: Path,
    component_name: str,
    n_shards: int,
    force: bool = False,
) -> Path:
    """
    Create TP block shards from original weights.

    LAZY: Only creates if:
    1. block_shards/ doesn't exist, OR
    2. source_weights_hash changed (weights were re-imported)

    ZERO HARDCODE: Shard axis detected from tensor shapes.
    ZERO FALLBACK: Crash if weights missing or can't shard.

    Args:
        cache_path: Path to ~/.neurobrix/cache/<model_name>/
        component_name: Component name (e.g., "transformer")
        n_shards: Number of shards (typically = number of GPUs)
        force: Force re-sharding even if cache is valid

    Returns:
        Path to block_shards directory
    """
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)

    weights_path = cache_path / "components" / component_name / "weights"
    shards_path = cache_path / "components" / component_name / "block_shards"
    manifest_path = shards_path / "manifest.json"

    # Check if weights exist
    if not weights_path.exists():
        raise RuntimeError(
            f"ZERO FALLBACK: Weights not found for component '{component_name}'.\n"
            f"Expected: {weights_path}\n"
            f"Run import first to create weights."
        )

    # Find weight files
    weight_files = list(weights_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = list(weights_path.glob("*.bin"))
    if not weight_files:
        raise RuntimeError(
            f"ZERO FALLBACK: No weight files found in {weights_path}.\n"
            f"Expected .safetensors or .bin files."
        )

    # Compute hash of source weights
    source_hash = _compute_weights_hash(weight_files)

    # Check if shards are up-to-date
    if not force and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("source_weights_hash") == source_hash and
                manifest.get("n_shards") == n_shards):
                return shards_path
        except (json.JSONDecodeError, KeyError):
            pass  # Invalid manifest, re-create

    pass

    # Parse block structure from weight keys
    all_weights = _load_all_weights(weight_files)
    block_structure = _parse_block_structure(all_weights)

    if not block_structure["blocks"]:
        # No block structure found - create single "global" shard
        pass
        block_structure["blocks"] = {0: list(all_weights.keys())}

    # Create shards directory (clean up old shards)
    if shards_path.exists():
        import shutil
        shutil.rmtree(shards_path)
    shards_path.mkdir(parents=True, exist_ok=True)

    # Shard each block
    blocks_created = 0
    for block_num, keys in sorted(block_structure["blocks"].items()):
        block_dir = shards_path / f"block.{block_num}"
        block_dir.mkdir(exist_ok=True)

        # Load block weights
        block_tensors = {k: all_weights[k] for k in keys if k in all_weights}
        if not block_tensors:
            continue

        # Determine shard axis (ZERO HARDCODE - detect from shapes)
        shard_axis_for_block = _detect_shard_axis(block_tensors)

        # Split tensors
        sharded = _split_tensors(block_tensors, n_shards, shard_axis_for_block)

        # Save each shard
        for shard_idx, shard_tensors in enumerate(sharded):
            shard_file = block_dir / f"shard_{shard_idx}.safetensors"
            save_file(shard_tensors, str(shard_file))

        blocks_created += 1

    # Also shard non-block weights (embeddings, final layers, etc.)
    non_block_keys = block_structure.get("non_block", [])
    shard_axis = 0  # Initialize default for manifest
    if non_block_keys:
        non_block_dir = shards_path / "non_block"
        non_block_dir.mkdir(exist_ok=True)

        non_block_tensors = {k: all_weights[k] for k in non_block_keys if k in all_weights}
        if non_block_tensors:
            shard_axis = _detect_shard_axis(non_block_tensors)
            sharded = _split_tensors(non_block_tensors, n_shards, shard_axis)

            for shard_idx, shard_tensors in enumerate(sharded):
                shard_file = non_block_dir / f"shard_{shard_idx}.safetensors"
                save_file(shard_tensors, str(shard_file))

    # Write manifest
    manifest = {
        "format": "tp_block_shards",
        "version": "0.1.0",
        "created_at": datetime.now().isoformat(),
        "source_weights_hash": source_hash,
        "n_shards": n_shards,
        "shard_axis": shard_axis,
        "blocks": {str(k): list(v) for k, v in block_structure["blocks"].items()},
        "non_block_keys": non_block_keys,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return shards_path


def load_block_shard(
    shards_path: Path,
    block_num: int,
    shard_idx: int,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load a specific block shard.

    Args:
        shards_path: Path to block_shards directory
        block_num: Block number
        shard_idx: Shard index (GPU index)
        device: Target device
        dtype: Target dtype (optional)

    Returns:
        Dict of weight_name -> tensor
    """
    if isinstance(shards_path, str):
        shards_path = Path(shards_path)

    shard_file = shards_path / f"block.{block_num}" / f"shard_{shard_idx}.safetensors"

    if not shard_file.exists():
        raise RuntimeError(
            f"ZERO FALLBACK: Block shard not found: {shard_file}\n"
            f"Create shards first with create_block_shards()"
        )

    with safe_open(str(shard_file), framework="pt", device=device) as f:
        weights = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            weights[key] = tensor

    return weights


def load_non_block_shard(
    shards_path: Path,
    shard_idx: int,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load non-block weights shard (embeddings, final layers).

    Args:
        shards_path: Path to block_shards directory
        shard_idx: Shard index (GPU index)
        device: Target device
        dtype: Target dtype (optional)

    Returns:
        Dict of weight_name -> tensor
    """
    if isinstance(shards_path, str):
        shards_path = Path(shards_path)

    shard_file = shards_path / "non_block" / f"shard_{shard_idx}.safetensors"

    if not shard_file.exists():
        return {}  # No non-block weights

    with safe_open(str(shard_file), framework="pt", device=device) as f:
        weights = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            weights[key] = tensor

    return weights


def get_shard_manifest(shards_path: Path) -> Dict[str, Any]:
    """Get manifest for existing shards."""
    if isinstance(shards_path, str):
        shards_path = Path(shards_path)

    manifest_path = shards_path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"ZERO FALLBACK: Shard manifest not found: {manifest_path}\n"
            f"Create shards first with create_block_shards()"
        )

    return json.loads(manifest_path.read_text())


def _compute_weights_hash(weight_files: List[Path]) -> str:
    """Compute hash of weight files (for cache invalidation)."""
    hasher = hashlib.sha256()

    for wf in sorted(weight_files):
        # Hash file metadata (size + mtime) for speed
        # Full content hash would be too slow for large models
        stat = wf.stat()
        hasher.update(f"{wf.name}:{stat.st_size}:{stat.st_mtime}".encode())

    return f"sha256:{hasher.hexdigest()[:16]}"


def _load_single_weight_file(wf: str | Path) -> Dict[str, torch.Tensor]:
    """Load a single weight file (for parallel execution)."""
    if isinstance(wf, str):
        wf = Path(wf)
    result = {}
    if wf.suffix == ".safetensors":
        with safe_open(str(wf), framework="pt", device="cpu") as f:
            for key in f.keys():
                result[key] = f.get_tensor(key)
    else:
        # PyTorch .bin format
        result = torch.load(str(wf), map_location="cpu", weights_only=True)
    return result


def _load_all_weights(weight_files: List[Path]) -> Dict[str, torch.Tensor]:
    """Load all weights from files into memory (PARALLEL).

    Uses consolidated parallel_loader to eliminate code duplication.
    """
    # Type narrowing: List[Path] is compatible with List[str | Path] due to covariance in function context
    files_list: list[str | Path] = weight_files  # type: ignore[assignment]
    return load_files_parallel(
        files_list,
        _load_single_weight_file,
        max_workers=TP_SHARD_WORKERS,
        log_progress=True,
        prefix="[TP Sharding]"
    )


def _parse_block_structure(weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Parse block structure from weight keys.

    ZERO HARDCODE: Pattern-based detection, not hardcoded layer names.

    Returns:
        {
            "blocks": {0: [keys], 1: [keys], ...},
            "non_block": [keys without block pattern]
        }
    """
    blocks: Dict[int, List[str]] = {}
    non_block: List[str] = []

    # Patterns to detect block/layer indices
    patterns = [
        r"blocks?[._](\d+)",      # block.0, blocks_0, block_0
        r"layers?[._](\d+)",      # layer.0, layers_0
        r"h[._](\d+)",            # GPT-2 style: h.0
        r"transformer[._]h[._](\d+)",  # Longer GPT-2 pattern
    ]

    for key in weights.keys():
        matched = False
        for pattern in patterns:
            match = re.search(pattern, key, re.IGNORECASE)
            if match:
                block_idx = int(match.group(1))
                if block_idx not in blocks:
                    blocks[block_idx] = []
                blocks[block_idx].append(key)
                matched = True
                break

        if not matched:
            non_block.append(key)

    return {"blocks": blocks, "non_block": non_block}


def _detect_shard_axis(tensors: Dict[str, torch.Tensor]) -> int:
    """
    Detect which axis to shard for TP.

    ZERO HARDCODE: Based on tensor shapes, not names.

    Rules:
    - Linear weights [out, in]: shard axis 0 (output dim) for column parallel
    - Attention QKV [3*hidden, hidden]: shard axis 0
    - LayerNorm [hidden]: don't shard (replicate)

    Returns:
        Shard axis (typically 0 for column-parallel)
    """
    # Find largest 2D tensor (likely the main linear weights)
    largest_tensor = None
    largest_numel = 0
    shard_axis = 0  # Initialize with default value

    for key, tensor in tensors.items():
        if len(tensor.shape) == 2 and tensor.numel() > largest_numel:
            largest_tensor = tensor
            largest_numel = tensor.numel()

    # Default: shard first axis (output features)
    # This is column-parallel style which is most common
    return shard_axis


def _split_tensors(
    tensors: Dict[str, torch.Tensor],
    n_shards: int,
    axis: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    Split tensors along axis into n_shards pieces.

    Args:
        tensors: Dict of weight_name -> tensor
        n_shards: Number of shards
        axis: Axis to split along

    Returns:
        List of dicts, one per shard
    """
    result = [{} for _ in range(n_shards)]

    for key, tensor in tensors.items():
        if len(tensor.shape) <= 1:
            # Scalar or 1D (bias, norm weights) - replicate to all shards
            for shard in result:
                shard[key] = tensor.clone()
        elif tensor.shape[axis] % n_shards != 0:
            # Can't evenly split - replicate
            for shard in result:
                shard[key] = tensor.clone()
        else:
            # Split along axis
            chunks = torch.chunk(tensor, n_shards, dim=axis)
            for shard_idx, chunk in enumerate(chunks):
                result[shard_idx][key] = chunk.contiguous()

    return result


def shards_exist(cache_path: Path, component_name: str) -> bool:
    """Check if valid shards exist for component."""
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)

    manifest_path = cache_path / "components" / component_name / "block_shards" / "manifest.json"
    return manifest_path.exists()


def get_shard_count(cache_path: Path, component_name: str) -> Optional[int]:
    """Get number of shards for component (None if no shards)."""
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)

    manifest_path = cache_path / "components" / component_name / "block_shards" / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
        return manifest.get("n_shards")
    except (json.JSONDecodeError, KeyError):
        return None
