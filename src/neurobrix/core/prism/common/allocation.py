# core/prism/common/allocation.py
"""
Allocation Utilities for Prism

Extracted from solver.py and smart_solver.py.
Common allocation helpers used by the Prism solver.
"""

from typing import Dict, List, Tuple, Optional

# Use consolidated dtype module (eliminates code duplication)
from neurobrix.core.dtype import calculate_dtype_multiplier, get_dtype_bytes


def get_device_index(device_string: str) -> int:
    """
    Extract device index from device string.

    Args:
        device_string: Device string (e.g., 'cuda:2', 'cuda:0')

    Returns:
        Device index (e.g., 2, 0)
    """
    if ':' not in device_string:
        return 0
    return int(device_string.split(':')[1])


def allocate_sequential_shards(
    shards_with_sizes: List[Tuple[str, int, float]],
    device_order: List[str],
    device_capacity: Dict[str, float],
    device_usage: Dict[str, float],
    log_prefix: str = "[Prism]",
) -> Tuple[Dict[str, str], List[str], float]:
    """
    Allocate shards SEQUENTIALLY for pipeline parallelism.
    
    Pipeline requires blocks to be sequential on GPUs:
    - GPU0: blocks 0, 1, 2 (sequential)
    - GPU1: blocks 3, 4, 5 (sequential)
    NOT: GPU0: blocks 0, 2, 4 (interleaved - would break pipeline)
    
    Args:
        shards_with_sizes: List of (shard_path, block_idx, size_mb) tuples,
                          sorted by block index
        device_order: List of devices in order of preference
        device_capacity: {device: capacity_mb}
        device_usage: {device: current_usage_mb} - MODIFIED IN PLACE
        log_prefix: Prefix for log messages
        
    Returns:
        (shard_map, devices_used, total_memory_mb)
    """
    shard_map: Dict[str, str] = {}
    devices_used: List[str] = []
    total_memory_mb = 0.0
    
    current_device_idx = 0
    current_device = device_order[current_device_idx]
    
    for shard_path, block_idx, shard_mb in shards_with_sizes:
        # Check if current device has space
        free_space = device_capacity[current_device] - device_usage[current_device]
        
        # If not enough space, move to next device
        while shard_mb > free_space and current_device_idx < len(device_order) - 1:
            current_device_idx += 1
            current_device = device_order[current_device_idx]
            free_space = device_capacity[current_device] - device_usage[current_device]
            
        # Check if we can fit on current device
        if shard_mb <= free_space:
            assigned_device = current_device
            device_usage[current_device] += shard_mb
        else:
            # No GPU space, must go to CPU
            assigned_device = "cpu"
        shard_map[shard_path] = assigned_device
        total_memory_mb += shard_mb
        
        if assigned_device not in devices_used and assigned_device != "cpu":
            devices_used.append(assigned_device)
            
    return shard_map, devices_used, total_memory_mb


