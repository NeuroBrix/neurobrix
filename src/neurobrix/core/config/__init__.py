# core/config/__init__.py
"""
NeuroBrix Configuration Package.

Provides both system configuration and family/vendor configuration loading.

Re-exports:
- System config (from neurobrix.core.config.system): Memory, Prism, dtype constants
- Family/Vendor config (from neurobrix.core.config.loader): Model-level defaults
"""

# System configuration (memory, prism, dtype)
from neurobrix.core.config.system import (
    load_system_config,
    get_memory_constants,
    get_prism_defaults,
    get_precision_config,
    get_dtype_bytes,
    bytes_to_mb,
    bytes_to_gb,
    mb_to_bytes,
    gb_to_bytes,
)

# Family/Vendor configuration (DATA-DRIVEN model config)
from neurobrix.core.config.loader import (
    # Family config
    get_family_config,
    get_family_defaults,
    get_output_processing,
    cascade_default,
    list_families,
    # Vendor config
    get_vendor_config,
    get_device_prefix,
    get_block_sizes,
    get_gemm_config,
    get_bmm_config,
    get_sdpa_block_size,
    get_precision_support,
    list_vendors,
    list_available_vendor_configs,
    # Helpers
    get_rope_base,
    get_default,
)

__all__ = [
    # System config
    "load_system_config",
    "get_memory_constants",
    "get_prism_defaults",
    "get_precision_config",
    "get_dtype_bytes",
    "bytes_to_mb",
    "bytes_to_gb",
    "mb_to_bytes",
    "gb_to_bytes",
    # Family config
    "get_family_config",
    "get_family_defaults",
    "get_output_processing",
    "cascade_default",
    "list_families",
    # Vendor config
    "get_vendor_config",
    "get_device_prefix",
    "get_block_sizes",
    "get_gemm_config",
    "get_bmm_config",
    "get_sdpa_block_size",
    "get_precision_support",
    "list_vendors",
    "list_available_vendor_configs",
    # Helpers
    "get_rope_base",
    "get_default",
]
