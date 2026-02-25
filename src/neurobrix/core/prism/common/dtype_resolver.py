# core/prism/common/dtype_resolver.py
"""
Dtype Resolver for Prism

Extracted from solver.py and smart_solver.py.
Unified dtype resolution logic used by the Prism solver.

bf16→fp16 is safe when Prism's _scan_bf16_fp16_safety() verifies weights.
"""

import torch
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from neurobrix.core.prism.structure import PrismProfile

# Use consolidated dtype module (eliminates code duplication)
from neurobrix.core.dtype import DTYPE_MAP, get_dtype_bytes, calculate_dtype_multiplier


class DtypeResolver:
    """
    Resolves optimal dtype based on model requirements and hardware capabilities.
    
    Priority order:
    1. Hardware preferred_dtype (if set and supported)
    2. Model's requested dtype (if supported)
    3. Safe fallback to float32
    
    bf16→fp16 allowed when Prism validates weight safety. Otherwise fp32.
    """
    
    def __init__(self, profile: "PrismProfile"):
        """
        Initialize resolver with hardware profile.
        
        Args:
            profile: PrismProfile with device capabilities
        """
        self.profile = profile
        
    def resolve(
        self,
        requested: str,
        log_prefix: str = "[DtypeResolver]",
    ) -> torch.dtype:
        """
        Resolve optimal dtype for the given request.
        
        Args:
            requested: Requested dtype string (e.g., "bfloat16", "float16")
            log_prefix: Prefix for log messages
            
        Returns:
            Resolved torch.dtype
        """
        if not self.profile.devices:
            return torch.float32
            
        # Check for hardware-specified preferred dtype
        if self.profile.preferred_dtype and self.profile.devices_support_dtype(self.profile.preferred_dtype):
            return DTYPE_MAP.get(self.profile.preferred_dtype, torch.float32)
            
        # Check if requested dtype is supported
        if self.profile.devices_support_dtype(requested):
            return DTYPE_MAP.get(requested, torch.float32)
            
        # Fallback rules - preserve numerical range
        return self._safe_fallback(requested, log_prefix)
        
    def _safe_fallback(self, requested: str, log_prefix: str) -> torch.dtype:
        """
        Determine fallback dtype. bf16→fp16 preferred when hw supports fp16
        (Prism validates weight safety upstream).
        """
        if requested == "bfloat16":
            if self.profile.devices_support_dtype("float16"):
                return torch.float16
            return torch.float32
        elif requested == "float16":
            return torch.float32
        else:
            return torch.float32
            
    def get_dtype_multiplier(self, source: str, target: str) -> float:
        """
        Calculate memory multiplier for dtype conversion.

        Args:
            source: Source dtype string
            target: Target dtype string

        Returns:
            Multiplier for memory calculation
        """
        # Use consolidated dtype module (eliminates code duplication)
        return calculate_dtype_multiplier(source, target)


def resolve_dtype(
    requested: str,
    profile: "PrismProfile",
    log_prefix: str = "[DtypeResolver]",
) -> torch.dtype:
    """
    Convenience function for dtype resolution.
    
    Args:
        requested: Requested dtype string
        profile: Hardware profile
        log_prefix: Log message prefix
        
    Returns:
        Resolved torch.dtype
    """
    resolver = DtypeResolver(profile)
    return resolver.resolve(requested, log_prefix)
