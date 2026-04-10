# core/prism/common/dtype_resolver.py
"""
Dtype Resolver for Prism

Extracted from solver.py and smart_solver.py.
Unified dtype resolution logic used by the Prism solver.

bf16→fp16 is safe when Prism's _scan_bf16_fp16_safety() verifies weights.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.prism.structure import PrismProfile

from neurobrix.core.dtype import calculate_dtype_multiplier


class DtypeResolver:
    """
    Resolves optimal dtype based on model requirements and hardware capabilities.

    Priority order:
    1. Hardware preferred_dtype (if set and supported)
    2. Model's requested dtype (if supported)
    3. Safe fallback to float32

    bf16→fp16 allowed when Prism validates weight safety. Otherwise fp32.
    Returns dtype as STRING — engines convert internally.
    """

    def __init__(self, profile: "PrismProfile"):
        self.profile = profile

    def resolve(
        self,
        requested: str,
        log_prefix: str = "[DtypeResolver]",
    ) -> str:
        """
        Resolve optimal dtype for the given request.

        Args:
            requested: Requested dtype string (e.g., "bfloat16", "float16")

        Returns:
            Resolved dtype string
        """
        if not self.profile.devices:
            return "float32"

        # Check for hardware-specified preferred dtype
        if self.profile.preferred_dtype and self.profile.devices_support_dtype(self.profile.preferred_dtype):
            return self.profile.preferred_dtype

        # Check if requested dtype is supported
        if self.profile.devices_support_dtype(requested):
            return requested

        # Fallback rules - preserve numerical range
        return self._safe_fallback(requested)

    def _safe_fallback(self, requested: str) -> str:
        """
        Determine fallback dtype. bf16→fp16 preferred when hw supports fp16
        (Prism validates weight safety upstream).
        """
        if requested == "bfloat16":
            if self.profile.devices_support_dtype("float16"):
                return "float16"
            return "float32"
        elif requested == "float16":
            return "float32"
        else:
            return "float32"

    def get_dtype_multiplier(self, source: str, target: str) -> float:
        """Calculate memory multiplier for dtype conversion."""
        return calculate_dtype_multiplier(source, target)


def resolve_dtype(
    requested: str,
    profile: "PrismProfile",
    log_prefix: str = "[DtypeResolver]",
) -> str:
    """Convenience function for dtype resolution. Returns dtype string."""
    resolver = DtypeResolver(profile)
    return resolver.resolve(requested, log_prefix)
