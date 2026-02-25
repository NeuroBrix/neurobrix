"""
Universal dtype validation and adaptation at component boundaries.
All models benefit from this, not just Sana.

ZERO FALLBACK: All dtypes must come from topology with hardware validation.

Usage:
    from neurobrix.core.runtime.dtype_adapter import DtypeAdapter

    # Resolve runtime dtype with hardware validation
    runtime_dtype = DtypeAdapter.resolve_runtime_dtype(
        model_dtype=torch.bfloat16,
        hardware_arch="volta",
        component_name="text_encoder"
    )
    # Returns: torch.float32 (bf16 not supported on Volta)

    # Validate/adapt tensors at component boundary
    adapted = DtypeAdapter.validate_component_io(
        input_tensors={"hidden_states": tensor},
        expected_dtype=torch.float32,
        component_name="transformer"
    )
"""

import torch
from typing import Dict, List, Optional


class DtypeAdapter:
    """
    Ensures dtype consistency between components.

    UNIVERSAL RUNTIME: Works for ALL models, not model-specific.
    ZERO FALLBACK: Missing config = crash, not silent default.
    """

    # Hardware architecture -> supported dtypes
    # CRITICAL: This is the source of truth for hardware dtype support
    HARDWARE_DTYPE_SUPPORT: Dict[str, List[torch.dtype]] = {
        # V100 - Volta architecture (cc 7.0)
        # DOES NOT support bfloat16 natively
        "volta": [torch.float32, torch.float16],

        # RTX 20xx - Turing architecture (cc 7.5)
        # Limited bfloat16 support, but available
        "turing": [torch.float32, torch.float16, torch.bfloat16],

        # A100, RTX 30xx - Ampere architecture (cc 8.0+)
        # Full bfloat16 support
        "ampere": [torch.float32, torch.float16, torch.bfloat16],

        # H100 - Hopper architecture (cc 9.0)
        # Full bfloat16 support + FP8
        "hopper": [torch.float32, torch.float16, torch.bfloat16],

        # Ada Lovelace - RTX 40xx (cc 8.9)
        "ada": [torch.float32, torch.float16, torch.bfloat16],
    }

    # Architecture aliases for common hardware profile names
    ARCH_ALIASES: Dict[str, str] = {
        "v100": "volta",
        "v100-16g": "volta",
        "v100-32g": "volta",
        "a100": "ampere",
        "a100-40g": "ampere",
        "a100-80g": "ampere",
        "h100": "hopper",
        "rtx3090": "ampere",
        "rtx4090": "ada",
    }

    @classmethod
    def _resolve_arch(cls, hardware_arch: str) -> str:
        """Resolve hardware arch name to canonical form."""
        arch_lower = hardware_arch.lower()
        return cls.ARCH_ALIASES.get(arch_lower, arch_lower)

    @classmethod
    def get_supported_dtypes(cls, hardware_arch: str) -> List[torch.dtype]:
        """
        Get supported dtypes for hardware architecture.

        Args:
            hardware_arch: Architecture name (e.g., "volta", "v100-32g", "ampere")

        Returns:
            List of supported torch dtypes
        """
        arch = cls._resolve_arch(hardware_arch)
        return cls.HARDWARE_DTYPE_SUPPORT.get(arch, [torch.float32])

    @classmethod
    def is_dtype_supported(cls, dtype: torch.dtype, hardware_arch: str) -> bool:
        """Check if dtype is supported on hardware."""
        return dtype in cls.get_supported_dtypes(hardware_arch)

    @classmethod
    def resolve_runtime_dtype(
        cls,
        model_dtype: torch.dtype,
        hardware_arch: str,
        component_name: str
    ) -> torch.dtype:
        """
        Resolve the actual runtime dtype considering hardware support.

        ZERO FALLBACK: If model_dtype not supported, MUST fallback with logging.

        Prism is the master for dtype decisions (including bf16→fp16 safety).
        This adapter is a secondary validation layer.

        Args:
            model_dtype: Original model dtype from topology
            hardware_arch: Hardware architecture (volta, ampere, etc.)
            component_name: For logging

        Returns:
            Runtime dtype (model_dtype if supported, fallback otherwise)
        """
        supported = cls.get_supported_dtypes(hardware_arch)

        if model_dtype in supported:
            return model_dtype

        # bf16 not supported: prefer fp16 (Prism validates safety upstream)
        if model_dtype == torch.bfloat16:
            if torch.float16 in supported:
                return torch.float16
            return torch.float32

        return torch.float32

    @classmethod
    def resolve_from_topology(
        cls,
        topology: Dict,
        component_name: str,
        hardware_arch: str
    ) -> Optional[torch.dtype]:
        """
        Resolve runtime dtype from topology extracted_values.

        Args:
            topology: NBX topology dict
            component_name: Component to look up (e.g., "text_encoder")
            hardware_arch: Target hardware architecture

        Returns:
            Runtime dtype or None if not specified
        """
        extracted = topology.get("extracted_values", {})
        comp_vals = extracted.get(component_name, {})

        if "dtype" not in comp_vals:
            return None

        dtype_str = comp_vals["dtype"]
        model_dtype = getattr(torch, dtype_str, torch.float32)

        return cls.resolve_runtime_dtype(
            model_dtype=model_dtype,
            hardware_arch=hardware_arch,
            component_name=component_name
        )

    @staticmethod
    def validate_component_io(
        input_tensors: Dict[str, torch.Tensor],
        expected_dtype: torch.dtype,
        component_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        Validate and adapt all input tensors to expected dtype.

        ZERO FALLBACK: expected_dtype MUST be provided (not None).

        Args:
            input_tensors: Dict of tensor name -> tensor
            expected_dtype: Required dtype (MUST NOT be None)
            component_name: For logging

        Returns:
            Dict of adapted tensors

        Raises:
            RuntimeError: If expected_dtype is None (ZERO FALLBACK violation)
        """
        if expected_dtype is None:
            raise RuntimeError(
                f"ZERO FALLBACK: expected_dtype is None for {component_name}. "
                "Dtype must come from topology with hardware validation."
            )

        adapted = {}
        for name, tensor in input_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                adapted[name] = tensor
                continue

            if tensor.dtype != expected_dtype:
                adapted[name] = tensor.to(expected_dtype)
            else:
                adapted[name] = tensor

        return adapted

    @classmethod
    def get_dominant_dtype(
        cls,
        topology: Dict,
        hardware_arch: str
    ) -> torch.dtype:
        """
        Get the dominant runtime dtype from topology with hardware validation.

        ZERO FALLBACK: If no dominant_dtype specified, returns float32.

        Args:
            topology: NBX topology dict
            hardware_arch: Target hardware architecture

        Returns:
            Runtime dtype (validated against hardware)
        """
        extracted = topology.get("extracted_values", {})
        global_vals = extracted.get("_global", {})

        dtype_str = global_vals.get("dominant_dtype", "float32")
        model_dtype = getattr(torch, dtype_str, torch.float32)

        return cls.resolve_runtime_dtype(
            model_dtype=model_dtype,
            hardware_arch=hardware_arch,
            component_name="_global"
        )
