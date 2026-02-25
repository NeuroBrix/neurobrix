"""
Kernels Registry - Pattern décorateur avec tiers de compatibilité.

Structure clé: (family, vendor, tier, op_name)
Tiers: common, volta, ampere, hopper (NVIDIA) | common, cdna, cdna2, cdna3 (AMD)
"""
from typing import Callable, Dict, NamedTuple, Any, List

KERNEL_REGISTRY: Dict[tuple, 'KernelMeta'] = {}


class KernelMeta(NamedTuple):
    """Metadata d'un kernel enregistré."""
    impl: Callable
    constraints: Dict[str, Any]


def register_kernel(
    family: str,
    vendor: str,
    tier: str,
    op_name: str,
    **constraints
) -> Callable:
    """
    Décorateur pour enregistrer un kernel.

    Args:
        family: Famille du modèle (image, llm, video, audio)
        vendor: Vendor hardware (nvidia, amd)
        tier: Tier de compatibilité minimum (common, volta, ampere, hopper)
        op_name: Nom de l'opération
        **constraints: Contraintes additionnelles (dtype, etc.)

    Usage:
        @register_kernel(family="image", vendor="nvidia", tier="common", op_name="add")
        def add_kernel(x, y):
            ...

        @register_kernel(family="image", vendor="nvidia", tier="ampere", op_name="flash_attention")
        def flash_attention_kernel(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        key = (family, vendor, tier, op_name)

        if key in KERNEL_REGISTRY:
            from .exceptions import KernelRegistrationError
            raise KernelRegistrationError(f"Duplicate kernel: {key}")

        KERNEL_REGISTRY[key] = KernelMeta(impl=fn, constraints=constraints)
        return fn

    return decorator


def list_kernels(family: str = None, vendor: str = None, tier: str = None) -> List[tuple]:
    """Liste les kernels avec filtres optionnels."""
    keys = list(KERNEL_REGISTRY.keys())

    if family:
        keys = [k for k in keys if k[0] == family]
    if vendor:
        keys = [k for k in keys if k[1] == vendor]
    if tier:
        keys = [k for k in keys if k[2] == tier]

    return keys
