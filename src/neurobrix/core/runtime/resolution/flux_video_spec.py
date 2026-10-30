"""Whether a model is of the FLUX video family, read by both branches' conditioning. Torch-free."""
from typing import Any, List


def is_flux_family(ctx: Any, components: List[str]) -> bool:
    """True iff a loop denoiser declares an ``img_ids`` input (FLUX-family)."""
    comps = ctx.pkg.topology.get("components", {})
    for c in components:
        inputs = comps.get(c, {}).get("interface", {}).get("inputs", []) or []
        if "img_ids" in inputs:
            return True
    return False
