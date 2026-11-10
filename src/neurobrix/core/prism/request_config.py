"""The batch and the dtype Prism plans a request with — one rule for `neurobrix run` and for the daemon.

Contract. In: the request (an argparse namespace, or None for the daemon: a served request states neither), the
container's runtime defaults, its family's defaults, its manifest, and `single` (1 when the request carries ONE image,
audio or prompt, else None). Out: (batch_size, guidance_passes, dtype). Refuses, as `runtime_values.MissingRuntimeValue`,
a batch or a dtype that neither the request, the container, the family nor the manifest states — never a literal.

Why it is a module: `batch_size=2` stood in `run` for every model ("CFG effectively doubles batch" — applied to
upscalers and speech models, which run no guidance, and over the batch_size: 1 diffusion containers declare), and the
same literal stood in the daemon until 2026-11-10 with dtype="float16" for every served family: a served LLM's
151 936-wide head was priced twice over for a batch nobody sends. CFG doubles the batch only where guidance is in
force, and the engine already treats `guidance_scale` as the declaration of that (core/flow/autoregressive.py refuses
without it). The dtype is the one execution resolves: every container declares one (float32 for swin2SR, bfloat16 for
TinyLlama, float16 for whisper), some ONLY in the manifest.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from neurobrix.core.runtime_values import resolve


def request_batch_and_dtype(request: Any, cached_defaults: Mapping[str, Any], fam_defaults: Mapping[str, Any],
                            manifest: Mapping[str, Any], single: Optional[int]) -> Tuple[int, Optional[int], str]:
    """(batch_size, guidance_passes, dtype): the request, then the container, then the family; the request's single
    item after its own explicit batch; the manifest's dtype after the defaults'."""
    batch_size = resolve("batch_size", request, cached_defaults, fam_defaults,
                         extra=[("the request's single input item", single)],
                         why="It is the batch Prism plans for.")
    guidance = resolve("guidance_scale", request, cached_defaults, fam_defaults, default=None)
    guidance_passes = None
    if guidance is not None and float(guidance) > 1.0:
        # [uncond, cond]: the two branches the CFG engines concatenate on the batch axis.
        guidance_passes = 2
        batch_size *= guidance_passes
    dtype = resolve("dtype", request, cached_defaults, fam_defaults,
                    extra=[("the container manifest's dtype", manifest.get("dtype"))],
                    why="It sizes every tensor in the plan.")
    return batch_size, guidance_passes, dtype


def served_plan_fields(family: Optional[str], cached_defaults: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict:
    """The same three, as `InputConfig` fields, for a serving session: no request states them and a served request
    carries one item."""
    from neurobrix.core.config import get_family_defaults
    batch_size, guidance_passes, dtype = request_batch_and_dtype(
        None, cached_defaults, get_family_defaults(family) if family else {}, manifest, 1)
    return {"batch_size": batch_size, "guidance_passes": guidance_passes, "dtype": dtype}
