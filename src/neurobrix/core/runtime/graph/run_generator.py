"""The ATen branch's run-scoped generator for the graph's own random draws.

A diffusers pipeline threads ONE seeded `torch.Generator` through every draw
of a run, in consumption order: an image encoder's posterior sample
(`retrieve_latents(vae.encode(x), generator)`), then the initial noise, then
each stochastic scheduler step. The ATen branch carries that generator as
`VariableResolver.sampling_generator()` and draws the initial noise and the
scheduler noise from it — but an RNG op INSIDE a graph (`aten::randn_like`
for a traced posterior sample, `aten::rand` for a vocoder phase) drew from
torch's GLOBAL generator, seeded with the SAME seed at the CLI. Two streams
from one seed re-emit one sequence: a [1, 16, 1, 60, 90] posterior draw from
the global stream equals, element for element, the first frame of the
[1, 13, 16, 60, 90] initial noise drawn from the run's generator (CPU,
2026-10-04, 86 400 of 86 400) — the correlated-injection class the
resolver's docstring records for the scheduler noise.

This module is the ATen mirror of `kernels/rng_stream` (the Triton branch,
where every draw — graph op, initial noise, scheduler — already comes from
one stream): the executor arms it once per request with the resolver's
generator, and the graph's random ops draw from it. Draws follow diffusers'
`randn_tensor` semantics: on the generator's device, then moved (a
`torch.Generator` is device-bound; a multi-device plan may run the op
elsewhere). A run with no generator (a seedless request) draws from the
global stream, as before.
"""
from typing import Any, Callable, Optional, Sequence

import torch

_PROVIDER: Optional[Callable[[], Optional[torch.Generator]]] = None

UNIFORM_OPS = ("rand", "rand_like")
NORMAL_OPS = ("randn", "randn_like")
DRAW_OPS = UNIFORM_OPS + NORMAL_OPS


def arm(provider: Optional[Callable[[], Optional[torch.Generator]]]) -> None:
    """Arm the run's generator provider (None disarms). The provider is
    called at each draw, so a lazily created generator is created once, by
    whichever draw comes first, and shared by all."""
    global _PROVIDER
    _PROVIDER = provider


def generator() -> Optional[torch.Generator]:
    return _PROVIDER() if _PROVIDER is not None else None


def draw(op_name: str, shape: Sequence[int], dtype: torch.dtype,
         device: Any) -> Optional[torch.Tensor]:
    """The op's draw from the run's generator, or None when the run has none.

    `op_name` is one of DRAW_OPS; anything else is refused by name."""
    if op_name not in DRAW_OPS:
        raise ValueError(f"run_generator.draw: {op_name!r} is not one of {DRAW_OPS}")
    gen = generator()
    if gen is None:
        return None
    fn = torch.rand if op_name in UNIFORM_OPS else torch.randn
    out = fn(tuple(int(s) for s in shape), generator=gen, device=gen.device, dtype=dtype)
    return out.to(device)
