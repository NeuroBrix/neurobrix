"""The decoder call of the PyTorch branch's audio_llm and encoder_decoder flows: one forward call of the decoding
component, and the token embedding to read after it. The same function stands once per branch (twins by path:
`neurobrix.pytorch.flow.decoder_call`, `neurobrix.triton.flow.decoder_call`); each branch's flows import their own."""
from typing import Any, Callable


def decoder_call(ctx, run: Callable[..., Any], name: str, embed_weight: Any, read: Callable[[], Any]) -> Any:
    """Run `name` once through the flow's component call `run`; return `embed_weight`, or what `read` reads again
    when the call re-planned. A device out-of-memory re-plan inside the call (`RuntimeExecutor._replan_after_oom`) frees every
    table and loads it again under the plan it gives the flow's ctx: the embedding read before the call is the old
    load's, so it is read again from the reloaded table. A re-plan is known by that plan: nothing else replaces it."""
    plan = ctx.plan
    run(name, "forward", None)
    return embed_weight if ctx.plan is plan else read()
