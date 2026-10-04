"""
A flow's row lookup in a table it reads by name — ONE function for every compiled flow.

A flow handler reads some weights BY NAME, outside the graph: the token embedding of a
language model whose graph takes `inputs_embeds`, a positional table, an RNNT decoder's
embedding. Where that table lives is the plan's decision, not the flow's:

  * a component held whole on a card has its table on the card;
  * a component Prism maps to the host (`zero3:<card>`: every shard on "cpu") has, on the
    compiled engine, ALL its weights pinned on the host for the whole run — the non-block
    ones included (`core/prism/host_footprint.py` prices exactly that; the graph's own
    non-block ops receive the weight as a per-op scratch copy,
    `CompiledSequence.mark_cpu_weighted_ops_for_transfer`). The card was budgeted for the
    activations only.

So the rule, written once here:

    the gather runs WHERE THE TABLE LIVES; its rows join the context WHERE THE CONTEXT LIVES.

The index is built on (or moved to) the table's device, the lookup runs there, and only the
gathered rows travel — a few rows, never the table. Moving the table to the index instead
(`table.to(device)`) would put bytes on a card that the plan never priced; building the
index on the context's device is the cross-device `index_select` torch refuses
(granite-speech-3.3-8b compiled on a 16 GB card, 2026-09-05 at the prompt and 2026-10-04 at
the decode step — the same line written at three call sites and fixed at two). And rows
left on the table's device meet the context at the next `torch.cat`, which refuses mixed
devices too (Voxtral-Mini-3B on mps, "Passed CPU tensor to MPS op", 2026-09-21): hence the
delivery on the caller's device, in the same function.

When table and context share a device every transfer below is a no-op and the result is the
plain `F.embedding(ids, table)`, byte for byte.

ATen only (the compiled engine). The Triton mirror gathers through `wrappers.embedding`,
and its loader keeps non-block weights on the card under the same plan
(`triton/weight_loader.py`), so it has no such boundary to cross.
"""

from typing import Any, Optional, Sequence, Union

import torch


def gather_rows(
    table: torch.Tensor,
    ids: Union[torch.Tensor, Sequence[Any]],
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """`table[ids]`, gathered on the table's device and delivered on `device`.

    table   the weight the flow read by name, wherever its plan placed it
    ids     the row indices: an integer tensor on any device, or a (nested) list of ints —
            the result has the shape of `ids` plus the table's trailing dims
    device  where the caller's context lives (required: the caller names it, the table's
            placement is never assumed)
    dtype   the dtype the rows are delivered in; None keeps the table's
    """
    if isinstance(ids, torch.Tensor):
        index = ids.to(device=table.device)
    else:
        index = torch.tensor(ids, dtype=torch.long, device=table.device)
    with torch.no_grad():
        rows = torch.nn.functional.embedding(index, table)
    if dtype is None:
        return rows.to(device=device)
    return rows.to(device=device, dtype=dtype)
