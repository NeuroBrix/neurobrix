"""The compiled engine's attention, as the device can run it (core/prism/attention_scratch.py is its price).

On a device whose vendor file states `memory.compiled_sdpa_full_scores: true` — PyTorch's MPS attention is the fused math
path, it writes the whole [B, H, Tq, Tk] score matrix (pytorch/pytorch#179294) — an attention whose scores exceed the
PLAN's budget (`ComponentAllocation.attention_scores_budget_bytes`, the figure Prism priced it at) runs in query chunks:
rows [i, i + r) attend every key, each row's softmax its own, so the chunked result is the whole one's. Elsewhere
`torch.nn.functional.scaled_dot_product_attention` runs as it is. A causal flag becomes the chunk's slice of the top-left
lower triangle (the flag's own definition in PyTorch's documentation); a mask is sliced on the axes it has at full extent.

The engine never reads a budget of its own: on a device that writes full scores, an attention given no budget (a plan
that states none) or no hardware is refused by name. A tensor on the host is the host's attention, which no accelerator
vendor file describes.

Compiled branch only (torch). The Triton branch has its own attention kernels (R33).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

_MEMORY = {}          # (vendor, architecture) -> the vendor file's memory section
_SAID = set()         # the (shape, budget) cuts already said: one line per distinct cut, never one per call


def _memory(hardware):
    from neurobrix.core.prism.attention_scratch import vendor_memory_of
    key = tuple(hardware)
    if key not in _MEMORY:
        _MEMORY[key] = vendor_memory_of(*key)
    return _MEMORY[key]


def _plain(q, k, v, attn_mask, dropout_p, is_causal, scale):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                          is_causal=is_causal, scale=scale)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False,
                                 scale: Optional[float] = None, hardware=None, budget: Optional[int] = None):
    """F.scaled_dot_product_attention, in query chunks where the device writes full scores beyond the plan's budget.
    `hardware`: the (vendor, architecture) of the executing device (the dtype engine's); `budget`: the plan's."""
    if q.device.type == "cpu":
        return _plain(q, k, v, attn_mask, dropout_p, is_causal, scale)
    if hardware is None:
        raise ValueError(f"ZERO FALLBACK: an attention on {q.device} was given no hardware — whether its device writes "
                         f"its full scores is read from the vendor file of the plan's device, never assumed")
    from neurobrix.core.prism.attention_scratch import writes_full_scores, chunk_plan
    memory = _memory(hardware)
    if not writes_full_scores(memory):
        return _plain(q, k, v, attn_mask, dropout_p, is_causal, scale)
    if budget is None:
        raise ValueError(f"ZERO FALLBACK: {hardware[0]}/{hardware[1]} writes an attention's full scores and the plan "
                         f"states no attention_scores_budget_bytes for this component — the chunk is the plan's, "
                         f"never the engine's own reading")
    if q.dim() == 3:          # [B, T, D]: one head
        out = scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            attn_mask.unsqueeze(1) if attn_mask is not None and attn_mask.dim() == 3 else attn_mask,
            dropout_p, is_causal, scale, hardware=hardware, budget=budget)
        return out.squeeze(1)
    if q.dim() != 4:
        raise ValueError(f"ZERO FALLBACK: an attention of rank {q.dim()} on a device that writes its full scores — "
                         f"its chunks are cut on [B, H, T, D] or [B, T, D]")
    b, h, tq, tk = int(q.shape[0]), int(q.shape[1]), int(q.shape[-2]), int(k.shape[-2])
    plan = chunk_plan(memory, budget, b, h, tq, tk)
    _say(b, h, tq, tk, budget, plan)
    if plan is None:
        return _plain(q, k, v, attn_mask, dropout_p, is_causal, scale)
    return attention_in_chunks(q, k, v, attn_mask, dropout_p, is_causal, scale, plan[1], heads=plan[0])


def _say(b, h, tq, tk, budget, plan):
    """One line per distinct cut: what the attention writes and at what budget — the evidence that it ran cut."""
    key = (b, h, tq, tk, budget)
    if key in _SAID:
        return
    _SAID.add(key)
    from neurobrix.core.module.tiling_sizes import sdpa_scores_bytes
    whole = sdpa_scores_bytes(b, h, tq, tk) / 2**20
    if plan is None:
        print(f"[attention] {b} x {h} heads, {tq} x {tk}: whole, scores {whole:.0f} MB within the plan's budget "
              f"{budget / 2**20:.0f} MB", flush=True)
        return
    group, rows = plan
    n = -(-(b * h) // group) * -(-tq // rows)
    print(f"[attention] {b} x {h} heads, {tq} x {tk}: {n} chunks of {group} head(s) x {rows} rows, scores "
          f"{sdpa_scores_bytes(1, group, rows, tk) / 2**20:.0f} MB a chunk (whole {whole:.0f} MB) within the plan's "
          f"budget {budget / 2**20:.0f} MB", flush=True)


def _heads(t, bi, h0, h1, b, h):
    """The (batch, head) slice of a key, value or mask: an axis at full extent is sliced, a broadcast one kept."""
    t = t[bi:bi + 1] if t.shape[0] == b else t
    if t.shape[1] == h:
        return t[:, h0:h1]
    if t.shape[1] == 1:
        return t
    raise ValueError(f"ZERO FALLBACK: an attention operand with {t.shape[1]} heads against {h} query heads — "
                     f"neither the query's heads nor a broadcast one")


def attention_in_chunks(q, k, v, attn_mask, dropout_p, is_causal, scale, rows: int, heads: Optional[int] = None):
    """The attention cut into groups of `heads` (batch, head) slices by bands of `rows` query rows, each band attending
    every key, reassembled in place."""
    b, h, tq, tk = int(q.shape[0]), int(q.shape[1]), int(q.shape[-2]), int(k.shape[-2])
    group = heads or b * h
    out = torch.empty(q.shape[:-1] + (v.shape[-1],), dtype=q.dtype, device=q.device)
    for s0 in range(0, b * h, group):
        s1 = min(s0 + group, b * h)
        for bi in range(s0 // h, (s1 - 1) // h + 1):            # the batch entries this group of slices covers
            h0, h1 = max(s0 - bi * h, 0), min(s1 - bi * h, h)
            qs, ks, vs = q[bi:bi + 1, h0:h1], _heads(k, bi, h0, h1, b, h), _heads(v, bi, h0, h1, b, h)
            m_b = None
            if attn_mask is not None:
                m_b = _heads(attn_mask, bi, h0, h1, b, h) if attn_mask.dim() == 4 else attn_mask
            for i in range(0, tq, rows):
                j = min(i + rows, tq)
                mask = m_b
                if mask is not None and mask.dim() >= 2 and mask.shape[-2] == tq:
                    mask = mask[..., i:j, :]
                if is_causal:
                    tri = torch.ones(j - i, tk, dtype=torch.bool, device=q.device).tril(diagonal=i)
                    mask = tri if mask is None else (mask & tri if mask.dtype == torch.bool
                                                     else mask.masked_fill(~tri, float("-inf")))
                out[bi:bi + 1, h0:h1, i:j] = F.scaled_dot_product_attention(
                    qs[..., i:j, :], ks, vs, attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=scale)
    return out
