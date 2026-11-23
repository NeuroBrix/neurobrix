"""The PyTorch engine's host-held weights pinned in ONE block registered in place (cudaHostRegister, exact size), each
original dropped as its copy is made — per-weight pin_memory() made power-of-two-rounded pinned copies beside the
originals (Qwen3-Coder-30B native: 60.5 GB + 76.5 GB for 57.6 GB of weights, 2026-11-06). Unregistered with its holder."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch                    # the PyTorch engine only (imported by its strategies' native branch)

_ALIGN = 64


class PinnedHostBlock:                  # owns the registered block; the weights are views of it
    def __init__(self, block):
        self.block = block
        self.nbytes = int(block.numel())

    def __del__(self):          # torch held by the module: an import here fails while the interpreter shuts down
        torch.cuda.cudart().cudaHostUnregister(self.block.data_ptr())


def pin_in_one_block(weights: Dict[str, Any]) -> Optional[PinnedHostBlock]:
    """Pin the unpinned CPU torch tensors of `weights` in one registered block, in place in the dict; None when none."""
    names = [n for n, t in weights.items() if isinstance(t, torch.Tensor) and t.device.type == "cpu" and not t.is_pinned()]
    if not names:
        return None
    sizes = {n: weights[n].numel() * weights[n].element_size() for n in names}
    total = sum(-(-s // _ALIGN) * _ALIGN for s in sizes.values())
    # Registered once filled: registering pins every page at once — registered empty, it held the weights twice while
    # loading (Qwen3-Coder-30B native: 118 GB anonymous at the first copy, 2026-11-06); filled, the host holds one copy.
    block, off = torch.empty(total, dtype=torch.uint8), 0
    for n, size in sizes.items():
        t = weights[n]
        view = block[off:off + size].view(t.dtype).view(t.shape) if size else torch.empty(t.shape, dtype=t.dtype)
        view.copy_(t)
        weights[n], off = view, off + -(-size // _ALIGN) * _ALIGN          # the original goes with this assignment
    rc = torch.cuda.cudart().cudaHostRegister(block.data_ptr(), total, 0)
    if int(getattr(rc, "value", rc)) != 0:
        raise RuntimeError(f"ZERO FALLBACK: cudaHostRegister of {total} host bytes failed ({rc}) — the host-held weights "
                           f"cannot be pinned for their asynchronous copies")
    return PinnedHostBlock(block)
