"""Triton-side device utilities — device-string parsing and cross-device
NBXTensor transfer, shared by the compiled (TritonSequence) and op-by-op
(triton_sequential) execution paths.

Pipeline-parallel placement shards a component's weights across GPUs (block N
on cuda:i, block N+k on cuda:j). At a stage boundary the activation produced on
the previous stage's device must be moved to the next stage's device before the
op runs — `NBXTensor.to('cuda:N')` is intentionally a no-op (it only casts
dtype), so the move is a real D2D `memcpy` here. R33-pure (DeviceAllocator +
NBXTensor only; no torch). Extracted from `sequence.py:_transfer_tensor` /
`_needs_move` so both modes share ONE implementation (R30 parity by
construction). Model-agnostic.

`parse_device_idx` / `parse_device_idxs` are the SINGLE triton-side device-
string parsers (brick-consolidation E1). They replace six divergent
`_parse_device_idx` copies that lived in the triton flow handlers, which
disagreed on every non-trivial form of the Prism device string:

  - the fast path `device.split(":")[-1]` returned the LAST ordinal for a
    bare comma list ("cuda:2,cuda:3" → 3) while the compound branch of the
    same copies returned the FIRST for the fgp-prefixed form
    ("fgp:cuda:0,cuda:1" → 0);
  - three copies had no compound branch at all and silently returned 0 for
    every "fgp:"/"tp:" string (wrong device whenever the primary was not
    cuda:0);
  - one naive copy (`int(s.split(":")[1])`) CRASHED with ValueError on any
    compound string.

Canonical semantics here: the primary device is the FIRST CUDA ordinal named
in the string ("fgp:cuda:1,cuda:2" → 1, "cuda:2,cuda:3" → 2), a bare integer
string is accepted as an ordinal, and anything unparseable (bare "cuda",
"cpu") resolves to 0 — the behaviour every flow copy converged on for the
degenerate forms.
"""

import re
from typing import List

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator

_CUDA_ORDINAL_RE = re.compile(r"cuda:(\d+)")


def parse_device_idxs(device) -> List[int]:
    """All CUDA ordinals named in a device string, in order of appearance.

    Handles every Prism device-string form: "cuda:2" → [2],
    "fgp:cuda:0,cuda:1" → [0, 1], "tp:cuda:2,cuda:3" → [2, 3],
    "cuda" / "cpu" / None → [].
    """
    return [int(m) for m in _CUDA_ORDINAL_RE.findall(str(device))]


def parse_device_idx(device) -> int:
    """Primary (FIRST) CUDA ordinal of a Prism device string.

    "cuda:2" → 2, "fgp:cuda:1,cuda:2" → 1, "cuda:2,cuda:3" → 2, "3" → 3,
    "cuda" / "cpu" / unparseable → 0. Never raises — the flow handlers feed
    this straight into `DeviceAllocator.set_device` at setup time and every
    historical copy resolved degenerate strings to device 0.
    """
    idxs = parse_device_idxs(device)
    if idxs:
        return idxs[0]
    try:
        return int(str(device).strip())
    except (ValueError, TypeError):
        return 0


def needs_move(t: NBXTensor, target_dev: int) -> bool:
    """True iff `t` must be transferred to land on cuda:target_dev.

    1. CPU source (zero3 offload) → must H2D (even if `_device_idx` is 0).
    2. Different CUDA device → must D2D.
    3. Already on target → no-op.
    """
    if getattr(t, "_device", "cuda") == "cpu":
        return True
    return t._device_idx != target_dev


def transfer_tensor(tensor: NBXTensor, target_dev: int) -> NBXTensor:
    """Copy an NBXTensor to cuda:target_dev, preserving shape/strides/dtype.

    kind=1 (H2D) for a CPU source (zero3), kind=3 (D2D) for a cross-GPU move.
    Expanded/broadcast views (stride 0) are materialised first so the memcpy
    does not over-read the backing allocation. Strides are carried over (not
    forced contiguous) — a pre-transposed weight view keeps its `.t()` stride
    semantics, matching the native `torch.Tensor.to(device)` contract; a
    downstream `.contiguous()` inside mm/bmm materialises correctly.
    """
    if tensor.is_expanded():
        tensor = tensor.contiguous()
    DeviceAllocator.set_device(target_dev)
    src_device = getattr(tensor, "_device", "cuda")
    kind = 1 if src_device == "cpu" else 3
    if tensor._nbytes > 0:
        dst_raw_ptr = DeviceAllocator.malloc_cuda(tensor._nbytes)
        DeviceAllocator.memcpy(dst_raw_ptr, tensor.data_ptr(),
                               tensor._nbytes, kind=kind)
    else:
        dst_raw_ptr = 0
    return NBXTensor(
        dst_raw_ptr, tensor._shape, tensor._strides, tensor._dtype,
        "cuda", owns_data=True, device_idx=target_dev, offset=0)
