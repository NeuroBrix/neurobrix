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


def is_dense_window(t: NBXTensor) -> bool:
    """True iff the view's strided address span equals its logical numel —
    i.e. the flat ``[data_ptr, data_ptr + nbytes)`` window contains ALL of
    the view's elements and addresses NOTHING outside it. Only such views
    may be transferred by a flat ``memcpy(nbytes)`` with their strides
    carried over.

    Dense: contiguous tensors, full-tensor permute/transpose views (the
    zero3 pre-transposed weight ``.t()``), dim-0 narrows.
    NOT dense: interior narrows / slices (offset window, span > numel),
    step>1 slices (de-interleave views, span > numel), broadcast/expand
    views (stride-0 axes, span < numel).
    """
    if t._numel == 0:
        return True
    span = 1 + sum((d - 1) * s
                   for d, s in zip(t._shape, t._strides) if d > 0)
    return span == t._numel


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
    """Copy an NBXTensor to cuda:target_dev — one call into the primitive.

    Every obligation of a card-to-card copy now lives in `NBXTensor.to_cuda`, which is the only
    place the copy is written: materialise a view whose address span exceeds its numel, wait on
    the SOURCE card before reading it, enable the direct card-to-card link. This function keeps
    its name and its callers — `_run_multi_device`, the triton_sequential per-op block, the MoE
    dispatch, the flows' device hand-offs and the strategies' component boundary all reach the
    same path, and none of them can reach a copy that skips a step.

    The history that produced those obligations, kept because it is why they exist. The
    dense-window rule came from the DeepSeek-Coder-V2-Lite pipeline_parallel boundary (block.13
    MLA rope q/k de-interleave: a [1,16,23,64] narrow-of-transpose view whose strides address a
    70528-element span over a 23552-element allocation, 2/3 of reads out of bounds, surfacing as
    an async fault at aten.cat::91). The source wait came from the Qwen3-Omni audio triton leg
    (deterministic error-700 poison at step-2 entry on a block-scatter placement, gone under
    CUDA_LAUNCH_BLOCKING, allocator ledger clean, flow tail quiesced — only the transfer region
    remained). The peer link came from a measured 8.7x on a 34 MB stage boundary, 48 ms to 5.5.
    """
    return tensor.to_cuda(target_dev)
