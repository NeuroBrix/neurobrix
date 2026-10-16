"""Run a component in temporal chunks through the stream Forge derived from its graph — one brick, both engines (R30).

The plan (`ExecutionPlan.component_stream`, `core/prism/temporal_stream.py`) names the schedule: the first chunk's
input frames, then each next chunk's. The container carries the two graphs: the first chunk's (the clip graph plus the
states it leaves) and every later chunk's (the states in and out), and the declaration (`stream.json`) naming the
input, the output and its time axis, and the states. This brick feeds each chunk with the states the previous one
left and writes each chunk's output into the clip's output at its place on the time axis — the vendor's own loop
(Wan's `feat_cache`, CogVideoX's `conv_cache`), with no vendor code and no knowledge of the model.

Mode-polymorphic: a chunk is a slice of the input (torch or NBXTensor), an output buffer is allocated in the input's
own tensor kind on the chunk output's device, a state is copied out of the run that produced it (an engine's arena may
reuse that memory at the next run). R33: no torch at import; the compiled branch imports it where it allocates.

ZERO FALLBACK: a chunk whose output does not have the frames its graph's dims give, a state the run did not return, a
schedule that does not cover the input — each raises by name.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from neurobrix.core.runtime import symexpr as _symexpr


def _is_nbx(t: Any) -> bool:
    return hasattr(t, "to_cuda") and hasattr(t, "_device")


def _out_frames(dag: Dict[str, Any], out_tid: str, axis: int, symbol: str, frames: int) -> int:
    """The output's time extent a chunk graph gives at a chunk of `frames` input frames (its dims hold no other
    symbol on that axis)."""
    node = dag["tensors"][out_tid]["symbolic_shape"]["dims"][axis]

    def sym(sid, _n):
        if sid != symbol:
            raise _symexpr.ExpressionError(f"the stream output's time dim reads {sid}, not the chunk's {symbol}")
        return int(frames)
    return int(_symexpr.evaluate(node, sym, _symexpr.SHAPE))


def _empty_like_shape(ref: Any, shape: List[int]) -> Any:
    if _is_nbx(ref):
        from neurobrix.kernels.nbx_tensor import NBXTensor
        dev = f"cuda:{getattr(ref, '_device_idx', 0)}" if ref._device == "cuda" else ref._device
        return NBXTensor.empty(shape, dtype=ref.dtype, device=dev)
    import torch
    return torch.empty(*shape, dtype=ref.dtype, device=ref.device)


def _copy(t: Any) -> Any:
    buf = _empty_like_shape(t, list(t.shape))
    buf.copy_(t)
    return buf


def _key(t_meta: Dict[str, Any], tid: str) -> str:
    return (t_meta or {}).get("output_name") or tid


def run_stream(decl: Dict[str, Any], schedule: List[int], x: Any, input_name: str,
               run_first: Callable[[Dict[str, Any]], Dict[str, Any]],
               run_next: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Any:
    """The component's output over the whole input `x` ([B, C, T, ...], time on axis 2), computed chunk by chunk."""
    total = int(x.shape[2])
    if sum(schedule) != total:
        raise ValueError(f"temporal stream: the schedule {schedule} covers {sum(schedule)} frames of an input of "
                         f"{total}")
    first, nxt = decl["dags"]["first"], decl["dags"]["next"]
    out_tid, axis, symbol = decl["output"], int(decl["output_axis"]), decl["symbol"]
    out_key = {"first": _key(first["tensors"].get(out_tid), out_tid), "next": _key(nxt["tensors"].get(out_tid), out_tid)}
    widths = [_out_frames(first if i == 0 else nxt, out_tid, axis, symbol, n) for i, n in enumerate(schedule)]
    out = None
    pos = place = 0
    states: Dict[str, Any] = {}
    for i, n in enumerate(schedule):
        sl = [slice(None)] * len(x.shape)
        sl[2] = slice(pos, pos + n)
        chunk = x[tuple(sl)].contiguous()
        pos += n
        inputs = {input_name: chunk}
        inputs.update(states)
        states = {}                     # the run holds the states it reads: the plan prices them as its inputs
        result = run_first(inputs) if i == 0 else run_next(inputs)
        del inputs, chunk
        kind = "first" if i == 0 else "next"
        y = result.get(out_key[kind])
        if y is None:
            raise KeyError(f"temporal stream: chunk {i} returned no output {out_key[kind]!r} "
                           f"(it returned {sorted(result)})")
        if int(y.shape[axis]) != widths[i]:
            raise ValueError(f"temporal stream: chunk {i} of {n} frames gave {int(y.shape[axis])} output frames on "
                             f"axis {axis}, its graph's dims give {widths[i]}")
        if out is None:
            shape = list(y.shape)
            shape[axis] = sum(widths)
            out = _empty_like_shape(y, shape)
        osl = [slice(None)] * len(y.shape)
        osl[axis] = slice(place, place + widths[i])
        out[tuple(osl)] = y
        place += widths[i]
        missing = [s["name"] for s in decl["states"] if s["name"] not in result]
        if missing:
            raise KeyError(f"temporal stream: chunk {i} returned no state {missing[0]!r}")
        states = {s["name"]: _copy(result[s["name"]]) for s in decl["states"]}
    return out
