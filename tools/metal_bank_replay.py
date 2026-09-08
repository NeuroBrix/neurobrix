#!/usr/bin/env python3
"""Replay the Dell's kernel reference bank on this machine.

Each `.npz` in the bank holds the seeded inputs an op was run with, the
output CUDA produced, and an fp64 oracle. This runs the same op here through
the engine, and reports the Metal distance to that oracle beside the CUDA
distance already recorded in the file's `meta`.

    python tools/metal_bank_replay.py --bank <dir> --out results.json
    python tools/metal_bank_replay.py --bank <dir> --only matmul,addmm

The bar the owner set: **Metal ULP no greater than CUDA ULP**, per kernel and
per shape. A kernel that refuses is reported as refused with its reason — it
is a measured gap, not a missing row.

The bank lives on a read-only mount; nothing here writes to it.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np


def ulp_distance(got: np.ndarray, oracle: np.ndarray) -> dict:
    """Distance in ULP of the result's own dtype, plus the plain errors.

    The oracle is fp64; it is rounded to the result's dtype first, so the
    comparison is "how many representable steps away", which is what the
    bank's own numbers mean.
    """
    dtype = got.dtype
    rounded = oracle.astype(dtype)

    # A boolean or integer result has no ULP: there is no rounding between
    # representable neighbours to count, and reinterpreting one byte of bool
    # as a wider float raises rather than lying. They are compared BIT-
    # IDENTICALLY, which is the same rule the autotune screen already states
    # for them. Nine of the copy family's bank entries — eq, ne, lt, le, gt,
    # ge and the three logicals — refused here for exactly this reason, and a
    # refusal that comes from the measuring tool tells you nothing about the
    # engine.
    if dtype.kind in ("b", "i", "u"):
        same = np.array_equal(got, rounded)
        differing = int((got != rounded).sum())
        return {
            "max_ulp": 0 if same else 1,
            "mean_ulp": 0.0 if same else float(differing) / max(got.size, 1),
            "max_abs_err": 0.0 if same else 1.0,
            "rel_err": 0.0 if same else 1.0,
            "nonfinite": 0,
            "identical": same,
            "comparison": "bit-identical (no ULP for a boolean or integer)",
            "differing_elements": differing,
        }
    # The sign-magnitude -> ordered mapping is done in int64 throughout: the
    # bias for fp16 is 0x8000, which does not fit in the int16 the bits are
    # VIEWED as, and constructing it there raises rather than wrapping.
    view = {np.dtype(np.float16): np.int16,
            np.dtype(np.float32): np.int32}.get(dtype, np.int64)
    bias = {np.int16: 0x8000, np.int32: 0x80000000}.get(
        view, 0x8000000000000000)

    def ordered(v):
        i = v.view(view).astype(np.int64)
        return np.where(i < 0, np.int64(bias) - i, i)

    finite = np.isfinite(got) & np.isfinite(rounded)
    ulp = np.abs(ordered(got[finite]) - ordered(rounded[finite])) if finite.any() \
        else np.array([0])
    scale = float(np.abs(oracle).max()) or 1.0
    return {
        "max_ulp": int(ulp.max()),
        "mean_ulp": float(ulp.mean()),
        "max_abs_err": float(np.abs(got.astype(np.float64) - oracle).max()),
        "rel_err": float(np.abs(got.astype(np.float64) - oracle).max() / scale),
        "nonfinite": int((~np.isfinite(got)).sum()),
        "identical": bool(np.array_equal(got, rounded)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", default=None,
                        help="comma-separated op substrings")
    parser.add_argument("--launched", default=None,
                        help="comma-separated KERNEL names; keeps only the "
                             "entries whose recorded launch list contains one "
                             "of them — 'the GEMV family' as the bank itself "
                             "records it, rather than as a guess from op names")
    parser.add_argument("--extra-dtypes", default=None,
                        help="comma-separated dtypes to ALSO run the same "
                             "seeded inputs in. The bank's oracle belongs to "
                             "the dtype it was recorded in, so these rows are "
                             "compared against an fp64 oracle recomputed here "
                             "and carry no CUDA column — there is no CUDA "
                             "number for a dtype the Dell did not run.")
    args = parser.parse_args()

    from neurobrix.kernels import launcher
    launcher.install()

    wanted = [w.strip() for w in args.only.split(",")] if args.only else None
    launched_filter = ([w.strip() for w in args.launched.split(",")]
                       if args.launched else None)
    extra_dtypes = ([np.dtype(d.strip()) for d in args.extra_dtypes.split(",")]
                    if args.extra_dtypes else [])
    rows = []
    files = sorted(p for p in args.bank.rglob("*.npz"))
    for path in files:
        payload = np.load(path, allow_pickle=True)
        meta = json.loads(str(payload["meta"]))
        op = meta["op"]
        if wanted and not any(w in op for w in wanted):
            continue
        if launched_filter and not any(
                k in (meta.get("launched") or []) for k in launched_filter):
            continue
        row = {"op": op, "tag": meta["tag"], "file": path.name,
               "launched": meta.get("launched", []),
               "cuda": meta.get("stats", [])}
        started = time.time()
        try:
            got = run_op(op, payload, meta)
            # An op with several outputs may carry an oracle for only some of
            # them — `native_layer_norm` returns (result, mean, rstd) and the
            # bank records `oracle0` alone. Comparing the ones it recorded is
            # a partial answer; demanding all of them was a refusal that said
            # nothing about the engine, and the count is reported so nobody
            # reads a one-output row as if it covered three.
            row["metal"] = [ulp_distance(g, payload[f"oracle{i}"])
                            for i, g in enumerate(got)
                            if f"oracle{i}" in payload.files]
            row["outputs_compared"] = f"{len(row['metal'])} of {len(got)}"
            row["status"] = "ok"
            # The same expression evaluated in float32 on the host, against the
            # same oracle. It is not a second opinion on the kernel — it is the
            # scale on which to read a ULP gap: where Metal and this agree, the
            # distance to the oracle is the arithmetic's, not the backend's,
            # and a CUDA number closer than both is a luckier summation ORDER
            # rather than a defect on this side.
            row["host_fp32"] = _host_fp32_stats(op, payload)
        except Exception as exc:
            row["status"] = "refused"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}"
            row["traceback_tail"] = traceback.format_exc().splitlines()[-1][:200]
        row["wall_s"] = round(time.time() - started, 4)
        rows.append(row)
        mark = "ok " if row["status"] == "ok" else "REF"
        extra = ""
        if row["status"] == "ok" and row["cuda"] and row["metal"]:
            host = row.get("host_fp32")
            # The bank records no ULP for a boolean or integer output — there
            # is none to record — so these are formatted as values, not as
            # numbers. Formatting a None crashed the run after 14 entries.
            def _n(v):
                return "-" if v is None else str(v)
            extra = (f"metal_ulp={_n(row['metal'][0].get('max_ulp')):<8}"
                     f"cuda_ulp={_n(row['cuda'][0].get('max_ulp')):<8}"
                     f"host_fp32_ulp="
                     f"{_n(host[0].get('max_ulp')) if host else '-'}")
        if mark == "REF":
            extra = (row.get("error") or "")[:96]
        print(f"  {mark} {op:<38} {meta['tag']:<22} {extra}", flush=True)

        for dt in extra_dtypes:
            rows.append(_run_in_dtype(op, payload, meta, path, dt))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    refused = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows)} entries, {len(refused)} refused -> {args.out}")
    return 0


def _host_fp32_stats(op: str, payload):
    """The op in float32 on the host, measured against the bank's oracle.
    ``None`` when no reference is written for this op."""
    arrays = [np.ascontiguousarray(payload[k].astype(np.float32))
              for k in sorted(payload.files) if k.startswith("in")]
    # In float32 THROUGHOUT — the point of the column is the arithmetic's own
    # distance to the oracle at this width, so widening anywhere would just
    # reproduce the oracle and report zero.
    if op in ("mm", "matmul"):
        ref = [arrays[0] @ arrays[1]]
    elif op == "addmm":
        ref = [(arrays[0] + arrays[1] @ arrays[2]).astype(np.float32)]
    else:
        return None
    out = []
    for i, r in enumerate(ref):
        key = f"oracle{i}"
        if key not in payload.files:
            return None
        out.append(ulp_distance(np.asarray(r).astype(np.float32), payload[key]))
    return out


def _fp64_reference(op: str, arrays):
    """The op, written out in float64, for a dtype the bank did not record.

    Only the shapes the GEMV family takes — a matrix-vector product, with or
    without a bias — are written here. An op without a reference REFUSES; a
    number compared against a reference nobody wrote is not a measurement.
    """
    if op in ("mm", "matmul"):
        a, b = arrays[0].astype(np.float64), arrays[1].astype(np.float64)
        return [a @ b]
    if op == "addmm":
        bias = arrays[0].astype(np.float64)
        a, b = arrays[1].astype(np.float64), arrays[2].astype(np.float64)
        return [bias + a @ b]
    raise NotImplementedError(
        f"no float64 reference written for {op!r}; refusing to compare against "
        f"a reference that does not exist")


def _run_in_dtype(op: str, payload, meta, path, dtype):
    """Re-run one bank entry's SEEDED INPUTS in another dtype.

    The inputs are the Dell's; the oracle is recomputed here in float64
    because the bank's belongs to the dtype it was recorded in. There is no
    CUDA column: the Dell did not run this dtype, and inventing one would be
    the whole point of the bank thrown away.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    row = {"op": op, "tag": meta["tag"], "file": path.name,
           "dtype": str(dtype), "source": "bank inputs, oracle recomputed here",
           "launched": meta.get("launched", []), "cuda": None}
    started = time.time()
    try:
        arrays = [np.ascontiguousarray(payload[k].astype(dtype))
                  for k in sorted(payload.files) if k.startswith("in")]
        _require_inputs(op, meta, arrays, path)
        oracle = _fp64_reference(op, arrays)
        handler = _resolve_handler(op, wrappers)
        kwargs = _with_reduction_axis(op, kwargs, arrays, payload)
        kwargs = _derived_kwargs(op, kwargs, arrays, payload, meta)
        if handler is None:
            raise NotImplementedError(f"no bank handler for {op!r}")
        inputs = [NBXTensor.from_numpy(a) for a in arrays]
        out = handler(wrappers, inputs, dict(meta.get("kwargs") or {}))
        outs = out if isinstance(out, (tuple, list)) else [out]
        got = [np.asarray(o.numpy()) for o in outs]
        row["metal"] = [ulp_distance(g, o) for g, o in zip(got, oracle)]
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "refused"
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}"
    row["wall_s"] = round(time.time() - started, 4)
    mark = "ok " if row["status"] == "ok" else "REF"
    extra = (f"metal_ulp={row['metal'][0]['max_ulp']:<8}(no CUDA number)"
             if row["status"] == "ok" else row.get("error", "")[:90])
    print(f"  {mark} {op:<38} {meta['tag'] + ' @' + str(dtype):<22} {extra}",
          flush=True)
    return row



def _require_inputs(op, meta, arrays, path=None):
    """Refuse an entry that records no inputs, by name and with the reason.

    `cat/cat3` sits in the bank's `_unlaunched/` directory with an empty
    `launched` list and no `in*` array at all: on CUDA the op produced its
    output without launching a kernel, so the bank has the result and the
    fp64 oracle but nothing to call the engine WITH. Replaying it anyway
    reached the handler with an empty input list and died inside numpy
    ("operands could not be broadcast together with shapes (0,) (9,5)"), a
    message that names neither the entry nor the reason and reads like an
    engine failure. It is not one.
    """
    if arrays:
        return
    where = ""
    if path is not None and "_unlaunched" in str(path):
        where = " (it sits in the bank's `_unlaunched/` directory)"
    raise NotImplementedError(
        f"the bank records no inputs for {op!r} tag "
        f"{meta.get('tag')!r}{where}: it has the CUDA output and the fp64 "
        f"oracle but nothing to call the engine with, and "
        f"{len(meta.get('launched') or [])} kernels were launched for it on "
        f"CUDA. There is no call to replay")


def run_op(op: str, payload, meta):
    """Run one bank entry through the engine's own wrappers."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    arrays = [np.ascontiguousarray(payload[k])
              for k in sorted(payload.files) if k.startswith("in")]
    _require_inputs(op, meta, arrays)
    inputs = [NBXTensor.from_numpy(a) for a in arrays]
    kwargs = _with_reduction_axis(op, dict(meta.get("kwargs") or {}),
                                  arrays, payload)
    kwargs = _derived_kwargs(op, kwargs, arrays, payload, meta)

    handler = _resolve_handler(op, wrappers)
    if handler is None:
        raise NotImplementedError(
            f"the engine exports no wrapper named {op!r}, {op + '_wrapper'!r} "
            f"or {op + '_forward'!r}; this is a real gap, not a missing lambda")
    out = handler(wrappers, inputs, kwargs)
    outs = out if isinstance(out, (tuple, list)) else [out]
    return [np.asarray(o.numpy()) if hasattr(o, "numpy") else np.asarray(o)
            for o in outs]


#: Ops whose bank entries carry no reduction axis; it is derived from the
#: recorded shapes exactly as for `sum` and `mean`.
_REDUCING = ("sum", "mean", "amax", "amin", "argmax", "argmin", "prod",
             "var", "std", "any", "all", "count_nonzero", "max", "min")


#: Ops whose CALL ARGUMENTS the bank does not record. Running them with the
#: engine's defaults compares an oracle against a call nobody made: `clamp`
#: scored 2755 ULP that way on 2026-09-07, and resolving handlers by name
#: silently un-refused it on 2026-09-08 until the census printed the 2755
#: again. The bank would have to record the bounds, the pad widths, the fill
#: value; until it does these are refused BY NAME with what is missing.
#: The SDPA entries store raw q/k/v arrays and no layout. Where seq_len and
#: head_dim differ the shape settles which layout K is in; where they are
#: equal it cannot, and the engine refuses rather than guess — the correction
#: of 2026-09-07. That refusal reaching this tool is the guard working, not a
#: gap in it, and the table says so.
_ARGUMENTS_NOT_RECORDED = {
    "clamp": "the bounds (min, max)",
    "clamp_min": "the lower bound",
    "clamp_max": "the upper bound",
    "fill": "the fill value",
    "full": "the fill value",
    "constant_pad_nd": "the pad widths and the pad value",
    "rsub": "the scalar and alpha",
}


def _derived_kwargs(op, kwargs, arrays, payload, meta):
    """Parameters the bank does not put in `kwargs` but the arrays do carry.

    Derived and then USED only where the derivation is checkable against a
    recorded shape — the same rule as the reduction axis. Anything that would
    have to be invented is in `_ARGUMENTS_NOT_RECORDED` and refused instead.
    """
    out = dict(kwargs)
    tag = meta.get("tag", "")
    if op == "native_layer_norm" and len(arrays) >= 2:
        # weight has exactly the normalized shape
        out.setdefault("normalized_shape", tuple(arrays[1].shape))
    if op == "native_group_norm" and len(arrays) >= 1:
        x = arrays[0]
        if x.ndim >= 3:
            n, c = int(x.shape[0]), int(x.shape[1])
            hxw = 1
            for d in x.shape[2:]:
                hxw *= int(d)
            groups = None
            found = re.search(r"G(\d+)", tag)
            if found:
                groups = int(found.group(1))
                if c % groups:
                    groups = None       # the tag does not fit the array
            if groups is not None:
                out.setdefault("N", n)
                out.setdefault("C", c)
                out.setdefault("HxW", hxw)
                out.setdefault("num_groups", groups)
    if op in ("cumsum",):
        found = re.search(r"dim(\d+)", tag)
        if found:
            out.setdefault("dim", int(found.group(1)))
        elif arrays and arrays[0].ndim == 1:
            out.setdefault("dim", 0)
    if op.startswith("upsample_") and "out0" in payload.files:
        recorded = payload["out0"].shape
        if len(recorded) >= 2:
            out.setdefault("output_size", tuple(int(d) for d in recorded[-2:]))
    if op in ("_softmax", "_log_softmax", "softmax", "log_softmax"):
        found = re.search(r"dim(-?\d+)", tag)
        if found:
            out.setdefault("dim", int(found.group(1)))
    return out


def _resolve_handler(op: str, wrappers):
    """The engine's own entry point for a bank op, by name.

    The bank names ops as the graph does; the engine exports them as
    `<op>_wrapper` or `<op>`. Sixty of the 325 entries were refused for "no
    bank handler" when the handler was simply the obvious name — a refusal
    that says nothing about the engine and hides everything the entry could
    say. Resolving by name keeps this tool honest as the kernel library
    grows, instead of needing a lambda written for each new op.

    Returns a callable taking (wrappers, inputs, kwargs), or None when the
    engine exports nothing by that name — which IS a real refusal and is
    reported as one.
    """
    missing = _ARGUMENTS_NOT_RECORDED.get(op)
    if missing is not None:
        raise NotImplementedError(
            f"the bank records no {missing} for {op!r}; running it with the "
            f"engine's defaults would compare the oracle against a call that "
            f"was never made")
    explicit = _HANDLERS.get(op)
    if explicit is not None:
        return explicit
    bare = op.lstrip("_")
    for candidate in (f"{op}_wrapper", op, f"{bare}_wrapper", bare,
                      f"{op}_forward", f"{bare}_forward", f"nbx_{op}"):
        fn = getattr(wrappers, candidate, None)
        if callable(fn):
            def handler(w, i, k, _fn=fn):
                clean = {n: v for n, v in k.items() if not n.startswith("_")}
                if "_reduce_dim" in k:
                    clean["dim"] = k["_reduce_dim"]
                    clean["keepdim"] = k["_reduce_keepdim"]
                return _fn(*i, **clean)
            handler.__name__ = f"resolved:{candidate}"
            return handler
    return None


def _with_reduction_axis(op, kwargs, arrays, payload):
    """Add the derived reduction axis for the ops whose bank entries omit it."""
    if op not in _REDUCING:
        return kwargs
    derived = _reduction_axis(arrays[0].shape, payload["out0"].shape)
    if derived is None:
        raise NotImplementedError(
            f"the bank records no reduction axis for {op!r} and the shapes "
            f"{tuple(arrays[0].shape)} -> {tuple(payload['out0'].shape)} do "
            f"not name one unambiguously; refusing to guess it")
    out = dict(kwargs)
    out["_reduce_dim"], out["_reduce_keepdim"] = derived
    return out


def _reduction_axis(in_shape, out_shape):
    """The axis a reduction was taken over, DERIVED from the recorded shapes.

    The bank does not record it — `kwargs` is empty for these entries and the
    axis lives only in the tag. Reading it out of the tag would be guessing,
    and guessing a call's arguments is how `clamp` once scored 2755 ULP
    against an oracle for a call nobody made. The shapes decide it instead:
    exactly one axis of the input, removed (or kept as 1), must give the
    recorded output. When none does, or more than one does, this returns None
    and the entry is refused with that said.

    Returns (dim, keepdim) or None.
    """
    in_shape, out_shape = tuple(in_shape), tuple(out_shape)
    matches = []
    for axis in range(len(in_shape)):
        dropped = in_shape[:axis] + in_shape[axis + 1:]
        kept = in_shape[:axis] + (1,) + in_shape[axis + 1:]
        if out_shape == dropped:
            matches.append((axis, False))
        elif out_shape == kept:
            matches.append((axis, True))
    if len(matches) == 1:
        return matches[0]
    return None


_HANDLERS = {
    "scaled_dot_product_attention":
        lambda w, i, k: w.scaled_dot_product_attention_wrapper(*i[:3], **k),
    "_scaled_dot_product_efficient_attention":
        lambda w, i, k: w.scaled_dot_product_attention_wrapper(*i[:3], **k),
    "mm": lambda w, i, k: w.mm(i[0], i[1]),
    "matmul": lambda w, i, k: w.mm(i[0], i[1]),
    "addmm": lambda w, i, k: w.addmm(i[0], i[1], i[2]),
    "bmm": lambda w, i, k: w.bmm(i[0], i[1]),
    "baddbmm": lambda w, i, k: w.baddbmm_wrapper(i[0], i[1], i[2]),

    # The copy family is exercised THROUGH these: a broadcast binary launches
    # `strided_copy_kernel` to materialise the expanded operand, and the bank
    # records the launch list that proves it. Without them the copy family had
    # 29 of its 32 bank entries refused by this tool — a refusal that says
    # nothing about the engine and hides everything the entries could say.
    "add": lambda w, i, k: w.add(i[0], i[1]),
    "sub": lambda w, i, k: w.sub(i[0], i[1]),
    "mul": lambda w, i, k: w.mul(i[0], i[1]),
    "div": lambda w, i, k: w.div(i[0], i[1]),
    "remainder": lambda w, i, k: w.remainder_wrapper(i[0], i[1]),
    "pow": lambda w, i, k: w.pow_wrapper(i[0], i[1]),
    "maximum": lambda w, i, k: w.maximum_wrapper(i[0], i[1]),
    "minimum": lambda w, i, k: w.minimum_wrapper(i[0], i[1]),
    "eq": lambda w, i, k: w.eq(i[0], i[1]),
    "ne": lambda w, i, k: w.ne(i[0], i[1]),
    "lt": lambda w, i, k: w.lt(i[0], i[1]),
    "le": lambda w, i, k: w.le(i[0], i[1]),
    "gt": lambda w, i, k: w.gt(i[0], i[1]),
    "ge": lambda w, i, k: w.ge(i[0], i[1]),
    "logical_and": lambda w, i, k: w.logical_and_wrapper(i[0], i[1]),
    "logical_or": lambda w, i, k: w.logical_or_wrapper(i[0], i[1]),
    "logical_xor": lambda w, i, k: w.logical_xor_wrapper(i[0], i[1]),
    # Positional signatures the generic resolver cannot satisfy by keyword.
    # `normalized_shape` IS the weight's shape and `N`/`C`/`HxW` ARE the
    # input's, so both are checked against a recorded array; `eps` is the one
    # value neither carries, and the ATen default is used and SAID so rather
    # than the entry being dropped.
    "native_layer_norm": lambda w, i, k: w.native_layer_norm(
        i[0], k["normalized_shape"], i[1] if len(i) > 1 else None,
        i[2] if len(i) > 2 else None, float(k.get("eps", 1e-5))),
    "native_group_norm": lambda w, i, k: w.native_group_norm_wrapper(
        i[0], i[1] if len(i) > 1 else None, i[2] if len(i) > 2 else None,
        k["N"], k["C"], k["HxW"], k["num_groups"], float(k.get("eps", 1e-5))),
    "cat": lambda w, i, k: __import__(
        "neurobrix.kernels.nbx_tensor", fromlist=["NBXTensor"]
    ).NBXTensor.cat(list(i), dim=int(k.get("dim", 0))),
    "index_select": lambda w, i, k: w.index_select_wrapper(
        i[0], int(k.get("dim", 0)), i[1]),
    "sum": lambda w, i, k: w.sum_wrapper(i[0], dim=k["_reduce_dim"],
                                         keepdim=k["_reduce_keepdim"]),
    "mean": lambda w, i, k: w.mean_wrapper(i[0], dim=k["_reduce_dim"],
                                           keepdim=k["_reduce_keepdim"]),
    "convolution": lambda w, i, k: w.conv2d_wrapper(
        i[0], i[1], i[2] if len(i) > 2 else None,
        stride=k.get("stride", 1), padding=k.get("padding", 0),
        dilation=k.get("dilation", 1), groups=int(k.get("groups", 1))),
}


if __name__ == "__main__":
    raise SystemExit(main())
