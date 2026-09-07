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
            row["metal"] = [ulp_distance(g, payload[f"oracle{i}"])
                            for i, g in enumerate(got)]
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
        oracle = _fp64_reference(op, arrays)
        handler = _HANDLERS.get(op)
        kwargs = _with_reduction_axis(op, kwargs, arrays, payload)
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


def run_op(op: str, payload, meta):
    """Run one bank entry through the engine's own wrappers."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    arrays = [np.ascontiguousarray(payload[k])
              for k in sorted(payload.files) if k.startswith("in")]
    inputs = [NBXTensor.from_numpy(a) for a in arrays]
    kwargs = _with_reduction_axis(op, dict(meta.get("kwargs") or {}),
                                  arrays, payload)

    handler = _HANDLERS.get(op)
    if handler is None:
        raise NotImplementedError(f"no bank handler for {op!r}")
    out = handler(wrappers, inputs, kwargs)
    outs = out if isinstance(out, (tuple, list)) else [out]
    return [np.asarray(o.numpy()) for o in outs]


def _with_reduction_axis(op, kwargs, arrays, payload):
    """Add the derived reduction axis for the ops whose bank entries omit it."""
    if op not in ("sum", "mean"):
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
