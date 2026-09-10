#!/usr/bin/env python3
"""Do the two engines load the same constant? A CPU differential, no card needed.

A graph's constants are stored as a base64 `torch.save` archive. The ATen branch calls
`torch.load`, which reconstructs the tensor from its pickled SHAPE AND STRIDES and then hands
it to the DtypeEngine. The Triton branch cannot import torch (R33), so it parses the ZIP and
views the raw storage with numpy — a reinterpretation that is only equal to the first when the
saved tensor is C-contiguous over exactly its own storage, and that carries no dtype policy.

Every way those two can differ is a SILENT one: no exception, no gate, just a constant whose
values or dtype are not what the oracle used. This walks every constant of every model and
reports each difference with its kind.

    python3 tools/constant_load_differential.py            # the whole local cache
    python3 tools/constant_load_differential.py --models Kokoro-82M,Wan2.1-T2V-1.3B-Diffusers

torch is imported here as the ORACLE, which is what tools/ is for; nothing under src/ does.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int32": 4,
                "int64": 8, "int8": 1, "uint8": 1, "bool": 1, "complex64": 8,
                "complex128": 16}
_NP = {"float16": np.float16, "float32": np.float32, "float64": np.float64,
       "int32": np.int32, "int64": np.int64, "int8": np.int8, "uint8": np.uint8,
       "bool": np.bool_, "complex64": np.complex64, "complex128": np.complex128}


def triton_side(b64: str, tdata: dict, dag: dict):
    """The Triton branch's reading of one constant, as `_load_constant_triton` does it."""
    shape = tuple(tdata.get("shape", []))
    dtype_str = str(tdata.get("dtype", "float32")).replace("torch.", "")
    raw_zip = base64.b64decode(b64)
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = [n for n in zf.namelist() if n.startswith("archive/data/")]
        if not names:
            return None, "no data file in the archive (the engine returns, binding nothing)"
        n_files = len(names)
        tensor_bytes = zf.read(sorted(names)[0])
    dtype_bytes = _DTYPE_BYTES.get(dtype_str, 4)
    declared = int(np.prod(shape)) if shape else 1
    actual = len(tensor_bytes) // dtype_bytes
    note = f"{n_files} storage(s)" if n_files > 1 else ""
    if shape and declared != actual and actual > 0:
        sym = (dag.get("symbolic_context") or {}).get("symbols", {})
        tsl = next((si.get("trace_value") for si in sym.values()
                    if si.get("name") == "seq_len"), None)
        fixed = None
        if tsl:
            for axis in range(len(shape)):
                other = declared // shape[axis] if shape[axis] else 0
                if other and other * tsl == actual:
                    fixed = list(shape)
                    fixed[axis] = tsl
                    break
        if fixed is None and len(shape) == 1:
            if actual >= declared:
                fixed = list(shape)
                tensor_bytes = tensor_bytes[:declared * dtype_bytes]
            else:
                fixed = [actual]
        if fixed is None:
            return None, f"declared {shape} ({declared}) vs storage {actual}: no reconcile"
        shape = tuple(fixed)
    if dtype_str == "bfloat16":
        arr = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(shape)
        return ("bf16", arr), note
    np_dt = _NP.get(dtype_str, np.float32)
    try:
        arr = np.frombuffer(tensor_bytes, dtype=np_dt).reshape(shape)
    except ValueError as e:
        return None, f"reshape refused: {e}"
    return (dtype_str, np.ascontiguousarray(arr)), note


def torch_side(b64: str):
    """The ATen branch's reading: torch.load, shape AND strides from the pickle."""
    import torch
    t = torch.load(io.BytesIO(base64.b64decode(b64)), map_location="cpu", weights_only=True)
    return t


def compare(name: str, b64: str, tdata: dict, dag: dict):
    import torch
    t = torch_side(b64)
    got, note = triton_side(b64, tdata, dag)
    if got is None:
        return f"REFUSED   {name}: {note}"
    kind, arr = got
    if kind == "bf16":
        ref = t.view(torch.uint16).numpy() if t.dtype == torch.bfloat16 else None
        if ref is None:
            return f"DTYPE     {name}: graph says bfloat16, archive holds {t.dtype}"
    else:
        ref = t.numpy()
    if tuple(ref.shape) != tuple(arr.shape):
        return (f"SHAPE     {name}: torch {tuple(ref.shape)} vs triton {tuple(arr.shape)}"
                + (f"  [{note}]" if note else ""))
    if not t.is_contiguous():
        return f"STRIDES   {name}: the saved tensor is NOT contiguous {tuple(t.stride())}"
    same = np.array_equal(ref, arr)
    if not same:
        d = np.abs(ref.astype(np.float64) - arr.astype(np.float64)) if ref.dtype.kind in "fiu" else None
        worst = float(np.nanmax(d)) if d is not None and d.size else float("nan")
        return f"VALUES    {name}: {int((ref != arr).sum())} of {ref.size} differ, max |d| {worst:g}"
    if note:
        return f"NOTE      {name}: {note}, values equal"
    return None


def walk(model_dir: Path):
    out = []
    for graph in sorted(model_dir.glob("components/*/graph.json")):
        dag = json.loads(graph.read_text())
        comp = graph.parent.name
        for tid, tdata in (dag.get("tensors") or {}).items():
            if not tdata.get("constant"):
                continue
            b64 = tdata.get("constant_data")
            if not b64:
                continue
            try:
                r = compare(f"{comp}/{tdata.get('weight_name') or tid}", b64, tdata, dag)
            except BaseException as e:                    # a constant that neither side reads
                r = f"ERROR     {comp}/{tid}: {type(e).__name__}: {str(e)[:110]}"
            if r:
                out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(Path.home() / ".neurobrix" / "cache"))
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    cache = Path(args.cache)
    names = ([m for m in args.models.split(",") if m] if args.models
             else sorted(p.name for p in cache.iterdir() if (p / "manifest.json").exists()))
    total = 0
    for name in names:
        rows = walk(cache / name)
        if rows:
            print(f"=== {name}")
            for r in rows:
                print("   " + r)
            total += len(rows)
        else:
            print(f"    {name}: every constant reads the same on both branches")
    print(f"\n{total} finding(s) over {len(names)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
