"""The dtype table and its reader — the ONE place a dtype's width is written (`dtypes.yml`, beside this module).

Every width a dtype has is read from `dtypes.yml` through this module, and through no other reader: NeuroBrix's two
engine branches read it here.

Torch-free and numpy-free, at import and at execution.

A dtype is named as the code passes it today: a name ("float16"), a torch spelling ("torch.float16", as a traced
graph writes some — read as text, torch is never imported), or an enum member read by its NAME (NBXDtype is an IntEnum,
and `str()` of one is its bare number on Python 3.11+ — never parsed). A name the table does not carry is refused by
name, never given a width.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path
from enum import Enum
from typing import Any, Dict, Tuple

_SOURCE = "nbxtensor/dtype/dtypes.yml"
_TABLE = Path(__file__).resolve().parent / "dtypes.yml"
_TORCH_PREFIX = "torch."
_KINDS = ("float", "int", "uint", "bool", "complex")


@functools.lru_cache(maxsize=1)
def _raw_table() -> Dict[str, Any]:
    import yaml
    if not _TABLE.is_file():
        raise FileNotFoundError(f"ZERO FALLBACK: the dtype table is not found: {_TABLE}")
    table = yaml.safe_load(_TABLE.read_text()) or {}
    for section in ("dtypes", "packed"):
        if not isinstance(table.get(section), dict):
            raise KeyError(f"ZERO FALLBACK: {_TABLE} carries no {section!r} section")
    return table


def dtype_table() -> Dict[str, Any]:
    """The whole table as written (`dtypes`, `packed`, and every other section), a copy: the reader a client keeps of
    its own is fed from here, never from a second file."""
    return copy.deepcopy(_raw_table())


@functools.lru_cache(maxsize=1)
def _tables() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    table = _raw_table()
    dense, packed = dict(table["dtypes"]), dict(table["packed"])
    for name, entry in dense.items():
        bits = entry.get("bits")
        if not isinstance(bits, int) or bits <= 0 or bits % 8:
            raise ValueError(f"ZERO FALLBACK: {_SOURCE} gives dtype {name!r} {bits!r} bits — a dense "
                             f"dtype is a whole number of bytes")
        if entry.get("kind") not in _KINDS:
            raise ValueError(f"ZERO FALLBACK: {_SOURCE} gives dtype {name!r} kind {entry.get('kind')!r} — one of "
                             f"{', '.join(_KINDS)}")
    for name, entry in packed.items():
        for key in ("bits", "group_size", "scale_dtype", "min_dtype"):
            if key not in entry:
                raise KeyError(f"ZERO FALLBACK: {_SOURCE} packed encoding {name!r} states no {key}")
    safetensors = {e["safetensors"]: n for n, e in dense.items() if "safetensors" in e}
    keys = {k: n for n, e in dense.items() for k in (e.get("autotune_keys") or ())}
    return dense, packed, safetensors, keys


def canonical_name(dtype: Any) -> str:
    """The table's name for `dtype` (not checked against the table: `itemsize` refuses)."""
    if isinstance(dtype, Enum):
        name = dtype.name
        if name.endswith("_"):                 # `bool_`: Python's keyword escape, not a dtype name
            name = name[:-1]
    elif isinstance(dtype, str):
        name = dtype
    elif type(dtype).__module__ == "torch":    # a torch.dtype: its str() is its torch spelling
        name = str(dtype)
    else:
        raise TypeError(f"ZERO FALLBACK: {dtype!r} ({type(dtype).__name__}) is not a dtype name, "
                        f"a torch dtype or a dtype enum member ({_SOURCE})")
    return name[len(_TORCH_PREFIX):] if name.startswith(_TORCH_PREFIX) else name


def _dense_entry(dtype: Any) -> Tuple[str, Dict[str, Any]]:
    dense, packed, _, _ = _tables()
    name = canonical_name(dtype)
    if name in dense:
        return name, dense[name]
    if name in packed:
        raise ValueError(f"ZERO FALLBACK: {name!r} is a packed encoding of {packed[name]['bits']} bits "
                         f"per element ({_SOURCE}) and has no whole itemsize — its bytes are "
                         f"representation_bytes({name!r}, numel)")
    raise ValueError(f"ZERO FALLBACK: unknown dtype {dtype!r}: {_SOURCE} has no entry {name!r} "
                     f"(known: {', '.join(sorted(dense))}). Add it there with its source; no width "
                     f"is ever guessed.")


def known_dtypes() -> Tuple[str, ...]:
    """Every dense dtype the table carries, sorted."""
    return tuple(sorted(_tables()[0]))


def itemsize(dtype: Any) -> int:
    """Bytes per element of `dtype`, refused by name when the table does not carry it."""
    return _dense_entry(dtype)[1]["bits"] // 8


def kind(dtype: Any) -> str:
    """The class of `dtype`: float, int, uint, bool or complex (the table's `kind`)."""
    return _dense_entry(dtype)[1]["kind"]


def finite_max(dtype: Any) -> float:
    """The largest finite value of a floating `dtype` (the table's `max`); any other kind is refused by name."""
    name, entry = _dense_entry(dtype)
    if entry.get("kind") != "float":
        raise ValueError(f"ZERO FALLBACK: {name!r} is not a floating dtype: it has no finite maximum")
    if "max" not in entry:
        raise KeyError(f"ZERO FALLBACK: {_SOURCE} states no max for {name!r}")
    return float(entry["max"])


def itemsize_table() -> Dict[str, int]:
    """{name: bytes per element} over every dense dtype (a copy: the caller may not edit the table)."""
    return {name: entry["bits"] // 8 for name, entry in _tables()[0].items()}


def from_safetensors(code: str) -> str:
    """The dtype name of a safetensors header code ("BF16" -> "bfloat16"), refused by name."""
    names = _tables()[2]
    if code not in names:
        raise ValueError(f"ZERO FALLBACK: safetensors dtype code {code!r} has no entry in {_SOURCE} "
                         f"(known: {', '.join(sorted(names))})")
    return names[code]


def from_autotune_key(key: str) -> str:
    """The dtype name of a launch-key spelling ("fp16" -> "float16", "int1" -> "bool")."""
    names = _tables()[3]
    if key not in names:
        raise ValueError(f"ZERO FALLBACK: launch-key dtype {key!r} has no entry in {_SOURCE} "
                         f"(known: {', '.join(sorted(names))})")
    return names[key]


def storage_dtype(dtype: Any, stores_fp64: bool) -> str:
    """The dtype a tensor of `dtype` is STORED in by a store that does (`stores_fp64`) or does not
    hold 64-bit floats: float64 -> float32 and complex128 -> complex64 without, unchanged with."""
    name, entry = _dense_entry(dtype)
    if stores_fp64:
        return name
    return entry.get("without_fp64", name)


def representation_bytes(dtype: Any, numel: int, split: bool = False) -> int:
    """Bytes `numel` elements of `dtype` occupy as represented: a dense dtype at its itemsize, a
    hi+lo SPLIT operand as two tensors of `dtype` (its low dtype), a packed encoding as its packed
    values plus one scale and one minimum per group."""
    numel = int(numel)
    packed = _tables()[1]
    name = canonical_name(dtype)
    if name in packed:
        if split:
            raise ValueError(f"ZERO FALLBACK: a packed encoding ({name!r}) is not split hi+lo")
        e = packed[name]
        groups = -(-numel // int(e["group_size"]))
        return -(-numel * int(e["bits"]) // 8) + groups * (itemsize(e["scale_dtype"]) + itemsize(e["min_dtype"]))
    return numel * itemsize(dtype) * (2 if split else 1)
