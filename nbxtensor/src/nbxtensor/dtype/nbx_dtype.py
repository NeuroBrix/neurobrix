"""NBXDtype and its tables: the dtype enum NBXTensor carries (its values frozen: new members are appended, never
renumbered), and the tables derived from the ONE dtype table (`dtypes.yml`): byte sizes, the spellings it parses (a
table name, a launch-key spelling, an alias such as torch's `half`; a `torch.` prefix read as text), the numpy names
and the array-interface type strings, the floating and complex kinds."""

from __future__ import annotations

from enum import IntEnum

# ============================================================================
# DTYPE SYSTEM
# ============================================================================

class NBXDtype(IntEnum):
    float16 = 0
    bfloat16 = 1
    float32 = 2
    float64 = 3
    int8 = 4
    int16 = 5
    int32 = 6
    int64 = 7
    uint8 = 8
    bool_ = 9
    complex64 = 10
    complex128 = 11


def _table_name(m: "NBXDtype") -> str:
    """A member's name in the dtype table (`bool_` is Python's keyword escape of `bool`)."""
    return "bool" if m is NBXDtype.bool_ else m.name


def _derive():
    """Every table of the enum, read from the one dtype table (`dtypes.yml`): widths, the spellings a caller may name
    a dtype by, the numpy names, the array-interface type strings, the floating and complex kinds."""
    from .itemsize import dtype_table
    entries = dtype_table()["dtypes"]
    sizes, names, numpy_names, typestr, floating, complex_ = {}, {}, {}, {}, set(), set()
    for m in NBXDtype:
        e = entries[_table_name(m)]
        sizes[m] = int(e["bits"]) // 8
        for spelling in [_table_name(m), *(e.get("autotune_keys") or ()), *(e.get("aliases") or ())]:
            names[spelling] = m
        if "numpy" in e:
            numpy_names[e["numpy"]] = m
        typestr[m] = e["typestr"]
        if e["kind"] == "float":
            floating.add(m)
        elif e["kind"] == "complex":
            complex_.add(m)
    return sizes, names, numpy_names, typestr, frozenset(floating), frozenset(complex_)


# Read once from the table: no width, spelling or type string is written here (R-NBXT 2, one fact once).
_DTYPE_SIZES, _DTYPE_FROM_STR, _NUMPY_DTYPE_MAP, _DTYPE_TYPESTR, _FLOATING_DTYPES, _COMPLEX_DTYPES = _derive()


def dtype_size(dtype: NBXDtype) -> int:
    return _DTYPE_SIZES[dtype]


def parse_dtype(d) -> NBXDtype:
    """Parse dtype from string or NBXDtype."""
    if isinstance(d, NBXDtype):
        return d
    if isinstance(d, str):
        s = d.replace('torch.', '')
        return _DTYPE_FROM_STR[s]
    # Unknown type — try string conversion
    s = str(d).replace('torch.', '')
    return _DTYPE_FROM_STR[s]
