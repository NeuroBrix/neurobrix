"""NBXDtype and its tables: the dtype enum NBXTensor carries, its byte sizes, the names it parses (a NeuroBrix or a
torch spelling, read as text), the numpy and array-interface maps. Cut out of tensor.py as it was (M2, 2026-10-29); the
dtype area reads it from the one table next (the Mac's M2)."""

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


# Bytes per element of every NBXDtype: the library's own, beside the enum it defines (the library
# boundary: nbx_tensor imports no engine module). The engine's table (config/dtypes.yml) is held
# equal to it member by member by tests/unit/dtype/test_the_two_itemsize_readers_are_one.py.
_DTYPE_SIZES = {
    NBXDtype.float16: 2, NBXDtype.bfloat16: 2,
    NBXDtype.float32: 4, NBXDtype.float64: 8,
    NBXDtype.int8: 1, NBXDtype.int16: 2, NBXDtype.int32: 4, NBXDtype.int64: 8,
    NBXDtype.uint8: 1, NBXDtype.bool_: 1,
    NBXDtype.complex64: 8, NBXDtype.complex128: 16,
}

_DTYPE_FROM_STR = {
    'float16': NBXDtype.float16, 'fp16': NBXDtype.float16, 'half': NBXDtype.float16,
    'bfloat16': NBXDtype.bfloat16, 'bf16': NBXDtype.bfloat16,
    'float32': NBXDtype.float32, 'fp32': NBXDtype.float32, 'float': NBXDtype.float32,
    'float64': NBXDtype.float64, 'fp64': NBXDtype.float64, 'double': NBXDtype.float64,
    'int8': NBXDtype.int8, 'int16': NBXDtype.int16,
    'int32': NBXDtype.int32, 'int': NBXDtype.int32,
    'int64': NBXDtype.int64, 'long': NBXDtype.int64,
    'uint8': NBXDtype.uint8, 'byte': NBXDtype.uint8,
    'bool': NBXDtype.bool_,
    'complex64': NBXDtype.complex64, 'complex128': NBXDtype.complex128,
}

# numpy dtype string → NBXDtype
_NUMPY_DTYPE_MAP = {
    'float16': NBXDtype.float16, 'float32': NBXDtype.float32,
    'float64': NBXDtype.float64, 'int8': NBXDtype.int8,
    'int16': NBXDtype.int16, 'int32': NBXDtype.int32,
    'int64': NBXDtype.int64, 'uint8': NBXDtype.uint8,
    'bool': NBXDtype.bool_,
    # Complex: numpy stores complex as interleaved [real, imag] pairs — exactly
    # the NBXTensor complex storage convention (complex64 = fp32 pairs,
    # complex128 = fp64 pairs). Tagging the raw-byte copy with the complex dtype
    # therefore yields a valid complex NBXTensor (view_as_real/_complex reinterpret
    # the same bytes). Without these entries from_numpy fell through to float32,
    # mistagging complex constants (e.g. Wan RoPE freqs, complex128 [seq, hd/2])
    # as garbage floats -> NaN through the RoPE rotation -> dead output.
    'complex64': NBXDtype.complex64, 'complex128': NBXDtype.complex128,
}

# __cuda_array_interface__ typestr
_DTYPE_TYPESTR = {
    NBXDtype.float16: '<f2', NBXDtype.bfloat16: '<V2',
    NBXDtype.float32: '<f4', NBXDtype.float64: '<f8',
    NBXDtype.int8: '<i1', NBXDtype.int16: '<i2',
    NBXDtype.int32: '<i4', NBXDtype.int64: '<i8',
    NBXDtype.uint8: '<u1', NBXDtype.bool_: '|b1',
    NBXDtype.complex64: '<c8', NBXDtype.complex128: '<c16',
}

_FLOATING_DTYPES = frozenset({
    NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64,
})


_COMPLEX_DTYPES = frozenset({NBXDtype.complex64, NBXDtype.complex128})


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
