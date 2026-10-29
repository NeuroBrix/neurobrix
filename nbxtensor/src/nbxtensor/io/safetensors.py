"""safetensors -> NBXTensor primitives (moved from the engine's Triton weight loader, M2): the dtype table of the
format, the parallel read of a tensor's bytes, the host-side conversions to the dtype it is held at, its placement in a
device arena or in page-locked host memory. The loading POLICY — which weight goes to which device or stays on the host,
the dtype a stored weight takes under a compute dtype, whether an fp16 target is widened — is the application's: it
passes each decision in. So are the I/O workers: every function that splits its work takes `io = (workers, executor)`,
the application's pool (its size is a fact of the machine the application owns).

Torch-free.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.dtype.convert import bf16_bits_to_f16, f32_to_bf16_bits
from nbxtensor.shape import contiguous_strides
from nbxtensor.memory.memory_pool import ComponentArena, HostPinnedArena

PAGE = os.sysconf("SC_PAGE_SIZE")


def spans(n: int, unit: int, align: int, workers: int):
    """[start, end) element spans of an n-element run, one per worker, each a multiple of `align`
    elements; a run under one page per worker stays whole (a split costs more than it shares)."""
    if workers < 2 or n * unit < workers * PAGE:
        return [(0, n)]
    step = -(-n // workers)
    step = -(-step // align) * align
    return [(a, min(a + step, n)) for a in range(0, n, step)]


def read_bytes(fd: int, offset: int, nbytes: int, io) -> np.ndarray:
    """nbytes of the file at offset, read by the loader's workers into one buffer (page-aligned
    spans; `preadv` may return short, so each span reads until it is full)."""
    buf = np.empty(nbytes, dtype=np.uint8)
    view = memoryview(buf)

    def _one(span):
        a, b = span
        while a < b:
            got = os.preadv(fd, [view[a:b]], offset + a)
            if got <= 0:
                raise RuntimeError(f"weight read: {nbytes} bytes at offset {offset} of fd {fd} ended "
                                   f"{b - a} bytes short (file truncated?)")
            a += got

    list(io[1].map(_one, spans(nbytes, 1, PAGE, io[0])))
    return buf


def split_across_workers(src: np.ndarray, out: np.ndarray, convert, io) -> np.ndarray:
    """`convert(src[a:b], out=flat_out[a:b])` over the loader's worker spans: one conversion of the library run in
    parallel on a host-held weight — the widening ran on one thread for every host-held bf16 weight (a third of
    Qwen3-30B-A3B's 447 s load, py-spy 2026-10-27). The split is the loader's; the conversion is nbxtensor's."""
    flat_src = np.ascontiguousarray(src).reshape(-1)
    flat = out.reshape(-1)

    def _one(span):
        a, b = span
        convert(flat_src[a:b], out=flat[a:b])

    list(io[1].map(_one, spans(flat_src.size, flat_src.itemsize, 1, io[0])))
    return out


def cast(arr: np.ndarray, want, io) -> np.ndarray:
    """`arr.astype(want)`, C-contiguous, the elements split across the loader's workers."""
    src = np.ascontiguousarray(arr).reshape(-1)
    out = np.empty(arr.shape, dtype=want)
    flat = out.reshape(-1)

    def _one(span):
        np.copyto(flat[span[0]:span[1]], src[span[0]:span[1]], casting="unsafe")

    list(io[1].map(_one, spans(src.size, src.itemsize, 1, io[0])))
    return out


# safetensors dtype → (numpy dtype for reading, NBXDtype, bytes per element)
SF_DTYPE_INFO = {
    "F16":  (np.float16,  NBXDtype.float16,  2),
    "BF16": (None,        NBXDtype.bfloat16, 2),
    "F32":  (np.float32,  NBXDtype.float32,  4),
    "F64":  (np.float64,  NBXDtype.float64,  8),
    "I8":   (np.int8,     NBXDtype.int8,     1),
    "I16":  (np.int16,    NBXDtype.int16,    2),
    "I32":  (np.int32,    NBXDtype.int32,    4),
    "I64":  (np.int64,    NBXDtype.int64,    8),
    "U8":   (np.uint8,    NBXDtype.uint8,    1),
    "BOOL": (np.bool_,    NBXDtype.bool_,    1),
}


def half_in_place(ptr: int, count: int, device_idx: int):
    """The loader's raw bf16 weight buffer at `ptr` rewritten in place as float16 by nbxtensor's one device
    conversion (`ops.wrappers.bf16_to_fp16_gpu` with `out=` the same buffer)."""
    from nbxtensor.ops.elementwise_wrappers import bf16_to_fp16_gpu

    DeviceAllocator.set_device(device_idx)
    DeviceAllocator.ensure_triton_device(device_idx)
    flat = (count,)
    strides = contiguous_strides(flat)
    bits = NBXTensor(ptr, flat, strides, NBXDtype.int16, 'cuda', owns_data=False, device_idx=device_idx)
    half = NBXTensor(ptr, flat, strides, NBXDtype.float16, 'cuda', owns_data=False, device_idx=device_idx)
    bf16_to_fp16_gpu(bits, out=half)


def load_to_pinned_host(
    raw: bytes,
    shape: tuple,
    source_dtype: NBXDtype,
    target_dtype: NBXDtype,
    io,
    host_arena: Optional[HostPinnedArena] = None,
) -> NBXTensor:
    """Decode safetensors raw bytes into a pinned-host NBXTensor.

    Applies the standard dtype remap chain (bf16→fp16, fp16→bf16) via
    numpy before copying to pinned memory. bf16→fp16 uses the same
    uint16-bit reinterpret path the GPU arena loader uses, so the
    resulting values match exactly.
    """
    # Decode source bytes to a numpy array matching target_dtype.
    if source_dtype == NBXDtype.bfloat16:
        # Raw bf16 bits stored as uint16. Stays uint16 if target is
        # bf16, or gets bit-shifted to fp32 then cast to fp16 if target
        # is fp16.
        raw_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        if target_dtype == NBXDtype.float16:
            arr = split_across_workers(raw_u16, np.empty(raw_u16.shape, dtype=np.float16), bf16_bits_to_f16, io)
        else:
            arr = np.ascontiguousarray(raw_u16)
    else:
        np_dtype = {
            NBXDtype.float16: np.float16,
            NBXDtype.float32: np.float32,
            NBXDtype.float64: np.float64,
            NBXDtype.int64: np.int64,
            NBXDtype.int32: np.int32,
            NBXDtype.int16: np.int16,
            NBXDtype.int8: np.int8,
            NBXDtype.uint8: np.uint8,
            NBXDtype.bool_: np.bool_,
        }[source_dtype]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        arr = np.ascontiguousarray(arr)
        if target_dtype == NBXDtype.bfloat16 and source_dtype == NBXDtype.float16:
            # fp16 → bf16, rounded to nearest even (fp16 → fp32 is exact)
            arr = f32_to_bf16_bits(arr.astype(np.float32))
        elif target_dtype == NBXDtype.bfloat16 and source_dtype == NBXDtype.float32:
            # fp32 → bf16 rounded to nearest even, the SAME transform the
            # GPU arena placement uses (`place_in_arena`), so the zero3
            # host-staged copy of a weight is bit-identical to its arena copy.
            # An fp32 encoder (PixArt / CogVideoX T5) staged to a bf16 compute
            # target reached here still 4 bytes wide and overflowed the 2-byte
            # pinned buffer — _NUMPY_FOR has no bf16 entry (numpy cannot
            # represent it). The target itself is decided by
            # stored_dtype_in_compute, the one rule shared with the census
            # shadow, so completing this pair keeps the shadow's dtypes and the
            # real run's identical rather than diverging.
            arr = f32_to_bf16_bits(arr)

    # Any remaining source->target pair: cast through numpy.
    #
    # The two bit-level paths above cover bf16, which numpy cannot represent.
    # Everything else is an ordinary cast, and it MUST happen: before
    # 2026-09-03 only the two bf16 pairs were converted, so a float32 weight
    # with a float16 target reached the copy still 4 bytes wide and wrote
    # `arr.nbytes` into a buffer sized for the TARGET dtype — 1024 bytes into
    # 512. Kokoro-82M's decoder is exactly that case (a (256,) fp32 tensor),
    # and the full-zoo gate recorded it as a 240 s TIMEOUT rather than as the
    # overflow it is, because the run never got far enough to fail.
    _NUMPY_FOR = {
        NBXDtype.float16: np.float16, NBXDtype.float32: np.float32,
        NBXDtype.float64: np.float64, NBXDtype.int64: np.int64,
        NBXDtype.int32: np.int32, NBXDtype.int16: np.int16,
        NBXDtype.int8: np.int8, NBXDtype.uint8: np.uint8,
        NBXDtype.bool_: np.bool_,
    }
    want = _NUMPY_FOR.get(target_dtype)
    if want is not None and arr.dtype != want:
        arr = cast(arr, want, io)

    # Pinned host memory: a slice of the component's one block when the load sized it, else its own block.
    if host_arena is not None:
        from nbxtensor.dtype.nbx_dtype import dtype_size as _dsize
        ptr = host_arena.alloc((math.prod(shape) if shape else 1) * _dsize(target_dtype))
        dst = NBXTensor(ptr, tuple(shape), contiguous_strides(tuple(shape)), target_dtype, 'cpu',
                        owns_data=False, device_idx=0, base=host_arena, pinned=True)
    else:
        dst = NBXTensor.empty_cpu(shape, target_dtype, pinned=True)

    # The copy is sized from the SOURCE and the buffer from the TARGET, so a
    # dtype pair nobody converted is a write past the end of a cudaMallocHost
    # region. That is only caught at all because a pinned range is registered
    # with the driver; the same defect on ordinary host memory corrupts the
    # heap in silence. Refuse here, naming the pair, rather than rely on the
    # driver noticing.
    if arr.nbytes != dst.nbytes():
        raise RuntimeError(
            f"weight staging would overflow: {source_dtype!r} -> "
            f"{target_dtype!r} for shape {tuple(shape)} produced "
            f"{arr.nbytes} bytes for a {dst.nbytes()}-byte buffer. "
            f"This dtype pair has no conversion in load_to_pinned_host — add "
            f"one; never widen the buffer to match."
        )
    DeviceAllocator.memcpy(dst.data_ptr(), arr.ctypes.data,
                           arr.nbytes, kind=0)
    return dst


def host_weight_layout(arr: np.ndarray, krsc: bool):
    """The host array to copy to the device and the element strides of the logical tensor over
    it. A KRSC weight (`launch_keys.conv_weight_krsc`) is written channels-innermost, once, here,
    and stays a logical (K, C, R, S) tensor: the original order is never on the device."""
    if not krsc:
        return arr, contiguous_strides(arr.shape)
    K, C, R, S = arr.shape
    return np.ascontiguousarray(arr.transpose(0, 2, 3, 1)), (R * S * C, 1, S * C, C)


def place_in_arena(raw, shape: tuple, nbx_dtype: NBXDtype, np_dtype, target_dtype: NBXDtype, upcast_this: bool,
                   krsc: bool, arena: ComponentArena, target_dev: int, io) -> NBXTensor:
    """One safetensors tensor's raw bytes (stored as `nbx_dtype`, read through `np_dtype`) placed in `arena` on
    `target_dev` at `target_dtype` — as fp32 when `upcast_this` (pre-Ampere overflow protection of an fp16 target),
    channels-innermost when `krsc` (`host_weight_layout`) — and the NBXTensor over the arena's sub-allocation. The
    current device is the caller's to set."""
    nbytes = len(raw)
    # GPU-accelerated bf16→fp16 path (same buffer, 2 bytes/elem).
    # Only used when we're NOT also upcasting to fp32 — otherwise
    # the staged bf16→fp32 conversion happens via numpy below.
    if (nbx_dtype == NBXDtype.bfloat16
            and target_dtype == NBXDtype.float16
            and not upcast_this):
        ptr = arena.alloc(nbytes)
        arr_u16, strides = host_weight_layout(
            np.frombuffer(raw, dtype=np.uint16).reshape(shape), krsc)
        DeviceAllocator.memcpy(ptr, arr_u16.ctypes.data, nbytes, kind=1)
        half_in_place(ptr, arr_u16.size, target_dev)
        return NBXTensor(ptr, shape, strides, NBXDtype.float16, 'cuda',
                         owns_data=False, device_idx=target_dev)

    # Standard path: numpy → H2D, with optional fp32 upcast.
    if nbx_dtype == NBXDtype.bfloat16:
        # bf16 bits in uint16 containers. If upcast_this, expand
        # them to fp32 via the standard bit-shift: bf16 is the top
        # 16 bits of a fp32, so (bf16_bits << 16) reinterpret-cast
        # as fp32 reproduces the value exactly (no approximation).
        raw_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        if upcast_this:
            u32 = raw_u16.astype(np.uint32) << 16
            arr = np.ascontiguousarray(u32.view(np.float32))
            final_dtype = NBXDtype.float32
        else:
            arr = np.ascontiguousarray(raw_u16)
            final_dtype = target_dtype  # bf16 (kept as uint16)
    else:
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        arr = np.ascontiguousarray(arr)
        if upcast_this:
            # Native fp16 → fp32 widening (exact, no precision loss).
            arr = cast(arr, np.float32, io)
            final_dtype = NBXDtype.float32
        elif (nbx_dtype == NBXDtype.float32
              and target_dtype == NBXDtype.float16):
            # fp32 → fp16 downcast. Required when an encoder ships
            # fp32 on disk (PixArt T5, SDXL text_encoder_2, ...)
            # but the compute path wants fp16. Native's
            # WeightLoader(torch_dtype=fp16) does this implicitly;
            # triton needs to do it explicitly so arena sizing
            # and the kernel-side dtype agree.
            arr = cast(arr, np.float16, io)
            final_dtype = NBXDtype.float16
        elif (nbx_dtype == NBXDtype.float32
              and target_dtype == NBXDtype.bfloat16):
            # fp32 → bf16, rounded to nearest even.
            arr = f32_to_bf16_bits(arr)
            final_dtype = NBXDtype.bfloat16
        elif target_dtype != nbx_dtype:
            # fp16 → bf16, rounded to nearest even (fp16 → fp32 is exact).
            arr = f32_to_bf16_bits(arr.astype(np.float32))
            final_dtype = target_dtype
        else:
            final_dtype = target_dtype

    arr, strides = host_weight_layout(arr, krsc)
    arr_bytes = arr.nbytes
    ptr = arena.alloc(arr_bytes)
    DeviceAllocator.memcpy(ptr, arr.ctypes.data, arr_bytes, kind=1)  # H2D

    # Create NBXTensor wrapping the arena sub-allocation (KRSC: the logical shape over
    # the channels-innermost memory)
    return NBXTensor(ptr, shape, strides, final_dtype, 'cuda',
                     owns_data=False, device_idx=target_dev)
