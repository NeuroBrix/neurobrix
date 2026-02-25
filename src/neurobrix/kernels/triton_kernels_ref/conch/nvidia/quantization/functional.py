# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Bitsandbytes Triton kernel wrappers."""

from dataclasses import dataclass
from typing import Final, Optional

import torch
import triton

from conch.kernels.quantization.bitsandbytes.dequantize_blockwise import dequantize_blockwise_launcher
from conch.kernels.quantization.bitsandbytes.quantize_blockwise import quantize_blockwise_launcher

SUPPORTED_QUANT_TYPES: Final = ["nf4", "fp4", "fp8"]

SUPPORTED_BLOCKSIZES: Final = [4096, 2048, 1024, 512, 256, 128, 64]

_BYTES_PER_ELEMENT: Final = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.uint8: 1,
    torch.int8: 1,
}

# Mapping from quantization type to representable values
_NAME_TO_QMAP: dict[str, torch.Tensor] = {}


def _create_dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8) -> torch.Tensor:
    """Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """
    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += (max_exponent_bits * means).tolist()
        if signed:
            data += (-max_exponent_bits * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits  # noqa: S101

    gap = 256 - len(data)
    for _ in range(gap):
        data.append(0)

    data.sort()
    return torch.tensor(data, dtype=torch.float32)


@dataclass
class QuantState:
    """Quantization state for bitsandbytes compatibility."""

    absmax: torch.Tensor
    shape: torch.Size
    dtype: torch.dtype
    blocksize: int
    quant_type: str
    code: torch.Tensor | None = None
    offset: float | None = None
    state2: Optional["QuantState"] = None

    @property
    def nested(self) -> bool:
        """Whether or not there is a nested state."""
        return self.state2 is not None


def get_absmax_shape(input_size: int, blocksize: int) -> torch.Size:
    """Get shape of quantized output."""
    num_blocks = triton.cdiv(input_size, blocksize)
    return torch.Size((num_blocks,))


def get_quantized_output_shape(
    input_size: int, quant_type: str, quant_storage: torch.dtype = torch.uint8
) -> torch.Size:
    """Get shape of quantized output."""
    if quant_type == "fp8":
        return torch.Size((input_size,))

    mod = _BYTES_PER_ELEMENT[quant_storage] * 2
    return torch.Size(((input_size + 1) // mod, 1))


def quantize_blockwise(  # noqa: PLR0913
    x: torch.Tensor,
    absmax: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    code: torch.Tensor | None = None,
    blocksize: int = 64,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize input to tensor in blocks."""
    if quant_type not in SUPPORTED_QUANT_TYPES:
        error_msg = f"Unsupported quant_type: {quant_type} ({SUPPORTED_QUANT_TYPES = })"
        raise NotImplementedError(error_msg)

    if blocksize not in SUPPORTED_BLOCKSIZES:
        error_msg = f"Unsupported blocksize: {blocksize} ({SUPPORTED_BLOCKSIZES = })"
        raise NotImplementedError(error_msg)

    input_size = x.numel()

    expected_absmax_shape: Final = get_absmax_shape(input_size, blocksize)
    if absmax is None:
        absmax = torch.zeros(expected_absmax_shape, device=x.device, dtype=torch.float32)

    assert absmax.shape == expected_absmax_shape  # noqa: S101

    expected_out_shape: Final = get_quantized_output_shape(input_size, quant_type, quant_storage)
    if out is None:
        out = torch.zeros(expected_out_shape, device=x.device, dtype=quant_storage)

    assert out.shape == expected_out_shape  # noqa: S101

    out, absmax = quantize_blockwise_launcher(
        x=x, absmax=absmax, out=out, code=code, blocksize=blocksize, input_size=input_size, quant_type=quant_type
    )

    return out, QuantState(
        absmax=absmax, shape=x.shape, dtype=x.dtype, blocksize=blocksize, quant_type=quant_type, code=code
    )


def quantize_4bit(  # noqa: PLR0913
    x: torch.Tensor,
    absmax: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    blocksize: int = 64,
    compress_statistics: bool = False,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize input to tensor in blocks of packed, 4-bit values."""
    out, state = quantize_blockwise(
        x=x, absmax=absmax, out=out, code=None, blocksize=blocksize, quant_type=quant_type, quant_storage=quant_storage
    )

    if compress_statistics:
        absmax = state.absmax
        offset = absmax.mean()
        absmax -= offset
        if "dynamic" not in _NAME_TO_QMAP:
            _NAME_TO_QMAP["dynamic"] = _create_dynamic_map()
        code = _NAME_TO_QMAP["dynamic"].to(x.device)
        qabsmax, state2 = quantize_blockwise(
            x=absmax, absmax=None, out=None, code=code, blocksize=256, quant_type="fp8"
        )
        del absmax
        state = QuantState(
            absmax=qabsmax,
            shape=state.shape,
            dtype=state.dtype,
            blocksize=blocksize,
            code=None,
            quant_type=quant_type,
            offset=offset.item(),
            state2=state2,
        )

    return out, state


def dequantize_blockwise(  # noqa: PLR0913
    x: torch.Tensor,
    quant_state: QuantState | None = None,
    absmax: torch.Tensor | None = None,
    code: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    blocksize: int = 64,
    quant_type: str = "fp4",
) -> torch.Tensor:
    """Dequantize input to tensor in blocks."""
    if quant_type not in SUPPORTED_QUANT_TYPES:
        error_msg = f"Unsupported quant_type: {quant_type} ({SUPPORTED_QUANT_TYPES = })"
        raise NotImplementedError(error_msg)

    if blocksize not in SUPPORTED_BLOCKSIZES:
        error_msg = f"Unsupported blocksize: {blocksize} ({SUPPORTED_BLOCKSIZES = })"
        raise NotImplementedError(error_msg)

    if quant_state is None:
        if absmax is None:
            error_msg = "Must pass either quant_state or absmax!"
            raise ValueError(error_msg)
        if out is None:
            error_msg = "Must pass either quant_state or out!"
            raise ValueError(error_msg)
        if code is None and quant_type == "fp8":
            error_msg = "Must pass either quant_state or code!"
            raise ValueError(error_msg)
    else:
        absmax = quant_state.absmax if absmax is None else absmax
        code = quant_state.code if code is None else code
        if out is None:
            out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=x.device)

    output_size = out.numel()
    num_blocks = triton.cdiv(output_size, blocksize)

    assert absmax.shape == (num_blocks,)  # noqa: S101
    assert out.shape == (output_size,)  # noqa: S101

    return dequantize_blockwise_launcher(
        x=x, absmax=absmax, out=out, code=code, blocksize=blocksize, output_size=output_size, quant_type=quant_type
    )


def dequantize_4bit(  # noqa: PLR0913
    x: torch.Tensor,
    quant_state: QuantState | None = None,
    absmax: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    blocksize: int = 64,
    quant_type: str = "fp4",
) -> torch.Tensor:
    """Quantize input to tensor in blocks of packed, 4-bit values."""
    if quant_state is not None and quant_state.nested:
        assert quant_state.state2 is not None  # noqa: S101 (for mypy)
        assert quant_state.offset is not None  # noqa: S101 (for mypy)
        absmax = dequantize_blockwise(
            x=quant_state.absmax,
            quant_state=quant_state.state2,
            blocksize=quant_state.state2.blocksize,
            quant_type=quant_state.state2.quant_type,
        )
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    return dequantize_blockwise(
        x=x, quant_state=quant_state, absmax=absmax, code=None, out=out, blocksize=blocksize, quant_type=quant_type
    )
