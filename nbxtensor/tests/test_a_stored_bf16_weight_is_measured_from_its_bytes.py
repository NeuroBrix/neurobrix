"""`io.safetensors.bf16_abs_maxima`: every bfloat16 tensor of a safetensors file measured (key, element count, largest
magnitude) from its stored 16-bit patterns, other dtypes and the metadata skipped. Prism's float16-safety scan asks it
(moved from the engine's solver, which mapped the shard and viewed its bytes itself, 2026-11-12)."""
import json
import struct

import numpy as np

from nbxtensor.dtype.convert import f32_to_bf16_bits
from nbxtensor.io.safetensors import bf16_abs_maxima


def _write(path, tensors):
    header, blobs, at = {"__metadata__": {"format": "pt"}}, [], 0
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return header, 8 + len(h)


def test_each_bf16_tensor_gives_its_largest_magnitude_and_nothing_else_is_read(tmp_path):
    a = np.array([[1.0, -7.5e4], [3.0, np.nan]], dtype=np.float32)
    b = np.array([0.5, -2.0, 1.0], dtype=np.float32)
    p = tmp_path / "w.safetensors"
    header, start = _write(p, [("a", "BF16", a.shape, f32_to_bf16_bits(a).tobytes()),
                               ("f", "F32", (2,), np.array([1e9, 2.0], np.float32).tobytes()),
                               ("b", "BF16", b.shape, f32_to_bf16_bits(b).tobytes())])
    got = {k: (n, m) for k, n, m in bf16_abs_maxima(p, header, start)}
    assert set(got) == {"a", "b"}
    assert got["a"][0] == 4 and abs(got["a"][1] - 7.5e4) / 7.5e4 < 1e-2      # bf16 rounding of 75000; NaN ignored
    assert got["b"] == (3, 2.0)
