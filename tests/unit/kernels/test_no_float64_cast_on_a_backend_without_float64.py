"""On a backend that declares no float64 (Metal), NBXTensor.to() refuses a cast from or to float64 by name instead of
returning what the backend's silent narrowing makes of it.

Why: `triton/metal_backend.py` records that the Metal lowering narrows f64 silently, "not a loud refusal", and that a
future f64 expression would be narrowed just as silently with nothing to catch it. On 2026-09-28 that future arrived:
the Sana 4K positional embedding, placed as float64 and cast to bf16 on Metal, came back 1 139 of 143 360 bits right.
A census says "not this time"; this door says "never". Seen red (garbage returned, nothing raised) before the door."""
import numpy as np
import pytest
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, _detect_gpu_backend


def test_a_float64_cast_is_refused_by_name_where_the_backend_has_no_float64():
    from neurobrix.kernels import nbx_tensor as T
    name = _detect_gpu_backend()
    if name is None:
        pytest.skip("no device")
    if getattr(T, "_BACKEND_HAS_FP64", {}).get(name, True):
        pytest.skip(f"backend {name} has float64: the door is not on its path")
    x = NBXTensor.from_numpy(np.linspace(-1.0, 1.0, 4096, dtype=np.float64))
    with pytest.raises(RuntimeError, match="float64"):
        x.to(NBXDtype.bfloat16)
    y = NBXTensor.from_numpy(np.linspace(-1.0, 1.0, 4096, dtype=np.float32))
    with pytest.raises(RuntimeError, match="float64"):
        y.to(NBXDtype.float64)
