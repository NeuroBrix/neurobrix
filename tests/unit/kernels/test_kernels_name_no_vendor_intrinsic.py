"""A portable kernel may not call one vendor's device library.

`tl.extra.CUDA.libdevice.*` is NVIDIA's device library. Any other backend fails
to lower it AT COMPILE TIME, so a kernel that calls it runs on exactly one
vendor's hardware however portable the rest of it is. It would fail on AMD for
the same reason it fails on Apple.

Measured on triton-ext, 2026-09-17, with bare kernels containing no engine code
(`repro_pow_libdevice_cuda_namespace.py`):

    tl.extra.cuda.libdevice.pow   FAILED   PassManager::run failed
    portable exp/log/sign         COMPILED max rel err 2.522e-07

Kokoro-82M walked into this four times in a row — aten.pow, aten.round,
aten.angle, each a different intrinsic — and could not run at all until every
one was replaced. Triton 3.8.0 exposes no portable `pow`, `round`, `rint`,
`nearbyint` or `atan2`, so the semantics are built in our kernels and checked
against numpy.

The vendored reference tree is exempt: `triton_kernels_ref/**/nvidia/**` is
third-party code in an explicitly NVIDIA-named directory, and is not what the
engine dispatches on a Metal device.
"""
from __future__ import annotations

import pathlib

import pytest

OPS = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "kernels" / "ops"


def test_no_ops_kernel_calls_the_cuda_device_library():
    offenders = []
    for f in sorted(OPS.rglob("*.py")):
        text = f.read_text()
        for n, line in enumerate(text.splitlines(), 1):
            if "tl.extra.cuda.libdevice." in line and not line.lstrip().startswith("#"):
                offenders.append(f"{f.relative_to(OPS.parent.parent.parent)}:{n}: {line.strip()[:90]}")
    assert not offenders, (
        "these kernels call NVIDIA's device library and cannot compile on any "
        "other backend:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("name", ["pow", "floor", "atan2", "vector_norm"])
def test_the_replaced_kernels_still_exist(name):
    """Guard against a 'fix' that deletes the op rather than porting it."""
    assert (OPS / f"{name}.py").exists()
