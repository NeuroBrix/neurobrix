"""The Metal pin under a census shadow: a RECORDED shadow allocation is recognised and left
alone; an address nobody recorded still fails loudly, in a shadow and in a real run.

Measured 2026-09-24 (merged census, 76 runs, 7 MoE containers): every MoE census stopped at its
first MoE layer on

    RuntimeError: the triton-ext driver cannot bind pointer 0x1....: the allocator does not
    record it, so its length is unknown and it cannot be wrapped as a Metal buffer

because `census._shadow_malloc` hands out addresses the allocator never records, and
`pinned_addresses` (moe._build_ptr_tables) sizes its whole-allocation wrap from the allocator.

The repair is deliberately NOT "skip the pin in a census". A path that exists only in the census
would one day let a real, unrecorded address through in silence. Instead the census records its
shadow allocations AS shadows, the pin recognises a recorded shadow and does nothing for it, and
anything else, recorded by nobody, keeps failing in every run. Hence three cells, the third
being the one that keeps the door shut.

`census.install` rewires the allocator for the whole process, so each cell runs in its own
interpreter.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("triton")

SRC = str(Path(__file__).resolve().parents[3] / "src")


def _run(body: str, census: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=SRC, TOOLCHAINS="Metal")
    if census:
        env.update(NBX_CENSUS="1", NBX_CENSUS_DEVICES="1", CUDA_VISIBLE_DEVICES="")
    pre = ("from neurobrix.kernels import census\ncensus.install(hardware='default-9f169c79')\n"
           if census else "")
    code = pre + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=300)


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the pin is the triton-ext (Metal) driver's")
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


def test_a_recorded_shadow_is_pinned_as_nothing():
    """The 3-line repro: a shadow tensor's view goes through the pin without a wrap."""
    _metal_or_skip()
    p = _run("""
        from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
        from neurobrix.triton import triton_ext_driver as drv
        t = NBXTensor.empty((4, 64, 32), dtype=NBXDtype.bfloat16)
        v = t.select(0, 1)
        with drv.pinned_addresses(v):
            a = drv.pinned_gpu_address(v.data_ptr())
        assert a == v.data_ptr(), (hex(a), hex(v.data_ptr()))
        assert not drv._RESIDENT_WRAPS, drv._RESIDENT_WRAPS
        print("SHADOW_PINNED_AS_NOTHING")
    """, census=True)
    assert p.returncode == 0 and "SHADOW_PINNED_AS_NOTHING" in p.stdout, p.stderr[-1500:]


def test_an_unrecorded_address_still_fails_in_a_shadow():
    """In a census, an address the census did not hand out is not a shadow: loud."""
    _metal_or_skip()
    p = _run("""
        from neurobrix.triton import triton_ext_driver as drv
        class Fake:
            def data_ptr(self):
                return 0x7ff0_0000_1000
        try:
            with drv.pinned_addresses(Fake()):
                pass
        except RuntimeError as e:
            assert "does not record it" in str(e), e
            print("LOUD")
    """, census=True)
    assert p.returncode == 0 and "LOUD" in p.stdout, p.stderr[-1500:]


def test_an_unrecorded_address_still_fails_in_a_real_run():
    """No census at all: an address the allocator never recorded is refused, as before."""
    _metal_or_skip()
    p = _run("""
        from neurobrix.triton import triton_ext_driver as drv
        class Fake:
            def data_ptr(self):
                return 0x1_0000_0000_2000   # inside the census's shadow range, but no census
        try:
            with drv.pinned_addresses(Fake()):
                pass
        except RuntimeError as e:
            assert "does not record it" in str(e), e
            print("LOUD")
    """, census=False)
    assert p.returncode == 0 and "LOUD" in p.stdout, p.stderr[-1500:]
