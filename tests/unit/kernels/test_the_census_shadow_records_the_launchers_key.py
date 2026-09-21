"""The certification census (kernels/census.py) runs the engine as a SHADOW — no device
memory, no launch, no value read — and records every key the dispatch hands the launcher.

What this test would do if the code were wrong: the record file stays empty (the seam in
`_configs.run_with_notice` is the only writer; remove it and `mm` forms its key in silence),
or the shadow reaches the driver (an unshadowed allocator method raises error 100 with no
device visible — the door this test runs behind). Seen failing on 2026-09-21: the first
shadow died in `_drain_device` on `DeviceAllocator.device_synchronize`, which the shadow
did not yet cover.

The shape (19, 2048, 2048): TinyLlama's chat-templated "Haiku" prefill, the key the live
served run and the replay cache both hold — the first proof target of the census.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

SCRIPT = r'''
import os
from neurobrix.kernels import census
census.install()
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator as DA
from neurobrix.kernels import wrappers as W
DA.device_synchronize()               # the shadow answers the whole driver surface
a = NBXTensor.empty((19, 2048), "float16", 2)
b = NBXTensor.empty((2048, 2048), "float16", 2)
out = W.mm(a, b)
print("OUT", tuple(out.shape), out.dtype)
'''


def _run(tmp_path, extra_env):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "4",
                "PYTHONPATH": os.pathsep.join(p for p in [env.get("PYTHONPATH", ""),
                                                          os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")] if p)})
    env.update(extra_env)
    return subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)


def test_the_shadow_records_the_key_the_live_launcher_forms(tmp_path):
    rec = tmp_path / "keys"
    r = _run(tmp_path, {"NBX_KEY_RECORD": str(rec)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OUT (19, 2048) fp16" in r.stdout, r.stdout
    lines = rec.read_text().splitlines() if rec.exists() else []
    assert lines, "the shadow formed no key: the record seam is gone"
    assert lines[0].startswith("neurobrix.kernels.ops.matmul.matmul_kernel::(19, 2048, 2048,"), lines


def test_without_a_record_path_the_shadow_writes_nothing(tmp_path):
    rec = tmp_path / "keys"
    r = _run(tmp_path, {"NBX_KEY_RECORD": ""})
    assert r.returncode == 0, r.stderr[-2000:]
    assert not rec.exists()


def test_without_the_census_the_engine_is_not_a_shadow(tmp_path):
    """The default path: NBX_CENSUS unset means nothing is installed — the allocator, the
    launcher and the kernels' `run` are the engine's own."""
    script = r'''
from neurobrix.kernels import census
from neurobrix.kernels import wrappers  # defines the autotuned kernels
from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
from neurobrix.kernels import launcher
assert census.active() is False
assert DA.malloc_cuda.__name__ == "malloc_cuda", DA.malloc_cuda
assert launcher.launch.__name__ == "launch", launcher.launch
print("NOT A SHADOW")
'''
    env = dict(os.environ)
    env.pop("NBX_CENSUS", None)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_KEY_RECORD": "",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "NOT A SHADOW" in r.stdout


def test_the_shadow_refuses_while_a_device_is_visible(tmp_path):
    """The door: with a card in sight an unshadowed path would succeed on real hardware in
    silence, so install() refuses at entry. Seen failing (the refusal) 2026-09-21 with
    CUDA_VISIBLE_DEVICES=0 on the rack; on a machine with no card this test cannot see the
    door and is skipped rather than read as green."""
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    if DeviceAllocator.device_count() < 1:
        pytest.skip("no device visible: the door cannot be exercised here")
    script = "from neurobrix.kernels import census\ncensus.install()\n"
    env = dict(os.environ)
    env.update({"NBX_CENSUS": "1",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")})
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert "census shadow refused" in r.stderr, r.stderr[-1500:]
