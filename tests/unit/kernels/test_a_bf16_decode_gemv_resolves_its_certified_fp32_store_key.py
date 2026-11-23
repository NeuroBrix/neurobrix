"""A bf16 decode GEMV (`mv`, `addmv`) on the Apple M4 Pro keys its fp32 STORE: the library's dtype rule keeps a decode
row's fp32 store on every hardware (nbxtensor ops/launch_keys.py `matmul_out_dtype`, lines 95-100: "A DECODE ROW
(M <= 4) keeps its fp32 store on every hardware", `if half and (M <= 4 ...): return F32`) — intended, with its two
measured reasons (MiniCPM-o "The!!!!!", whisper-large-v3-turbo's Triton decoder drift). So the bf16-store keys of
addmv_kernel.bf16.json (51) and gemv_vec_kernel.bf16.json (68) are formed by no call (the Apple re-prove of 23/11
found every one of them UNREACHABLE) and are retired; every launch they stood for resolves a key the fp32 file
certifies.

Red if a bf16 decode launch resolved a key no shipped file certifies, or if the bf16-store files were still shipped."""
import json
import pathlib

import pytest
import yaml

import neurobrix

ROOT = pathlib.Path(neurobrix.__file__).parent / "config" / "autotune" / "apple" / "apple_m4_pro"


def _keys(name):
    return {tuple(eval(k)) for k in json.loads((ROOT / name).read_text())["entries"]}


@pytest.fixture
def apple(monkeypatch):
    """The M4 Pro's own device file in force: its buckets, natively bf16."""
    import nbxtensor
    from nbxtensor.ops import _configs
    path = pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices" / "apple" / "apple_m4_pro.yml"
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: yaml.safe_load(path.read_text()))


def test_every_bf16_decode_addmv_resolves_a_certified_fp32_store_key(apple):
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    from nbxtensor.ops import launch_keys as LK
    certified = _keys("addmv_kernel.fp32.json")
    rows = [k for k in certified if k[2:5] == ("bf16", "bf16", "bf16")]
    assert len(rows) == 51
    for n, m, *_ in rows:
        (_, key), = LK.addmv_launches(n, m, D.bfloat16, D.bfloat16, D.bfloat16, True)
        assert key in certified and key[-1] == "fp32", key


def test_every_bf16_decode_gemv_resolves_a_certified_fp32_store_key(apple):
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    from nbxtensor.ops import launch_keys as LK
    certified = _keys("gemv_vec_kernel.fp32.json")
    rows = [k for k in certified if k[2] == "bf16"]
    for n, k, mat, vec, _out in rows:
        (_, key), = LK.mv_launches(n, k, D.bfloat16, getattr(D, {"bf16": "bfloat16", "fp32": "float32"}[vec]), True)
        assert key in certified and key[-1] == "fp32", key


def test_the_bf16_store_files_are_not_shipped():
    assert not (ROOT / "addmv_kernel.bf16.json").exists()
    assert not (ROOT / "gemv_vec_kernel.bf16.json").exists()
