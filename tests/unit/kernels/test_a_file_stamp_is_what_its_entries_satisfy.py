"""A file's format claim is a claim about every entry in it.

2026-09-13 22:40: a certification from a `/2`-era writer added keys to three
`/1` files and stamped the whole file `/2`; 6 344 legacy entries carry no
`built`, so the gate refused the three files and the runtime would have served
none of their 6 633 shapes. The stamp now follows the entries (`format_for`),
and `neurobrix autotune check --restamp` repairs a false claim in place.
Seen red: `format_for` replaced by the old constant stamp → validate refuses.
"""
import json

from neurobrix.kernels import autotune_certified as C

KERNEL = "neurobrix.kernels.ops.matmul.matmul_kernel"
KEY = "(10, 1536, 1536, True, True, 'fp32', 'fp16', 'fp32')"


def _cert(with_built, m=10):
    p = {"date": "2026-09-07T03:53:34+00:00", "engine_version": "0.5.3", "backend": {"name": "cuda"},
         "shape": [m, 1536, 1536, True, True, "fp32", "fp16", "fp32"], "deviation": 1e-6, "tolerance": 1e-4,
         "oracle": "fp64", "machine": {"hardware_profile": "auto-v100-16gb-16g"}}
    if with_built:
        p["built"] = {"gpu": True}
    return {"config": {"kwargs": {"BLOCK_M": 32}, "num_warps": 4, "num_stages": 3}, "proof": p, "excluded": []}


def test_a_mix_of_eras_is_stamped_by_its_oldest_entry():
    assert C.format_for({"a": _cert(True), "b": _cert(False)}) == "nbx-autotune-certified/1"
    assert C.format_for({"a": _cert(True), "b": _cert(True)}) == C.FORMAT
    e = _cert(True); e["variants"] = {"32g": _cert(False)}
    assert C.format_for({"a": e}) == "nbx-autotune-certified/1", "a variant without built is an entry without built"


def test_the_old_stamp_made_the_gate_refuse_and_restamp_repairs_it(tmp_path):
    root = tmp_path / "nvidia" / "volta"; root.mkdir(parents=True)
    path = root / "matmul_kernel.fp32.json"
    entries = {KEY: _cert(False), KEY.replace("10,", "11,"): _cert(True, m=11)}
    doc = {"format": C.FORMAT, "vendor": "nvidia", "profile": "volta", "kernel": KERNEL, "dtype": "fp32", "entries": entries}
    path.write_text(json.dumps(doc))
    assert any("without built" in p for p in C.validate(doc, path)), "the injection: a /2 stamp over a /1 entry is refused"
    assert C.restamp(path) == "nbx-autotune-certified/1"
    doc2 = json.loads(path.read_text())
    assert doc2["entries"] == entries, "entries untouched"
    assert C.validate(doc2, path) == []
    assert C.restamp(path) is None, "a true claim is left alone"
