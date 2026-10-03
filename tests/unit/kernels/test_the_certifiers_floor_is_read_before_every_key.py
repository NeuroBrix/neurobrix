"""The certifier's price door adds each key's priced peak to the process's footprint read BEFORE that
key — not to one figure read at the start.

The Mac, 2026-10-03 21:09: the door read ru_maxrss once (425 MiB) while the process sat at 1.1-1.7 GB
between keys, so after a few keys it admitted ~1 GB more than the working-set budget it was given; a
re-prove went from 4 005 to 7 503 MB inside 5 s and its guard killed the pass.

What would this file do if the code were wrong? The floor read once -> every key sees the first
figure, the first test RED; the reader falling back to ru_maxrss or to 0 on Linux -> the second, RED.
"""
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import host_footprint as HF
from neurobrix.kernels import autotune_certified as C
from neurobrix.kernels import autotune_certify as CF


def test_each_key_is_priced_over_the_footprint_read_just_before_it(tmp_path, monkeypatch):
    readings = iter([100 << 20, 1100 << 20, 1700 << 20])      # the process grows between keys
    monkeypatch.setattr(HF, "process_footprint_now", lambda: next(readings))
    monkeypatch.setattr(C, "output_dtype", lambda tuner, key: "fp32")
    monkeypatch.setattr(C, "file_for", lambda *a, **k: tmp_path / "k.fp32.json")
    monkeypatch.setattr(C, "describe_key", lambda tuner, key: str(key))
    monkeypatch.setattr(CF, "_tolerance", lambda *a: 1e-3)
    monkeypatch.setattr(CF, "_num_stages_space", lambda *a: [1])
    floors = []

    def fake_certify_key(qual, tuner, key, tol, rng, **kw):
        floors.append(kw["floor_bytes"])
        raise CF.UnreachableCensusKey("stand-in")             # counted, the loop goes on
    monkeypatch.setattr(CF, "certify_key", fake_certify_key)
    summary = {"unreachable": 0, "skipped": 0, "started": 0.0}
    CF._certify_loop({"q.k": [(1,), (2,), (3,)]}, {"q.k": SimpleNamespace()}, "nvidia", "volta", tmp_path,
                     None, False, False, False, {"memory_mb": 16384, "unified": False}, 16,
                     8 << 30, 0, None, summary, lambda *a: None, None, None)
    assert floors == [100 << 20, 1100 << 20, 1700 << 20]


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="the Linux reader")
def test_the_footprint_is_the_resident_bytes_now_not_the_high_water_mark():
    before = HF.process_footprint_now()
    blob = bytearray(256 << 20)                                 # touched: resident
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    grown = HF.process_footprint_now()
    del blob
    assert grown - before > 200 << 20
    import gc; gc.collect()
    assert HF.process_footprint_now() < grown - (200 << 20)     # a high-water mark never comes down
