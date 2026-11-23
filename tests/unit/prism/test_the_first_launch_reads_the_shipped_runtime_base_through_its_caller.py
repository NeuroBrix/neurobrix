"""qa 1263-1: the CALLER's wiring — `ensure_measured_facts` reads the launched engine's runtime base from the vendor
file's shipped figure (through `other_card_models.shipped_or_sibling_base`) and starts no measuring child. Red if the
caller goes back to the siblings alone (the mutant qa ran: it survived the earlier tests)."""
from __future__ import annotations

import pytest
import yaml

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import facts_probe as P
from neurobrix.core.prism import other_card_models as O

CARD = {"brand": "apple", "model": "Card", "architecture": "anyarch", "supports_dtypes": ["float16"], "memory_mb": 16384,
        "index": 0}


@pytest.fixture
def hw(tmp_path, monkeypatch):
    import nbxtensor
    from neurobrix.core.prism import loader
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(P, "matmul_tflops", lambda t, *a, **k: {"best": 10.0})
    monkeypatch.setattr(P, "copies", lambda *a, **k:
                        {"memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0, "h2d_pageable_gbps": 5.0})
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda i: {})
    monkeypatch.setattr(P, "ruled_chunk", lambda d, memory_mb: 64 << 20)
    shipped = {"runtime_base_mb": {"triton": 354, "compiled": 187}, "runtime_base_library": nbxtensor.__version__}
    monkeypatch.setattr(loader, "vendor_block", lambda d, k: shipped if k == "measured" else {})
    monkeypatch.setattr(O, "base_from_siblings", lambda *a, **k: {})
    measured = []
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda engine: measured.append(engine) or 999)
    (tmp_path / "default-one.yml").write_text(yaml.safe_dump({"id": "default-one", "devices": [CARD], "cpu": {}}))
    return tmp_path, measured


def test_the_caller_takes_the_shipped_figure_and_measures_nothing(hw):
    path, measured = hw
    A.ensure_measured_facts("default-one", engine="triton")
    cpu = yaml.safe_load((path / "default-one.yml").read_text())["cpu"]
    assert cpu["runtime_base_mb"] == {"triton": 354} and measured == []
