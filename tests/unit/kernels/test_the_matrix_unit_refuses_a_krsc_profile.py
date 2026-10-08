"""The m8n8k4 convolution reads a filter as one contiguous (C, R, S) run (ops/conv2d_m8n8k4.py takes w.stride(0)
alone), so a profile that declares the matrix unit AND relays 3x3 weights channels-innermost (`conv.weight_layout:
KRSC`, the M4 Pro's) would have that kernel read them in the wrong order. `_configs.matrix_unit` refuses the pair by
name where the two declarations meet. Injection: the refusal removed -> the KRSC cell is RED."""
from pathlib import Path

import pytest
import yaml

from neurobrix.kernels.ops import _configs

VOLTA = Path(__file__).resolve().parents[3] / "src/neurobrix/config/vendors/nvidia/volta.yml"


def _bound(monkeypatch, layout):
    prof = yaml.safe_load(VOLTA.read_text())
    assert prof.get("matrix_unit"), "volta.yml declares the matrix unit this cell speaks of"
    prof["conv"] = {"weight_layout": layout}
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: prof)


def test_the_unit_with_kcrs_weights_is_served(monkeypatch):
    _bound(monkeypatch, "KCRS")
    assert _configs.matrix_unit().get("mm")


def test_the_unit_with_krsc_weights_is_refused_by_name(monkeypatch):
    _bound(monkeypatch, "KRSC")
    with pytest.raises(ValueError, match="conv.weight_layout 'KRSC'.*reads KCRS weights only"):
        _configs.matrix_unit()
