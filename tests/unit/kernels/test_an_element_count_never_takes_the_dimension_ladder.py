"""An element-wise launch's element count is classed by its OWN ladder (`autotune.buckets.elements`), never the
profile's `default` dimension ladder; every vendor profile declares one, with its source.

Only nvidia/volta.yml declared it; every other profile answered the element count with the dimension ladder, and the
Mac's census on main (2026-10-24 12:26) formed 27 868 element-wise classes on apple_m4_pro — five days of certification
for one launch tile — with no error.

What these tests would do if the code were wrong: `ladder_for` falling back to `default` returns a ladder — the first
case expects a refusal; a profile left without its ladder fails the second by name.
"""
from pathlib import Path

import pytest
import yaml

from nbxtensor.ops import autotune_bucket as AB

VENDORS = Path(AB.__file__).resolve().parents[1] / "config" / "vendors"


def test_a_profile_without_an_element_ladder_is_refused_not_given_the_default():
    prof = {"architecture": "x", "autotune": {"buckets": {"default": [{"up_to": None, "step": 1}]}}}
    with pytest.raises(RuntimeError, match="never takes the `default`"):
        AB.ladder_for("elements", prof)
    assert AB.ladder_for("M", prof)                       # a tensor dimension still takes the default


@pytest.mark.parametrize("path", sorted(VENDORS.glob("*/*.yml")), ids=lambda p: p.stem)
def test_every_vendor_profile_declares_its_element_ladder(path):
    doc = yaml.safe_load(path.read_text())
    rows = ((doc.get("autotune") or {}).get("buckets") or {}).get("elements")
    assert rows, f"{path.name}: no autotune.buckets.elements"
    assert AB.ladder_for("elements", doc)
