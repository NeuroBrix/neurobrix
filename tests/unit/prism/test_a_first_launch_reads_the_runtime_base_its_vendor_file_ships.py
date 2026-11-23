"""A first launch READS the launched engine's runtime base from the vendor file's `measured.runtime_base_mb` where the
card model ships one (owner reset 22/11 point 1; first-launch step 4), else a sibling profile's, and measures it only
where neither exists. Red if `shipped_or_sibling_base` ignores the vendor figure."""
from neurobrix.core.prism import other_card_models as O


def test_the_vendor_figure_is_read_before_any_sibling_or_measure(monkeypatch, tmp_path):
    from neurobrix.core.prism import loader
    monkeypatch.setattr(loader, "vendor_block", lambda d, k: {"runtime_base_mb": {"compiled": 187, "triton": 354}})
    monkeypatch.setattr(O, "base_from_siblings", lambda *a, **k: {"triton": 999})
    assert O.shipped_or_sibling_base({}, tmp_path / "p.yml", "triton") == 354
    assert O.shipped_or_sibling_base({}, tmp_path / "p.yml", "compiled") == 187


def test_without_a_vendor_figure_a_sibling_answers_else_none(monkeypatch, tmp_path):
    from neurobrix.core.prism import loader
    monkeypatch.setattr(loader, "vendor_block", lambda d, k: {})
    monkeypatch.setattr(O, "base_from_siblings", lambda *a, **k: {"triton": 300})
    assert O.shipped_or_sibling_base({}, tmp_path / "p.yml", "triton") == 300
    assert O.shipped_or_sibling_base({}, tmp_path / "p.yml", "compiled") is None
