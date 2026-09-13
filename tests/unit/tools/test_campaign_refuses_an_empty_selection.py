"""A campaign that selects no model refuses at entry.

2026-09-13: the budget-unified gate's byte matrix called the campaign with
`--machine` and neither `--models` nor `--family`. The default selection keeps
the cached models whose family equals `--family`; unset, that is nothing. The
tool printed a header, an empty table and the script printed "gate termine" —
a green over zero cells (vacuous-gates register entry 52).

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_refuses_an_empty_selection.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool():
    p = Path(__file__).resolve().parents[3] / "tools" / "precision_zoo_campaign.py"
    spec = importlib.util.spec_from_file_location("pzc", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_models_and_no_family_is_refused_with_the_flags_named(tmp_path):
    t = _tool()
    (tmp_path / "A").mkdir(); (tmp_path / "A" / "manifest.json").write_text('{"family": "image"}')
    why = t.empty_selection_refusal([], models_arg=None, family=None)
    assert why and "--models" in why and "--family" in why, why


def test_a_family_nobody_has_is_refused_too():
    t = _tool()
    why = t.empty_selection_refusal([], models_arg=None, family="video")
    assert why and "video" in why, why


def test_a_named_list_passes():
    t = _tool()
    assert t.empty_selection_refusal(["X"], models_arg="X", family=None) is None
