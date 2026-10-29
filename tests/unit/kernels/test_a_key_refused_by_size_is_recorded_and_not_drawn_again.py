"""A census key the certifying card cannot hold is refused by size BEFORE any draw, at the capacity Prism plans
against, and the refusal is written to the class's register so a pass that only fills what is missing does not draw
it again.

Sana_1600M_4Kpx's 16g census holds rms_norm 16 252 928 rows x 128 fp32: input and output 7.75 GiB each, 15.5 GiB —
under the 16 GiB card, so the door (which priced against the WHOLE card) let it through, and the output's
allocation failed with 7.9 GB free, at every pass of the certification run beside the census (2026-10-23: passes 1
to 21 on card 0, "2 failed" each time).

What these tests would do if the code were wrong: with the door at the whole card, the first case draws and the
test fails on the raise it expects; without the register, the loop calls `certify_key` again on the second pass and
the third case fails on its count; a register that skipped on any bound would also skip after the card's capacity
changed, which the fourth case pins.
"""
from __future__ import annotations

import types

import pytest

from neurobrix.kernels import autotune_certify as AC

RMS = "neurobrix.kernels.ops.rms_norm.rms_norm_forward_kernel"
KEY = (16252928, 128, True, False, "fp32", "fp32", "fp32")
CARD = 16 * 2**30


def test_the_door_prices_in_and_out_against_the_capacity_prism_plans_against():
    with pytest.raises(AC.KeyTooLargeForClass) as e:
        AC._refuse_oversize(CARD, [((KEY[0], KEY[1]), "fp32"), ((KEY[0], KEY[1]), "fp32")], "rms_norm_forward_kernel")
    assert e.value.asked == 2 * KEY[0] * KEY[1] * 4 and e.value.asked < CARD   # under the whole card ...
    assert e.value.usable < e.value.asked and e.value.card == CARD             # ... over what a plan holds
    assert AC.oversize_for_class(e.value) == (e.value.asked, CARD)
    AC._refuse_oversize(CARD, [((1024, 128), "fp32"), ((1024, 128), "fp32")], "rms_norm_forward_kernel")   # fits


def test_the_register_round_trips_and_refuses_a_line_it_cannot_read(tmp_path):
    p = tmp_path / "refused.16g.jsonl"
    assert AC.read_refused(p) == {}
    AC.record_refusal(p, RMS, repr(KEY), 10, 9, 16, "too large")
    AC.record_refusal(p, RMS, repr(KEY), 11, 9, 16, "too large again")          # replaced, not added
    rows = AC.read_refused(p)
    assert list(rows) == [(RMS, repr(KEY))] and rows[(RMS, repr(KEY))]["asked_bytes"] == 11
    p.write_text(p.read_text() + "not json\n")
    with pytest.raises(ValueError, match="not a refusal line"):
        AC.read_refused(p)


def _loop(monkeypatch, tmp_path, memory_mb, calls):
    monkeypatch.setattr(AC.C, "output_dtype", lambda tuner, key: "fp32")
    monkeypatch.setattr(AC.C, "file_for", lambda v, p, q, d, root=None: tmp_path / v / p / f"{q.rsplit('.', 1)[-1]}.{d}.json")
    monkeypatch.setattr(AC.C, "entry_covers", lambda *a, **k: False)
    monkeypatch.setattr(AC, "_file_entries", lambda per, d, path: {})
    monkeypatch.setattr(AC, "_tolerance", lambda *a: 1e-4)
    monkeypatch.setattr(AC, "_num_stages_space", lambda *a: [3])
    monkeypatch.setattr(AC, "_release_between_keys", lambda: None)

    def certify_key(qual, tuner, key, tol, rng, card_bytes=None, **k):
        calls.append(key)
        AC._refuse_oversize(card_bytes, [((key[0], key[1]), "fp32"), ((key[0], key[1]), "fp32")], "rms")
        raise AssertionError("drawn")

    monkeypatch.setattr(AC, "certify_key", certify_key)
    summary = {"certified": 0, "skipped": 0, "failed": 0, "unreachable": 0, "excluded_configs": 0,
               "kernels": {}, "started": 0.0}
    dev = {"memory_mb": memory_mb, "name": "card", "ordinal": 0, "visible_devices": "0"}
    return AC._certify_loop({RMS: [KEY]}, {RMS: object()}, "nvidia", "volta", tmp_path, None, True, False, False,
                            dev, 16, None, None, None, summary, lambda _m: None, types.SimpleNamespace(add=None), None)


def test_a_pass_that_fills_what_is_missing_does_not_draw_a_refused_key_again(needs_device, monkeypatch, tmp_path):
    calls = []
    first = _loop(monkeypatch, tmp_path, 16384, calls)
    assert first["oversize"] == 1 and first["failed"] == 0 and len(calls) == 1
    reg = AC.read_refused(AC.refused_path("nvidia", "volta", 16, root=tmp_path))
    assert (RMS, AC.C.key_repr(KEY)) in reg
    second = _loop(monkeypatch, tmp_path, 16384, calls)
    assert second.get("refused_known") == 1 and len(calls) == 1, (second, calls)


def test_a_refusal_made_at_another_bound_is_tried_again(needs_device, monkeypatch, tmp_path):
    calls = []
    _loop(monkeypatch, tmp_path, 16384, calls)
    again = _loop(monkeypatch, tmp_path, 32768, calls)                 # the class's capacity moved: drawn again
    assert len(calls) == 2 and again.get("refused_known", 0) == 0
