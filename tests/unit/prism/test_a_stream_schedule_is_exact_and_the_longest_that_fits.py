"""Prism's reading of a temporal stream declaration (`core/prism/temporal_stream.py`): the schedule a request runs is
exact by the conditions the declaration writes, and the longest chunk that fits; the vendor's chunks when declared;
a stale declaration refused.

What each test would do if the code were wrong:
  * a schedule breaking a strided window's alignment accepted (Wan's encoder: a next chunk must be a multiple of 4
    input frames): the encoder's windows would straddle the chunks — the chosen schedule names its broken condition;
  * the choice not the longest fitting chunk: more chunks than needed (R5) — the length is checked;
  * the vendor's remainder put anywhere but the first chunk: CogVideoX's decode is another function (1.7e-1 in Forge's
    proof) — the schedule is checked;
  * a declaration derived from another graph read: the test writes one and expects the refusal by name.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from neurobrix.core.prism import temporal_stream as ts

S = {"type": "symbol", "id": "s1"}


def _decl(**kw):
    """Wan's encoder's conditions: the first chunk's strided windows (k 3, s 2) at 1 + (t - 1) // 2 frames after the
    first downsample, the next chunk's lengths multiples of 2 then of 4."""
    d = {"symbol": "s1", "states": [], "conditions": [
        {"graph": "first", "kind": "window", "k": 3, "s": 2, "op": "conv::12", "extent": S},
        {"graph": "first", "kind": "window", "k": 3, "s": 2, "op": "conv::20",
         "extent": {"type": "add", "left": {"type": "floordiv", "left": {"type": "add", "left": S, "right": -1},
                                             "right": 2}, "right": 1}},
        {"graph": "next", "kind": "multiple", "s": 2, "op": "conv::12", "extent": S},
        {"graph": "next", "kind": "multiple", "s": 2, "op": "conv::20",
         "extent": {"type": "floordiv", "left": S, "right": 2}},
    ]}
    d.update(kw)
    return d


def test_the_conditions_name_what_a_schedule_breaks():
    d = _decl()
    assert ts.broken(d, [5, 4, 4]) == []
    assert ts.broken(d, [5, 8]) == []
    bad = ts.broken(d, [5, 6])
    assert bad and "chunk 1 of 6 frames" in bad[0] and "conv::20" in bad[0]
    assert ts.broken(d, [1]) and ts.broken(d, [3, 4])          # 1 frame has no window; 3 leaves a frame unaligned


def test_the_longest_exact_chunk_that_fits_is_chosen():
    d = _decl()
    fits = lambda sched: max(sched[1:] or [0]) <= 12            # a chunk of more than 12 frames does not fit
    sched = ts.choose(d, 81, fits)
    assert sched is not None and not ts.broken(d, sched), sched
    assert sum(sched) == 81 and max(sched[1:]) == 12, sched
    assert ts.choose(d, 81, lambda s: False) is None


def test_the_vendors_chunks_put_the_remainder_first():
    d = _decl(conditions=[], vendor_chunks={"length": 2, "remainder": "first", "source": ["x"]})
    assert ts.choose(d, 13, lambda s: True) == [3, 2, 2, 2, 2, 2]
    assert ts.choose(d, 12, lambda s: True) == [2, 2, 2, 2, 2, 2]
    assert ts.choose(d, 1, lambda s: True) == [1]


def test_a_declaration_derived_from_another_graph_is_refused(tmp_path):
    comp = tmp_path / "components" / "vae"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps({"ops": {}}))
    decl = {"format": "temporal_stream", "version": 1, "clip_graph_sha256": hashlib.sha256(b"another").hexdigest(),
            "graphs": {"first": "stream_first.json", "next": "stream_next.json"}}
    (comp / "stream.json").write_text(json.dumps(decl))
    with pytest.raises(ts.StreamDeclarationError, match="stale stream"):
        ts.load(tmp_path, "vae")
    decl["clip_graph_sha256"] = hashlib.sha256((comp / "graph.json").read_bytes()).hexdigest()
    (comp / "stream.json").write_text(json.dumps(decl))
    for name in ("stream_first.json", "stream_next.json"):
        (comp / name).write_text("{}")
    assert ts.load(tmp_path, "vae")["dags"] == {"first": {}, "next": {}}


def test_a_vendor_tiling_declaration_that_says_when_is_refused(tmp_path):
    """A vendor declaration is the vendor's reference cut or does not exist (decision 2026-10-17 20:08): one carrying
    `when` — the retired 'memory' path, or 'always' — is refused by name until Forge re-derives it. Red if the loader
    reads it: the plan would take a cut no reference makes."""
    import hashlib
    import json
    import pytest
    from neurobrix.core.prism import temporal_stream as ts
    comp = tmp_path / "components" / "vae"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text("{}")
    decl = {"format": "vendor_tiling", "version": 1, "method": "ramp_crop", "when": "memory",
            "clip_graph_sha256": hashlib.sha256(b"{}").hexdigest(), "source": ["x"]}
    (comp / "vendor_tiling.json").write_text(json.dumps(decl))
    with pytest.raises(ts.StreamDeclarationError, match="carries 'when'"):
        ts.load_vendor_tiling(tmp_path, "vae")
    del decl["when"]
    (comp / "vendor_tiling.json").write_text(json.dumps(decl))
    assert ts.load_vendor_tiling(tmp_path, "vae")["method"] == "ramp_crop"
