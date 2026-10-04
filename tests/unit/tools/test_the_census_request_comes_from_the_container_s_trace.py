"""The census shadows a model at the request DERIVED from the current container's trace, and
refuses a request table that disagrees with it — by name, before any shadow runs.

Measured (supervisor, 2026-09-28 17:22): Sana_1600M_1024px_MultiLing was retraced and its trace
moved to 960x1088. The census took its request from a static table (`requests.json`, 768x1024,
derived from the OLD container) while the regression matrix derived 704x1088 from the new
container's trace: 72 keys were certified for a request the verification never makes, and the
zero-miss verification met 42 uncertified keys. The rule: after any retrace, a model's census is
re-run on the NEW container at the request derived from the new trace — never a stale table.

The fake container below is traced at 960x1088 (a latent 30x34 under a VAE scale of 32), so the
derivation the matrix uses (height at three quarters on the image lattice, width kept) gives
704x1088. The three sizes and the fake's arithmetic are Sana's own.

What this does if the code were wrong (seen RED on 093fcc88's tool, 2026-09-28):
  * no table: the old census shadowed the bare family request (`request_args`, no size flag),
    i.e. the TRACE size 960x1088 the matrix never runs — `test_the_default_request_is_derived`
    fails on the missing `--height 704 --width 1088`;
  * a stale table (768x1024): the old census shadowed the table's size silently —
    `test_a_stale_table_is_refused_by_name` fails because a shadow ran and no refusal exists;
  * a table with no size on an image model: shadowed at the trace size — same failure;
  * a table equal to the derived request passes on both, as it should.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import certified_census as CC  # noqa: E402

MODEL = "FakeImage-960x1088"
# computed by hand from the fake's trace: the image family's confirmation.size_fraction 0.5 -> 480x544,
# the height at three quarters -> 360, both on the 64 lattice -> 320x512 (704x1088 before the
# confirmation request, 2026-09-28; the refusal's logic is unchanged, only the size it names)
DERIVED = ["--height", "320", "--width", "512"]
STALE = ["--height", "768", "--width", "1024"]        # the old container's derivation


def _container(root: Path) -> Path:
    """An image container traced at 960x1088: the smallest set the runtime loader accepts."""
    from neurobrix.nbx.neurotax import NEUROTAX_VERSION
    d = root / MODEL
    (d / "runtime").mkdir(parents=True)
    (d / "components" / "transformer").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "model_name": MODEL, "family": "image", "vae_scale_factor": 32, "trace_resolution": 1024,
        "neurotax_version": NEUROTAX_VERSION}))
    (d / "topology.json").write_text(json.dumps({
        "components": {"transformer": {"shapes": {"hidden_states": [2, 32, 30, 34]}}}}))
    (d / "runtime" / "variables.json").write_text("{}")
    (d / "runtime" / "defaults.json").write_text("{}")
    (d / "components" / "transformer" / "graph.json").write_text("{}")
    return d


@pytest.fixture
def census(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _container(cache)
    monkeypatch.setenv("NEUROBRIX_CACHE", str(cache))
    monkeypatch.setattr(CC, "CACHE", cache)
    monkeypatch.setattr(CC._zoo, "CACHE", cache)
    monkeypatch.setattr(CC, "_device_count", lambda hardware: 1)
    calls = []

    def fake_shadow(model, request, mode, hardware, n_dev, timeout, log_dir, rung_mb=0, tag="", walk_extents=False):
        calls.append({"request": list(request), "tag": tag})
        return {"mode": mode, "rung_mb": rung_mb, "rc": 0, "wall_s": 0.0, "keys": [], "op_keys": [], "error": "", "command": ""}

    monkeypatch.setattr(CC, "shadow", fake_shadow)
    logs = tmp_path / "logs"
    logs.mkdir()

    def run(requests=None, extra=()):
        row = CC.census_model(MODEL, "test_profile", ["triton"], list(extra), requests, 60, logs, [])
        return row, calls

    return run


def _size(req):
    out = []
    for flag in ("--height", "--width"):
        if flag in req:                      # the last occurrence wins, as argparse reads it
            last = max(i for i, a in enumerate(req) if a == flag)
            out += [flag, req[last + 1]]
    return out


def _ordinary(calls):
    return [c["request"] for c in calls if c["tag"] == ""]


def test_the_default_request_is_derived(census):
    row, calls = census(None)
    ords = _ordinary(calls)
    assert ords, f"no ordinary shadow ran: {row}"
    assert _size(ords[0]) == DERIVED, (
        f"the census shadowed {ords[0]} — not the size derived from the container's trace {DERIVED}")
    assert row["status"] == "ok", row


def test_a_table_equal_to_the_derivation_is_accepted(census):
    req = ["--prompt", "a red apple on a wooden table", "--seed", "42", "--steps", "20", *DERIVED]
    row, calls = census([req])
    assert row["status"] == "ok", row
    assert _ordinary(calls) == [req], "the table's request (prompt and seed included) is the one shadowed"


def test_a_stale_table_is_refused_by_name(census):
    req = ["--prompt", "a red apple on a wooden table", "--seed", "42", "--steps", "20", *STALE]
    row, calls = census([req])
    assert calls == [], f"a shadow ran on a request the container's trace does not give: {calls}"
    assert row["status"] == "refused", row
    msg = row.get("refusal", "")
    assert MODEL in msg and "768x1024" in msg and "320x512" in msg, msg
    assert "trace_request.py" in msg, f"the refusal does not say how to regenerate the table: {msg}"


def test_a_table_without_a_size_is_refused_for_a_spatial_model(census):
    row, calls = census([["--prompt", "a red apple on a wooden table", "--seed", "42"]])
    assert calls == [] and row["status"] == "refused", row
    assert "320x512" in row.get("refusal", ""), row


def test_an_extra_size_flag_is_refused(census):
    row, calls = census(None, extra=STALE)
    assert calls == [] and row["status"] == "refused", row


def test_a_plan_refused_below_the_top_rung_is_not_a_failed_model(monkeypatch, tmp_path):
    """Janus-Pro-7B and CogVideoX-5b-I2V, 2026-09-28: the plan refused at the 4 GB rung, every other
    rung censused — the model read `failed`, and the table kept its OLD rows. A refusal below the top
    rung is an arithmetic answer; at the top rung it stays a failure. (Seen red with the old rule.)"""
    import certified_census as CC2

    def fake_shadow(model, req, mode, hw, n_dev, timeout, log_dir, rung_mb=0, tag="", walk_extents=False):
        if rung_mb == 4096:
            return {"mode": mode, "rung_mb": rung_mb, "rc": 1, "wall_s": 0.0, "keys": [], "op_keys": [],
                    "error": "UNEXPECTED ERROR: This model cannot run on this machine.", "command": ""}
        return {"mode": mode, "rung_mb": rung_mb, "rc": 0, "wall_s": 0.0, "keys": ["k::(1,)"], "op_keys": [],
                "error": "", "command": ""}

    monkeypatch.setattr(CC2, "shadow", fake_shadow)
    monkeypatch.setattr(CC2, "_family", lambda m: "llm")
    monkeypatch.setattr(CC2, "_graph_sha", lambda m: "s")
    monkeypatch.setattr(CC2, "frozen_dims", lambda m: [])
    monkeypatch.setattr(CC2, "_device_count", lambda hw: 1)
    monkeypatch.setattr(CC2, "_tiling_probe", lambda *a, **k: None)
    # `M` has no container: the derivation reads a container's own confirmation values (2026-10-04)
    monkeypatch.setattr(CC2._trace, "container_topology", lambda m: {})
    row = CC2.census_model("M", "hw", ["triton"], [], [["--prompt", "x"]], 60, tmp_path, rungs=[4096, 8192])
    assert row["status"] == "ok", row["status"]
    monkeypatch.setattr(CC2, "shadow", lambda *a, **k: {**fake_shadow(*a, **k), "rc": 1,
                                                        "error": "UNEXPECTED ERROR: This model cannot run on this machine."})
    assert CC2.census_model("M", "hw", ["triton"], [], [["--prompt", "x"]], 60, tmp_path, rungs=[4096, 8192])["status"] == "failed"
