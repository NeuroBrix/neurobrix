"""Standing gate: a brick that replaces a call site preserves that site's observability.

THE RULE
--------
An interceptor, a fusion, a proxy, a tiling wrapper — anything that stands in for
an op — inherits the watchers attached to what it replaced. If it cannot produce
what the original produced, it produces the record itself. Silence is not
permitted, because silence is indistinguishable from correctness.

WHY IT IS A GATE AND NOT A NOTE
-------------------------------
Three times, in three dresses:

1. the decode replay watched a call site the engine had replaced, and lost x9.7
   of throughput while staying byte-correct;
2. a harness metric read a key its brick never emitted, so an image gate could
   never pass (`test_harness_metric_keys.py`);
3. Prism's op-level tiling fused an upsample with its conv; the interceptor
   returns a `FusionUpsampleProxy` that computes nothing, and the per-op recorder
   dropped it on `if not isinstance(tensor, torch.Tensor): return` — the SAME
   guard that legitimately skips a tuple-returning op. Two of the three spatial
   upsamples of the Wan VAE, the two of highest resolution, produced no record.

The family is one family: an observer attached to a site something else now
occupies. It makes the instrumentation lie BY CONSTRUCTION.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
AUDIT = REPO / "tools" / "observability_gap_audit.py"


def test_the_audit_exists() -> None:
    assert AUDIT.is_file(), f"the observability gate is missing: {AUDIT}"


def test_a_replacement_reports_what_it_replaced() -> None:
    """`FusionUpsampleProxy` answers for the upsample it stands in for.

    The summary must be EXACT and free: a nearest upsample only replicates, so
    l2(up(x)) == sqrt(sh*sw) * l2(x) and the first row is the input's first row
    with each element repeated sw times. Nothing is materialized — which is the
    whole reason the proxy exists.
    """
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "src"))
    from neurobrix.kernels.ops.fused_upsample_conv import (      # noqa: E402
        FusionUpsampleProxy)

    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    proxy = FusionUpsampleProxy(x, 2.0, 2.0, (2, 3, 8, 10))
    assert callable(getattr(proxy, "nbx_observable_summary", None)), (
        "a replacement that cannot report itself re-opens the blind spot")
    s = proxy.nbx_observable_summary()

    assert s["shape"] == [2, 3, 8, 10]
    assert s.get("synthetic") is True and s.get("replaced_by")
    # exact, not approximate
    expected_l2 = float(torch.linalg.vector_norm(x.reshape(-1))) * 2.0   # sqrt(2*2)
    assert abs(s["l2_norm"] - expected_l2) < 1e-3, (s["l2_norm"], expected_l2)
    # first row: each input value repeated sw=2 times
    assert s["head10"] == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]


def test_an_unset_slot_does_not_silence_the_summary() -> None:
    """The proxy's optional __slots__ fields are not always set by the interceptor.

    Reading one eagerly (as a `getattr` default) raised AttributeError, the
    recorder swallowed it, and the site stayed silent — the defect wearing the
    fix's clothes.
    """
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "src"))
    from neurobrix.kernels.ops.fused_upsample_conv import (      # noqa: E402
        FusionUpsampleProxy)

    proxy = FusionUpsampleProxy(torch.randn(1, 2, 3, 4), 2.0, 2.0, (1, 2, 6, 8))
    assert not hasattr(proxy, "dtype") or True      # may legitimately be unset
    s = proxy.nbx_observable_summary()              # must not raise
    assert "l2_norm" in s and "error" not in s, s


def _fake_container(tmp: Path, ops: dict, order: list) -> Path:
    nbx = tmp / "model.nbx"
    with zipfile.ZipFile(nbx, "w") as z:
        z.writestr("components/c/graph.json",
                   json.dumps({"ops": ops, "execution_order": order}))
    return nbx


def _dump(tmp: Path, uids: list) -> Path:
    p = tmp / "dump.jsonl"
    p.write_text("\n".join(
        json.dumps({"engine": "compiled",
                    "record": {"component": "c", "op_uid": u, "op_type": "aten::up"}})
        for u in uids))
    return p


def test_the_gate_fires_when_a_site_goes_silent_while_its_siblings_speak(tmp_path) -> None:
    ops = {f"aten.up::{i}": {"op_type": "aten::up", "output_tensor_ids": ["o"],
                             "output_shapes": [[1, 2]], "parent_module": f"m{i}"}
           for i in range(3)}
    nbx = _fake_container(tmp_path, ops, list(ops))
    dump = _dump(tmp_path, ["aten.up::0"])          # 1 of 3 recorded
    r = subprocess.run([sys.executable, str(AUDIT), "--container", str(nbx),
                        "--dump", str(dump)], capture_output=True, text=True, timeout=120)
    assert "      BLIND aten.up::" in r.stdout, r.stdout
    assert "2 site(s) BLIND" in r.stdout, r.stdout
    assert r.returncode != 0, "the gate reported the gap but did not fail"


def test_the_gate_is_silent_on_a_type_that_records_nowhere(tmp_path) -> None:
    """A folded weight transpose records nowhere and is not a blinded site.

    Without this discriminator the gate reported 474 `aten::t` ops on one video
    request — crying wolf about the very thing it audits.
    """
    ops = {f"aten.t::{i}": {"op_type": "aten::t", "output_tensor_ids": ["o"],
                            "output_shapes": [[4, 4]], "parent_module": f"w{i}"}
           for i in range(3)}
    ops["aten.mul::0"] = {"op_type": "aten::mul", "output_tensor_ids": ["o"],
                          "output_shapes": [[4, 4]], "parent_module": "m"}
    nbx = _fake_container(tmp_path, ops, list(ops))
    p = tmp_path / "dump.jsonl"
    p.write_text(json.dumps({"engine": "compiled",
                             "record": {"component": "c", "op_uid": "aten.mul::0",
                                        "op_type": "aten::mul"}}))
    r = subprocess.run([sys.executable, str(AUDIT), "--container", str(nbx),
                        "--dump", str(p)], capture_output=True, text=True, timeout=120)
    assert "      BLIND " not in r.stdout, r.stdout
    assert "0 site(s) BLIND" in r.stdout, r.stdout
    assert "records NOWHERE" in r.stdout, r.stdout
    assert r.returncode == 0


def test_a_tuple_returning_op_is_not_a_blind_spot(tmp_path) -> None:
    """`split` declares 3 outputs and SDPA 4; the recorder's single-tensor guard
    skips them by design, and the container is what says so."""
    ops = {"aten.split::0": {"op_type": "aten::split",
                             "output_tensor_ids": ["a", "b", "c"],
                             "output_shapes": [[1], [1], [1]]},
           "aten.mul::0": {"op_type": "aten::mul", "output_tensor_ids": ["o"],
                           "output_shapes": [[4, 4]]}}
    nbx = _fake_container(tmp_path, ops, list(ops))
    p = tmp_path / "dump.jsonl"
    p.write_text(json.dumps({"engine": "compiled",
                             "record": {"component": "c", "op_uid": "aten.mul::0",
                                        "op_type": "aten::mul"}}))
    r = subprocess.run([sys.executable, str(AUDIT), "--container", str(nbx),
                        "--dump", str(p)], capture_output=True, text=True, timeout=120)
    assert "      BLIND " not in r.stdout, r.stdout
    assert "0 site(s) BLIND" in r.stdout, r.stdout
    assert r.returncode == 0
