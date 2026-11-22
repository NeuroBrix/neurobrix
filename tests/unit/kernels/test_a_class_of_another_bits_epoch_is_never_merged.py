"""`tools/merge_certified_class.py` proves nothing, so it may not carry a proof of another bits epoch.

The library refuses a certified file WHOLE when one proof in it does not name its `BITS_EPOCH`
(`epoch_refusal`). A merge that carried an epoch-less class into a stamped file would turn every
entry of that file into a refusal; a merge that wrote over an epoch-less destination would ship a
file the loader refuses. So a source class of another epoch is refused and counted, and a
destination the merge would leave mixed is not written (qa minor 1224-2).

Injection: the epoch check in `merge_file` removed -> the first test fails (the class is carried).
"""
import json
import subprocess
import sys
from pathlib import Path

from nbxtensor.ops.autotune_certified import BITS_EPOCH, epoch_refusal

TOOL = Path(__file__).resolve().parents[3] / "tools" / "merge_certified_class.py"


def _entry(cls, config, epoch=BITS_EPOCH):
    proof = {"backend": {"triton": "3.8.0", "name": "cuda"}, "machine": {"device": {"memory_mb": cls * 1024}}}
    if epoch is not None:
        proof["bits_epoch"] = epoch
    return {"config": {"BLOCK_M": config}, "proof": proof, "excluded": []}


def _file(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "nbx-autotune-certified/1", "vendor": "nvidia", "profile": "volta",
                                "kernel": "k", "dtype": "fp32", "entries": entries}))


def _run(src, dst, cls):
    r = subprocess.run([sys.executable, str(TOOL), "--from", str(src), "--into", str(dst), "--class", str(cls),
                        "--apply"], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout) if r.stdout.strip().startswith("{") else r.stderr


def test_an_epoch_less_source_class_is_refused_and_counted_and_the_destination_stays_loadable(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"shared": _entry(32, "OLD32", epoch=None), "only_src": _entry(32, "OLD", epoch=None)})
    _file(dst / "k.fp32.json", {"shared": _entry(16, "KEEP16")})
    rc, out = _run(src, dst, 32)
    assert rc == 0, out
    row = out["files"][0]
    assert row["refused_other_bits_epoch"] == 2 and row["moved"] == 0 and row["added"] == 0
    d = json.loads((dst / "k.fp32.json").read_text())["entries"]
    assert set(d) == {"shared"} and "variants" not in d["shared"]
    assert epoch_refusal(d) is None


def test_a_destination_the_merge_would_leave_mixed_is_not_written_and_the_run_says_so(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"shared": _entry(32, "NEW32")})
    _file(dst / "k.fp32.json", {"shared": _entry(16, "OLD16", epoch=None)})
    before = (dst / "k.fp32.json").read_bytes()
    rc, out = _run(src, dst, 32)
    assert rc == 1, out
    assert "bits epoch" in out["files"][0]["not_written"]
    assert (dst / "k.fp32.json").read_bytes() == before


def test_a_stamped_class_still_merges(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"shared": _entry(32, "NEW32")})
    _file(dst / "k.fp32.json", {"shared": _entry(16, "KEEP16")})
    rc, out = _run(src, dst, 32)
    assert rc == 0 and out["files"][0]["moved"] == 1, out
    d = json.loads((dst / "k.fp32.json").read_text())["entries"]["shared"]
    assert d["variants"]["32g"]["config"]["BLOCK_M"] == "NEW32" and epoch_refusal({"shared": d}) is None
