"""The gate checks the staged .nbx file it judges, and names every kernel key the arms swept.

verdict.json's `judged.nbx_sha256` is the sha256 the judge computes on the staged file build.json names: a file whose
sha256 (or byte count) is not build.json's, or that is absent, is REFUSED BY NAME and the verdict is red — also when
a green verdict was written before the file changed. Every key the shared replay cache holds that the frozen
autotune directory does not (a missing kernel key, swept at runtime) is counted and named in the gate's state, in
verdict.json and in the log line, and turns the gate RED (the ruling: the gate's own request is a conventional size,
where every key is certified): the verdict names every key and the remedy. Zero missing keys leave the gate green.
What this file does if the judge is wrong: a verdict passing on another file than the recorded one, or a swept key
nobody counted, or a gate green over a swept key — red.
"""
import json

from tests.unit.tools.container_judge_rig import NBX_SHA, J, Rig  # noqa: E402


def _arms(rig):
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new") == 0


def test_a_staged_file_with_another_sha_is_refused_by_name(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    rig.nbx.write_bytes(b"a staged contaiNer")          # same size, other bytes
    assert rig.main("--arm", "new", "--gate") == 1
    v = rig.verdict()
    assert v["pass"] is False
    assert "sha256" in v["verdict"] and str(rig.nbx) in v["verdict"] and NBX_SHA in v["verdict"]
    assert rig.state()["steps"]["gate"]["ok"] is False
    assert str(rig.nbx) in capsys.readouterr().err


def test_an_absent_staged_file_and_another_byte_count_are_refused_by_name(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    rig.nbx.unlink()
    assert rig.main("--arm", "new", "--gate") == 1
    v = rig.verdict()
    assert v["pass"] is False and "absent" in v["verdict"] and str(rig.nbx) in v["verdict"]
    rig.nbx.write_bytes(b"a staged container")
    build = json.loads((rig.builder / "build.json").read_text())
    rig.write("build.json", {**build, "bytes": 999})
    assert rig.main("--arm", "new", "--gate") == 1
    v = rig.verdict()
    assert v["pass"] is False and "999" in v["verdict"] and "bytes" in v["verdict"]


def test_a_green_verdict_is_overturned_when_the_staged_file_changes(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    assert rig.main("--arm", "new", "--gate") == 0 and rig.verdict()["pass"] is True
    rig.nbx.write_bytes(b"another container")
    assert rig.main("--arm", "new", "--gate") == 1
    assert rig.verdict()["pass"] is False


def test_the_digest_is_streamed_and_exact(tmp_path):
    f = tmp_path / "big.nbx"
    f.write_bytes(b"x" * (3 * 1000 + 7))
    import hashlib
    assert J.sha_streamed(f, chunk=1000) == (hashlib.sha256(f.read_bytes()).hexdigest(), 3007)


HELD = "(64, 64, 64, 'fp16')"
NAMES = ["matmul_kernel::(128, 64, 32, 'fp16')", "softmax_forward_kernel::(4096, 'fp32')"]


def _replay(rig, swept: bool):
    """The frozen directory certifies HELD; the replay cache holds HELD, plus the two NAMES when `swept`."""
    freeze = rig.state()["autotune_freeze"]
    cert = J.Path(freeze["directory"]) / "apple" / "apple_m4_pro" / "matmul_kernel.fp16.json"
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(json.dumps({"entries": {HELD: {"kwargs": {}}}}))
    replay = J.Path(freeze["replay"])
    replay.mkdir(parents=True, exist_ok=True)
    stored = {f"nbxtensor.ops.matmul.matmul_kernel::{HELD}": {"kwargs": {}}}
    if swept:
        stored.update({"nbxtensor.ops.matmul.matmul_kernel::(128, 64, 32, 'fp16')": {"kwargs": {}},
                       "nbxtensor.ops.softmax.softmax_forward_kernel::(4096, 'fp32')": {"kwargs": {}}})
    (replay / "autotune_configs_apple-42e-e3.json").write_text(json.dumps(stored))
    return freeze


def test_a_missing_kernel_key_is_counted_and_named(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    freeze = _replay(rig, swept=True)
    capsys.readouterr()
    assert rig.main("--arm", "new", "--gate") == 1     # the ruling: a key swept on the gate's request is red
    for at in (rig.verdict()["autotune"], rig.state()["steps"]["gate"]["autotune"]):
        assert at["replayed"] == 1 and at["missing_count"] == 2
        # written outside any arm's run here: named so, never guessed (the arm attribution: its own test file)
        assert at["missing"] == [{"key": n, "arm": J.NO_ARM} for n in NAMES]
        assert at["snapshot"] == freeze["snapshot"]
    err = capsys.readouterr().err
    assert "2 missing" in err and all(n in err for n in NAMES)


def test_a_key_swept_on_the_gates_request_turns_the_gate_red_naming_the_keys_and_the_remedy(tmp_path, monkeypatch,
                                                                                              capsys):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    _replay(rig, swept=True)
    capsys.readouterr()
    assert rig.main("--arm", "new", "--gate") != 0
    v, gate = rig.verdict(), rig.state()["steps"]["gate"]
    assert v["pass"] is False and v["verdict"].startswith("FAIL") and gate["ok"] is False
    for text in (v["verdict"], gate["verdict"], capsys.readouterr().err):
        assert all(n in text for n in NAMES)
        assert "conventional size" in text and "census and certification of those keys" in text


def test_zero_missing_keys_leave_the_gate_green(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    _replay(rig, swept=False)
    assert rig.main("--arm", "new", "--gate") == 0
    v = rig.verdict()
    assert v["pass"] is True and v["verdict"] == "PASS"
    assert v["autotune"]["missing_count"] == 0 and v["autotune"]["replayed"] == 1
