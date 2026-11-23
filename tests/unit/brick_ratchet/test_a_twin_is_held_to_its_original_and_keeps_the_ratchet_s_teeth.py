"""The brick ratchet reads the engine's two branches (RED_LINES.md line 19: the same mathematics once per branch is by
design, not a duplicate): a file `<root>/triton/<brick>/<rest>` is the TWIN of `<root>/pytorch/<brick>/<rest>`, by path
alone, and is held to its original's entries until the baseline records its own.

What this test does if the tool is wrong:
1. a twin as long as its listed original is refused as "new and over the limit" (doubling a brick would be red);
2. a twin that grew past its original's entry passes (the twin would have no teeth);
3. a new long function inside triton/ that is NOT a twin (no namesake file on the PyTorch side) passes, or a new long
   function added to a twin file passes;
4. the 95th percentile counts a twin function twice (doubling a brick would move the limit);
5. a private member taken through the branch door (`_branch_door.names("prism", "solver", "_hidden")`) is not counted
   as a private import (the door would hide what an import shows);
6. a twin's `x._name` reaches are added to its original's (the count would double with the brick).

Run: python -m pytest tests/unit/brick_ratchet/test_a_twin_is_held_to_its_original_and_keeps_the_ratchet_s_teeth.py
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("brick_ratchet", REPO / "tools" / "brick_ratchet.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

CONFIG = {"packages": {"pkg": {"root": "pkg", "top": "pkg", "max_code_lines": 5, "max_depth": 2}},
          "max_module_code_lines": 300, "direction": []}


def _long(name, lines, reach=False):
    body = "".join(f"    x += {i}\n" for i in range(lines - 2))
    return f"def {name}(x):\n" + body + ("    y = x._hidden\n" if reach else "") + "    return x\n"


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


BASE = {"pkg/__init__.py": "", "pkg/pytorch/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 8, reach=True)}


def _baseline(tmp_path):
    _write(tmp_path, BASE)
    return B.offenders(B.measure(tmp_path, CONFIG), CONFIG)


def _check(tmp_path, baseline, files):
    _write(tmp_path, files)
    return B.check(B.offenders(B.measure(tmp_path, CONFIG), CONFIG), baseline)


def test_a_twin_as_long_as_its_original_passes_and_one_line_more_is_refused(tmp_path):
    baseline = _baseline(tmp_path)
    twin = {"pkg/triton/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 8, reach=True)}
    assert _check(tmp_path, baseline, twin) == []
    out = _check(tmp_path, baseline, {"pkg/triton/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 9, reach=True)})
    assert any("pkg/triton/prism/solver.py::plan grew past its original's entry" in r for r in out), out


def test_a_long_function_in_triton_that_is_no_twin_is_refused(tmp_path):
    baseline = _baseline(tmp_path)
    out = _check(tmp_path, baseline, {"pkg/triton/prism/other.py": '"""Other."""\n\n\n' + _long("plan", 8)})
    assert any("pkg/triton/prism/other.py::plan is new and over the limit" in r for r in out), out
    out = _check(tmp_path, baseline, {"pkg/triton/prism/other.py": '"""Other."""\n',
                                      "pkg/triton/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 8, True)
                                      + "\n\n" + _long("extra", 8)})
    assert any("pkg/triton/prism/solver.py::extra is new and over the limit" in r for r in out), out


def test_the_percentile_counts_a_twin_function_once(tmp_path):
    _write(tmp_path, BASE | {"pkg/triton/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 8, reach=True)})
    st = B.measure(tmp_path, CONFIG)["pkg"]
    assert st["twins"] == {"pkg/triton/prism/solver.py": "pkg/pytorch/prism/solver.py"}
    assert list(B.distinct_functions(st)) == ["pkg/pytorch/prism/solver.py::plan"]
    assert len(st["functions"]) == 2


def test_a_twin_s_reaches_are_its_original_s(tmp_path):
    _write(tmp_path, BASE | {"pkg/triton/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 8, reach=True)})
    assert B.measure(tmp_path, CONFIG)["pkg"]["reach"] == 1
    _write(tmp_path, {"pkg/triton/prism/solver.py": '"""Plans."""\n\n\n' + _long("plan", 8, reach=True)
                      + "\n\ndef more(x):\n    return x._other\n"})
    assert B.measure(tmp_path, CONFIG)["pkg"]["reach"] == 2


def test_a_private_member_taken_through_the_door_is_a_private_import(tmp_path):
    baseline = _baseline(tmp_path)
    door = ('"""Runs."""\nfrom pkg import branch_door as _branch_door\n\n\n'
            'def run():\n    f = _branch_door.names("prism", "solver", "_hidden")[0]\n    return f\n')
    out = _check(tmp_path, baseline, {"pkg/cli.py": door})
    assert any("pkg/cli.py <- neurobrix.pytorch.prism.solver._hidden" in r for r in out), out
