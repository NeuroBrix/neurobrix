"""The brick ratchet (`tools/brick_ratchet.py`, the owner's rule of 2026-11-10: code in bricks, guard 6) refuses, by
name, a lot that makes the code more tangled, and holds the repository to its committed baseline.

What this file does if the tool is wrong:
1. counting: a docstring, a comment or a blank line counted as code, an `elif` chain or a `with` counted as depth, a
   kernel's body held to the length limit — red;
2. on a small package written in a temporary directory, each of these must be refused by name and is injected one at
   a time: a new function over the length limit, one over the depth limit, a listed offender that grew, a private
   name taken from another module, an import against the declared direction, more `x._name` reaches, a new long
   module, a module without a docstring, one more tensor mechanism in a file of the engine (the owner's rule of
   2026-11-12: tensor mechanics live in nbxtensor) and a first one in a clean file; and a listed offender that no
   longer offends must ask for `--update` — a tool that lets any of them through is red;
   the tensor-mechanism count itself: one hit per pattern kind, none for a comment, a string, or a call of torch (the
   PyTorch engine's own library) — red otherwise;
3. the repository itself: anything worse than `tools/brick_ratchet.baseline.json`, or a baseline entry already
   resolved, is red — this is the ratchet a lot meets before it lands.

No card. Run: python -m pytest tests/unit/brick_ratchet
"""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("brick_ratchet", REPO / "tools" / "brick_ratchet.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

CONFIG = {"packages": {"pkg": {"root": "pkg", "top": "pkg", "max_code_lines": 5, "max_depth": 2}},
          "max_module_code_lines": 30,
          "direction": [{"in": "pkg/core/", "never": ["pkg.door", "nbxtensor.backend.launcher"]}],
          "tensor_logic": json.loads((REPO / "tools" / "brick_ratchet.json").read_text())["tensor_logic"]
          | {"packages": ["pkg"]}}

CLEAN = {
    "pkg/__init__.py": "",
    "pkg/door.py": '"""The door."""\n\ndef open_():\n    return 1\n',
    "pkg/core/__init__.py": "",
    "pkg/core/a.py": '"""Does a."""\n_HIDDEN = 1\n\n\ndef small(x):\n    """Says x."""\n    # a reason\n\n    return x\n',
    "pkg/core/b.py": ('"""Does b."""\n\n\ndef old(x):\n' + "".join(f"    x += {i}\n" for i in range(8)) + "    return x\n"),
    "pkg/flow.py": ('"""Moves a block."""\nfrom nbxtensor.memory.allocator import DeviceAllocator as DA\n\n\n'
                    'def move(d, s, n):\n    DA.memcpy(d, s, n, 1)\n'),
}


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _refusals(tmp_path, change):
    """The check's refusals after `change` {path: text} is applied to the clean package (whose state is the baseline)."""
    _write(tmp_path, CLEAN)
    baseline = B.offenders(B.measure(tmp_path, CONFIG), CONFIG)
    _write(tmp_path, change)
    return B.check(B.offenders(B.measure(tmp_path, CONFIG), CONFIG), baseline)


def test_code_lines_are_neither_docstrings_comments_nor_blanks():
    src = '"""Module."""\n\n# comment\nX = 1  # trailing\n\n\ndef f():\n    """Doc\n    more."""\n    # why\n    return (\n        X\n    )\n'
    assert B.code_lines(src) == {4, 7, 11, 12, 13}


def test_an_elif_chain_and_a_with_add_no_depth():
    import ast
    fn = ast.parse("def f(a):\n    if a == 1:\n        pass\n    elif a == 2:\n        pass\n    elif a == 3:\n"
                   "        with open(a):\n            for i in a:\n                pass\n    else:\n        pass\n").body[0]
    assert B.depth(fn) == 2
    nested = ast.parse("def f(a):\n    if a:\n        pass\n    else:\n        if a:\n            pass\n        x = 1\n").body[0]
    assert B.depth(nested) == 2


def test_the_clean_package_and_its_listed_offender_pass(tmp_path):
    _write(tmp_path, CLEAN)
    found = B.offenders(B.measure(tmp_path, CONFIG), CONFIG)
    assert list(found["pkg"]["functions"]) == ["pkg/core/b.py::old"] and found["pkg"]["functions"]["pkg/core/b.py::old"] == [10, 0]
    assert B.check(found, found) == []


def test_a_new_long_function_is_refused_by_name(tmp_path):
    long = '"""Does a."""\n_HIDDEN = 1\n\n\ndef fresh(x):\n' + "".join(f"    x += {i}\n" for i in range(6)) + "    return x\n"
    out = _refusals(tmp_path, {"pkg/core/a.py": long})
    assert len(out) == 1 and "pkg/core/a.py::fresh is new and over the limit" in out[0], out


def test_a_new_deep_function_is_refused(tmp_path):
    deep = '"""Does a."""\n_HIDDEN = 1\n\n\ndef deep(x):\n    for i in x:\n        if i:\n            while i:\n                i -= 1\n'
    out = _refusals(tmp_path, {"pkg/core/a.py": deep})
    assert len(out) == 1 and "pkg/core/a.py::deep is new and over the limit ([5, 3]" in out[0], out


def test_a_kernel_body_is_exempt_from_length_not_from_depth(tmp_path):
    body = "".join(f"    x += {i}\n" for i in range(9))
    kernel = '"""Does a."""\n_HIDDEN = 1\nimport triton\n\n\n@triton.jit\ndef k(x):\n' + body + "    return x\n"
    assert [r for r in _refusals(tmp_path, {"pkg/core/a.py": kernel}) if "tensor logic" not in r] == []
    deep = kernel + "\n\n@triton.jit\ndef k2(x):\n    for i in x:\n        if i:\n            while i:\n                i -= 1\n"
    assert any("::k2 is new" in r for r in _refusals(tmp_path, {"pkg/core/a.py": deep}))


def test_a_listed_offender_that_grows_is_refused_and_one_that_shrinks_asks_for_update(tmp_path):
    grown = CLEAN["pkg/core/b.py"].replace("    return x\n", "    x += 99\n    return x\n")
    out = _refusals(tmp_path, {"pkg/core/b.py": grown})
    assert len(out) == 1 and "pkg/core/b.py::old grew ([10, 0] -> [11, 0]" in out[0], out
    shrunk = '"""Does b."""\n\n\ndef old(x):\n    return x\n'
    out = _refusals(tmp_path, {"pkg/core/b.py": shrunk})
    assert out and all(r.startswith("resolved, run --update") for r in out) and "pkg/core/b.py::old" in out[0], out


def test_a_private_name_taken_from_another_module_is_refused(tmp_path):
    out = _refusals(tmp_path, {"pkg/core/b.py": CLEAN["pkg/core/b.py"] + "\nfrom .a import _HIDDEN\n"})
    assert out == ["pkg private: pkg/core/b.py <- pkg.core.a._HIDDEN — a private name taken from another module"], out


def test_an_import_against_the_direction_is_refused_relative_or_absolute(tmp_path):
    for line in ("from pkg.door import open_", "from ..door import open_", "import pkg.door"):
        out = _refusals(tmp_path, {"pkg/core/b.py": CLEAN["pkg/core/b.py"] + "\n" + line + "\n"})
        assert out == ["pkg direction: pkg/core/b.py -> pkg.door — an import against the declared direction"], (line, out)
    assert _refusals(tmp_path, {"pkg/door.py": CLEAN["pkg/door.py"] + "\nfrom pkg.core.a import small\n"}) == []


def test_reaching_into_another_objects_private_state_is_counted(tmp_path):
    out = _refusals(tmp_path, {"pkg/door.py": CLEAN["pkg/door.py"] + "\n\ndef peek(o):\n    return o._inside\n"})
    assert len(out) == 1 and out[0].startswith("pkg reach: 1 uses of x._name"), out
    own = CLEAN["pkg/door.py"] + "\n\nclass C:\n    def get(self):\n        return self._inside\n"
    assert _refusals(tmp_path, {"pkg/door.py": own}) == []


def test_a_new_long_module_and_a_mute_module_are_refused(tmp_path):
    long = '"""Long."""\n' + "".join(f"V{i} = {i}\n" for i in range(31))
    out = _refusals(tmp_path, {"pkg/core/c.py": long})
    assert len(out) == 1 and "pkg/core/c.py is new and over the limit (31 code lines)" in out[0], out
    out = _refusals(tmp_path, {"pkg/core/d.py": "X = 1\n"})
    assert out == ["pkg mute: pkg/core/d.py — a module that states no responsibility (no docstring)"], out


def _hits(source):
    """The tensor-mechanism hits of one module, by kind."""
    import ast
    return [kind for _, kind, _ in B.tensor_hits(ast.parse(source), CONFIG["tensor_logic"])]


def test_each_tensor_mechanism_is_one_hit_and_comments_strings_and_torch_are_none():
    one = {
        "the library's allocator, arenas and pools": "from nbxtensor.memory.allocator import DeviceAllocator\nDeviceAllocator.set_device(0)\n",
        "ctypes pointers and buffers": "import ctypes as _ct\nb = (_ct.c_char * 8)()\n",
        "kernel definitions": "import triton\n\n\n@triton.jit\ndef k(x):\n    pass\n",
        "raw copies and pointers": "def f(t, u):\n    t.memcpy(u)\n",
        "array buffer addresses": "def f(a):\n    return a.ctypes.data\n",
        "driver calls": "def f(rt, d, s, n):\n    rt.cudaMemcpy(d, s, n, 1)\n",
        "kernel launches": "def f(x, grid):\n    add_kernel[grid](x)\n    other[(1,)](x)\n",
        "arena calls": "def f(self):\n    self._arena.alloc(4)\n",
    }
    for kind, source in one.items():
        assert set(_hits(source)) == {kind}, (kind, _hits(source))
    assert _hits(one["kernel launches"]) == ["kernel launches"] * 2
    quiet = ('"""DeviceAllocator.memcpy(d, s, n) in a docstring."""\n# t.data_ptr() and cudaMemcpy in a comment\n'
             'X = "DeviceAllocator.memcpy(d, s, n)"\nimport torch\n\n\ndef f(o, ops, op, block):\n'
             '    torch.cuda.cudart().cudaHostUnregister(block.data_ptr())\n    t = torch.empty(3)\n    st = o.untyped_storage()\n    torch.cuda.cudart().cudaHostRegister(t.data_ptr(), 4, 0)\n'
             '    return st.data_ptr(), o.untyped_storage().data_ptr(), ops[op](o)\n')
    assert _hits(quiet) == [], _hits(quiet)


def test_one_more_tensor_mechanism_in_the_engine_is_refused_and_said_to_belong_in_nbxtensor(tmp_path):
    more = CLEAN["pkg/flow.py"] + "    DA.memcpy(s, d, n, 2)\n"
    out = _refusals(tmp_path, {"pkg/flow.py": more})
    assert len(out) == 1 and "pkg/flow.py holds 2" in out[0] and "baseline 1" in out[0] and "nbxtensor" in out[0], out
    print("\n".join(out))
    fresh = CLEAN["pkg/door.py"] + "\n\ndef shut(d, s):\n    from nbxtensor.memory.allocator import DeviceAllocator\n" \
                                   "    DeviceAllocator.memcpy(d, s, 4, 1)\n"
    out = _refusals(tmp_path, {"pkg/door.py": fresh})
    assert len(out) == 1 and "pkg/door.py holds 1" in out[0] and "had none" in out[0] and "nbxtensor" in out[0], out
    out = _refusals(tmp_path, {"pkg/flow.py": '"""Moves a block."""\n'})
    assert out and all(r.startswith("resolved, run --update") for r in out) and "pkg/flow.py" in out[0], out


def test_an_import_of_a_name_out_of_a_forbidden_package_is_against_direction(tmp_path):
    out = _refusals(tmp_path, {"pkg/core/b.py": CLEAN["pkg/core/b.py"] + "\nfrom nbxtensor.backend import launcher\n"})
    assert out == ["pkg direction: pkg/core/b.py -> nbxtensor.backend.launcher — an import against the declared "
                   "direction"], out


def test_the_repository_holds_its_baseline():
    config = json.loads(B.CONFIG.read_text())
    refusals = B.check(B.offenders(B.measure(REPO, config), config), json.loads(B.BASELINE.read_text()))
    assert not refusals, "\n".join(refusals[:40])


def test_the_limits_are_not_above_the_percentile_they_were_drawn_from():
    """The limits are data drawn from the code's own 95th percentile: lowered when it falls, never raised above it."""
    config = json.loads(B.CONFIG.read_text())
    for name, st in B.measure(REPO, config).items():
        lengths = [n for n, _, kernel in st["functions"].values() if not kernel]
        depths = [d for _, d, _ in st["functions"].values()]
        lim = config["packages"][name]
        assert lim["max_code_lines"] <= B._percentile(lengths, .95) and lim["max_depth"] <= B._percentile(depths, .95), name
