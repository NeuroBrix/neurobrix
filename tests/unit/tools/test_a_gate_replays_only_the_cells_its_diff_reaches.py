"""The computed gate scope (tools/gate_scope.py) selects the cells a diff can reach, and only those.

The contract:

  * a statement-level diff of two normal forms names the inserted, deleted and replaced statements, inside
    the compound statements whose headers are equal;
  * the guard of an inserted early exit holds exactly where its path and its own test hold (a binding the
    path cannot give raises: a hit, conservatively);
  * a manual prune without its reason, and a scope with no change named, are refused by name;
  * BACKTESTS against the mq24 gate (skipped where the gate, the shared cache or the commits are absent):
      - the daemon-only range 192a41cf..6ce86b39 selects no model cell;
      - a29261aa (the triton `storage_dtype` change) selects every triton cell of the containers whose graphs
        pass bool as an explicit dtype argument — the oracle is read from the containers' records, never a
        list of names — and no cell of any other container, plus whatever the plan differential measured; every
        other cell is carried with its row. Their native cells may be selected too: the static reach of the
        native process holds `TritonSequence._parse_dtype` through `normalize_for_branch`, whose mode argument
        it cannot fold (named in the report, an over-approximation, never a miss);
      - 68333e77 (integer floor/trunc division) selects every triton cell of the containers holding an integer
        floor_divide/div signature or the fused MoE op (whose dispatch divides integer token ids), and no cell
        of any other container (their native cells: the same named over-approximation, the kernel wrappers
        being referenced at module level from what the native process imports).

What would these do if the code were wrong? Dropping the dtype dimension (the cells no longer carry the dtypes
their graphs hold) leaves a29261aa's differing points with no holder: the selected set loses the bool models
and the a29261aa backtest goes red — proved clean -> injected -> restored (see the commit).

Runnable: PYTHONNOUSERSITE=1 /home/mlops/venvs/nbx_t214/bin/python -m pytest \
    tests/unit/tools/test_a_gate_replays_only_the_cells_its_diff_reaches.py -v
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import time
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import gate_scope as G               # noqa: E402
import gate_scope_code as C          # noqa: E402

MQ24 = Path("/home/mlops/nbx/campaigns/2026_10_10_mq24")


def _fn(src: str):
    return ast.parse(src).body[0]


# -------------------------------------------------------------------------------------------------------------
# the static half
# -------------------------------------------------------------------------------------------------------------

def test_a_statement_diff_descends_equal_headers():
    a = _fn("def f(a, m):\n    if m == 'trunc':\n        x = a\n        return x\n    return a\n")
    b = _fn("def f(a, m):\n    if m == 'trunc':\n        if a > 0:\n            return 0\n        x = a\n"
            "        return x\n    return a\n")
    d = G.stmt_diff(a.body, b.body)
    assert [k for k, *_ in d] == ["ins"]
    kind, sa, sb, path = d[0]
    assert not sa and isinstance(sb[0], ast.If) and path and isinstance(path[0][0], ast.If)


def test_the_guard_of_an_inserted_early_exit_holds_only_under_its_path_and_test():
    b = _fn("def div(a, b, rounding_mode=None):\n    if rounding_mode == 'floor':\n        return 1\n"
            "    if rounding_mode == 'trunc':\n        if a > b:\n            return 2\n    return 3\n")
    a = _fn("def div(a, b, rounding_mode=None):\n    if rounding_mode == 'floor':\n        return 1\n"
            "    if rounding_mode == 'trunc':\n        pass\n    return 3\n")
    d = G.stmt_diff(a.body, b.body)
    kind, sa, sb, _ = d[0]
    assert kind in ("ins", "rep")
    s0 = sb[0]
    cond, pre = G.region_guard(b, {id(s0)}, s0.test)
    ns = {}
    exec(G.guard_source(b, cond, pre, "g"), ns)
    g = ns["g"]
    assert g(5, 1, "trunc") is True
    assert g(1, 5, "trunc") is False            # the inserted test is false: the old behaviour
    assert g(5, 1, "floor") is False            # an earlier exit leaves first
    assert g(5, 1, None) is False


def test_a_region_inside_a_loop_has_no_parameter_guard():
    b = _fn("def f(xs):\n    for x in xs:\n        y = x + 1\n    return 0\n")
    target = b.body[0].body[0]
    assert G.region_guard(b, {id(target)}) is None


def test_a_module_level_call_site_carries_no_parameter_guard():
    """A callee reached from a module-level statement (`X = f(...)` at import) has no caller parameters to
    guard over: the site carries no guard (None — treated as reached), never a crash building one."""
    callee = _fn("def f(a):\n    return a\n")
    caller = ast.parse("X = f(1)\n")
    node = caller.body[0].value

    class P:
        @staticmethod
        def _arg_for(cfn, node, recv, ctor, p):
            return ("arg", node.args[0])

        @staticmethod
        def canon(uid):
            return caller

    class T:
        units = {"m:f": callee}

    got = G.Judge.caller_guard(type("S", (), {})(), P, T, "m:f", ("", "g"), "m:<module>", node, None, False, "call")
    assert got is None


def test_a_prune_without_its_reason_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="reason"):
        G.main(["--commits", "HEAD", "--gate-dir", str(tmp_path), "--out", str(tmp_path / "o"),
                "--prune", "16g:M:triton:"])


def test_no_change_named_is_refused():
    with pytest.raises(SystemExit, match="no change named"):
        G.change_pairs(G.REPO)


def test_free_names_see_through_canonical_references_and_comprehensions():
    f = _fn("def f(x):\n    import numpy as np\n    return [y for y in x] + [np.zeros(1)] + [z]\n")
    assert G.free_names(f) == {"z"}
    f2 = ast.parse("def f(x):\n    return g(x)\n").body[0]
    f2.body[0].value.func = ast.Name(id=C.SIG + "pkg.mod:g", ctx=ast.Load())
    assert G.free_names(f2) == set()


# -------------------------------------------------------------------------------------------------------------
# the backtests (the whole tool, on this rack's gate)
# -------------------------------------------------------------------------------------------------------------

def _have(*shas):
    if not (MQ24 / "gate2").is_dir() or not G.CACHE.is_dir():
        return False
    for s in shas:
        if subprocess.run(["git", "-C", str(G.REPO), "cat-file", "-e", f"{s}^{{commit}}"],
                          capture_output=True).returncode:
            return False
    return True


backtest = pytest.mark.skipif(os.environ.get("NBX_GATE_SCOPE_BACKTESTS") != "1",
                              reason="the backtests run the whole scope (minutes): NBX_GATE_SCOPE_BACKTESTS=1")


@pytest.fixture(scope="module")
def cfg():
    return G.load_cfg()


def _scope(cfg, **kw):
    pairs = G.change_pairs(G.REPO, **kw)
    return G.compute(pairs, MQ24, cfg, log=lambda s: None)


def _retraced(res, cls):
    """The cells of containers retraced since the gate ran: replayed whatever the diff."""
    return {c for c in res["cells"].cells[cls] if c[0] in res["cells"].retraced}


@backtest
def test_a_daemon_only_range_replays_no_model_cell(cfg):
    if not _have("192a41cf", "6ce86b39"):
        pytest.skip("gate or commits absent")
    res = _scope(cfg, rng="192a41cf..6ce86b39")
    assert all(set(s) == _retraced(res, cls) for cls, s in res["sel"].items()), res["report"]["selected"]


@backtest
def test_a29261aa_selects_the_triton_cells_whose_graphs_pass_bool(cfg, tmp_path):
    if not _have("a29261aa"):
        pytest.skip("gate or commits absent")
    res = _scope(cfg, commits="a29261aa")
    cells = res["cells"]
    bool_models = {m for m, r in cells.recs.items() if "bool" in r["explicit_dtypes"]}
    plans = {}
    for p in res["report"]["pairs"]:
        for cls, pl in (p.get("plans") or {}).items():
            plans.setdefault(cls, set()).update(tuple(k.split("/")) for k in pl["which"])
    for cls, s in res["sel"].items():
        rest = set(s) - plans.get(cls, set()) - _retraced(res, cls)
        must = cells.of(cls, bool_models, {"triton"}) - _retraced(res, cls)
        may = cells.of(cls, bool_models, {"triton", "native"})
        assert must <= rest <= may, (cls, sorted(must - rest), sorted(rest - may))
    # every other cell is carried, with the gate and row it comes from
    G.write_outputs(res, tmp_path, cfg)
    carried = [l.split("\t") for l in (tmp_path / "carried.tsv").read_text().splitlines()[1:]]
    assert len(carried) == sum(len(cells.cells[c]) - len(res["sel"][c]) for c in cells.cells)
    assert all(r[3] and r[4] != "-" for r in carried if r[8] != "NO ROW")
    for cls in cells.cells:
        q = (tmp_path / f"queue_{cls}.txt").read_text().split("\n")
        assert sum(len(x.split()[1].split(",")) for x in q if x) == len(res["sel"][cls])


@backtest
def test_68333e77_selects_only_integer_division_holders_in_triton(cfg):
    if not _have("68333e77"):
        pytest.skip("gate or commits absent")
    res = _scope(cfg, commits="68333e77")
    cells = res["cells"]

    def int_div(m):
        r = cells.recs[m]
        if "custom::moe_fused" in r["ops"]:
            return True
        for t in ("aten::floor_divide", "aten::div"):
            for s in r["sigs"].get(t, []):
                s = json.loads(s)
                if t.endswith("div") and not t.endswith("floor_divide") and \
                        (s["kwargs"].get("rounding_mode") or {}).get("s") not in ("floor", "trunc") and \
                        ((s["kwargs"].get("rounding_mode") or {}).get("r") or {}).get("value") not in ("floor", "trunc"):
                    continue
                ints = [a for a in s["args"][:2] if ("t" in a and not a["t"].startswith(("float", "bfloat", "complex"))
                                                     and a["t"] != "bool") or ("s" in a and isinstance(a["s"], int)
                                                                               and not isinstance(a["s"], bool))]
                if len(ints) == 2 and any("t" in a for a in s["args"][:2]):
                    return True
        return False
    want_models = {m for m in cells.models if int_div(m)}
    for cls, s in res["sel"].items():
        s = set(s) - _retraced(res, cls)
        must = cells.of(cls, want_models, {"triton"}) - _retraced(res, cls)
        may = cells.of(cls, want_models, {"triton", "native"})
        assert must <= s <= may, (cls, sorted(must - s), sorted(s - may))


# -------------------------------------------------------------------------------------------------------------
# the probe's rack: the vendor profile is bound behind the door, or the rack is refused
# -------------------------------------------------------------------------------------------------------------

HW = "default-ff6008b7"


def _rack_tree():
    """A tree carrying the class's hardware profile (gitignored, so a checkout may lack it): this one, else a
    materialized gate tree."""
    for t in [ROOT, *sorted(G.SCRATCH.glob("trees/*"))]:
        if (t / "src/neurobrix/config/hardware" / f"{HW}.yml").is_file():
            return t
    pytest.skip(f"no tree at hand carries src/neurobrix/config/hardware/{HW}.yml")


def _probe(tree, rack, jobs):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NBX_", "NEUROBRIX_"))}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(tree / "src"), PYTHONNOUSERSITE="1")
    req = {"src": str(tree / "src"), "hw": HW, "jobs": jobs, "rack": rack}
    p = subprocess.run([G.PYTHON, str(G.TOOLS / "gate_scope_probe.py")], input=json.dumps(req),
                       capture_output=True, text=True, env=env, timeout=900)
    if G.MARK in p.stdout:
        return json.loads(p.stdout.rsplit(G.MARK, 1)[1])
    assert p.stdout.strip(), p.stderr[-800:]
    return json.loads(p.stdout.strip().splitlines()[-1])     # a refusal at the rack: one JSON line, no mark


# a point reads the vendor profile the way a launch key does (matrix_unit() -> active_vendor_profile())
READS_PROFILE = ("def f(x):\n"
                 "    import neurobrix; from nbxtensor.ops._configs import matrix_unit, active_vendor_profile\n"
                 "    return (x, active_vendor_profile().get('_profile'), bool(matrix_unit()))\n")


def test_a_probe_point_reads_the_racks_vendor_profile_and_not_an_empty_one(cfg):
    """Behind the door no driver answers; before 2026-10-12 the rack left the vendor profile to the driver, every
    point re-probed it (37.6 ms each, measured) and was answered under `{}` — the matrix unit unread."""
    tree = _rack_tree()
    rack = dict(cfg["rack"], package=cfg["package"])
    job = {"id": "p", "kind": "points", "module": "nbxtensor.ops.launch_keys", "source": READS_PROFILE,
           "entry": "f", "points": [{"x": {"lit": i}} for i in range(2000)]}
    t0 = time.perf_counter()
    out = _probe(tree, rack, [job])
    wall = time.perf_counter() - t0
    assert "error" not in out, out
    res = out["p"]["results"]
    assert len(res) == 2000 and all(r.endswith(", 'volta', True)") for r in res), res[:3]
    # 2 000 reads: bound, the whole probe ran in 1.8 s (measured 2026-10-12); unbound it is 75 s of driver probes
    assert wall < 30, wall


def test_a_rack_whose_vendor_profile_resolves_empty_is_refused_by_name(cfg):
    tree = _rack_tree()
    rack = dict(cfg["rack"], package=cfg["package"], target_binder="operator:is_")   # a binder that binds nothing
    out = _probe(tree, rack, [{"id": "p", "kind": "points", "module": "nbxtensor.ops.launch_keys",
                               "source": READS_PROFILE, "entry": "f", "points": [{"x": {"lit": 0}}]}])
    assert set(out) == {"error"} and "vendor profile resolved EMPTY" in out["error"], out
