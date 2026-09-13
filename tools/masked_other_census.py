#!/usr/bin/env python3
"""How many shapes a model reaches carry a masked load whose `other` is a
per-element array.

The question is not academic and it is not upstream's. Upstream lowers
`tt.load(ptr, mask, other)` with a per-element `other` correctly on the
ordinary MEPT path: it indexes `other[i]` alongside `ptr[i]`. The cooperative
staged fill is the one place that cannot, because it visits elements a thread
did not load and so has no `i` to index with — and the repair for the missing
mask there therefore carries a REFUSAL for that case. That refusal is OURS,
not upstream's, and it can only take away shapes that work today. So it gets
counted before it lands, not argued about.

Two numbers, and the pair is the answer:

  --scan     every `tt.load` in the compiled IR whose `other` is per-element,
             anywhere in the kernel. The POPULATION: shapes that would meet
             the refusal if they also reached the staged fill.

  --census   the same, per catalogue model, over what the model actually
             compiles. A pattern in a kernel nothing executes and a pattern on
             the decode path are not the same finding.

Neither number is the refusal's true reach on its own: a load is only refused
if it ALSO feeds a cooperative staged fill, which the lowerer decides and the
IR does not say. This tool gives the upper bound; the lowerer's own
`NBX_FILL_OTHER_CENSUS` marker gives the exact count. An upper bound of zero
settles the question without needing the marker, which is why it is worth
having.

    tools/masked_other_census.py --census <model> [--arm triton]
    tools/masked_other_census.py --selfproof
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ir_census import compile_census, refuse_if_empty        # noqa: E402

#: `%r = tt.load %a, %b, %c : tensor<...>` — the operand list is everything
#: between the op name and the type colon. Attributes (evictionPolicy, cache)
#: may follow the operands; they start with a letter, not a `%`.
_LOAD = re.compile(r"^\s*(%[\w:$.]+)\s*=\s*tt\.load\s+([^:]*?)\s*(?::|$)")
_DEF = re.compile(r"^\s*(%[\w:$.]+)\s*=\s*([\w.]+)\s*(.*)$")

#: Ops that cannot turn a uniform value into a varying one: a cast or a
#: reshape of a splat is still a splat. Chasing through them is what stops
#: `arith.sitofp %splat` being reported as per-element.
_ELEMENTWISE_UNARY = {
    "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui",
    "arith.extf", "arith.truncf", "arith.extsi", "arith.extui",
    "arith.trunci", "arith.bitcast", "tt.bitcast", "tt.reshape",
    "tt.broadcast", "tt.expand_dims", "arith.negf",
}
#: A `dense<...>` with no comma and no nested bracket is ONE value repeated:
#: `dense<3.0>` is a splat, `dense<[1.0, 2.0]>` is not.
_DENSE_SPLAT = re.compile(r"dense<([^<>\[\],]*)>")


def _operands(blob: str) -> list[str]:
    return [t for t in (x.strip() for x in blob.split(",")) if t.startswith("%")]


def _defs(ttir: str) -> dict[str, tuple[str, str]]:
    out = {}
    for line in ttir.splitlines():
        m = _DEF.match(line)
        if m and not m.group(2).startswith("tt.load"):
            out[m.group(1)] = (m.group(2), m.group(3))
        elif m:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def _is_uniform(ssa: str, defs: dict, depth: int = 0) -> bool:
    """Is `ssa` the same value in every element? Chased through casts.

    Unknown means NOT uniform. A detector that answers "uniform" when it
    cannot tell under-reports, and under-reporting is what makes a census
    say a refusal is free when it is not.
    """
    if depth > 8 or ssa not in defs:
        return False
    op, rest = defs[ssa]
    if op == "tt.splat":
        return True
    if op == "arith.constant":
        return bool(_DENSE_SPLAT.search(rest)) or "dense" not in rest
    if op in _ELEMENTWISE_UNARY:
        srcs = _operands(rest.split(":")[0])
        return bool(srcs) and all(_is_uniform(s, defs, depth + 1) for s in srcs)
    return False


def per_element_other(ttir: str) -> list[dict]:
    """Every masked `tt.load` whose `other` varies across elements."""
    defs = _defs(ttir)
    found = []
    for line in ttir.splitlines():
        m = _LOAD.match(line)
        if not m:
            continue
        ops = _operands(m.group(2))
        if len(ops) < 3:
            continue                      # no mask, or mask with default other
        other = ops[2]
        if _is_uniform(other, defs):
            continue
        origin = defs.get(other, ("<unknown>", ""))[0]
        entry = {"other": other, "produced_by": origin}
        if entry not in found:
            found.append(entry)
    return found


_SAMPLE = """
%b = arith.constant dense<3.000000e+00> : tensor<64xf32>
%m = tt.splat %n : i32 -> tensor<64xi32>
%arr = arith.sitofp %offs : tensor<64xi32> to tensor<64xf32>
%u = arith.sitofp %m : tensor<64xi32> to tensor<64xf32>
%a_2 = tt.load %a_1, %m_0, %arr : tensor<64x!tt.ptr<f32>>
%b_3 = tt.load %a_1, %m_0, %b : tensor<64x!tt.ptr<f32>>
%u_4 = tt.load %a_1, %m_0, %u : tensor<64x!tt.ptr<f32>>
%c = tt.load %a_1 : tensor<64x!tt.ptr<f32>>
"""


def selfproof() -> int:
    """The detector must separate four loads that a regex on the word `other`
    cannot. This text is not invented: it is the TTIR triton emitted for a
    kernel written to carry all four forms, with the uniform-through-a-cast
    row added — the one a detector that stops at the defining op gets wrong.
    """
    found = per_element_other(_SAMPLE)
    names = sorted(f["other"] for f in found)
    ok = names == ["%arr"]
    print(f"per-element `other` found : {names}")
    print(f"expected                  : ['%arr']")
    for label, cond in [
        ("a genuinely varying `other` is SEEN", "%arr" in names),
        ("a `dense<3.0>` splat is NOT counted", "%b" not in names),
        ("a splat behind a cast is NOT counted", "%u" not in names),
        ("an unmasked load is NOT counted", len(names) <= 1),
    ]:
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")
        ok = ok and cond
    print("SELFPROOF", "PASS" if ok else "FAIL")
    return 0 if ok else 1


#: The scan root is the whole package, not `kernels/`. A first version rooted
#: at `src/neurobrix/kernels` reported 255 of 255 scalar and was WRONG about
#: its own coverage: `src/neurobrix/triton/flow/rnnt.py` carries @triton.jit
#: kernels the stt path reaches and sits outside that directory. A census
#: whose population is smaller than it claims is the vacuous guard with a
#: number attached.
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "neurobrix"
#: A vendored reference tree. 894 files carry @triton.jit and NOTHING in the
#: engine imports it -- verified, not assumed. Counting it would inflate the
#: population with kernels no model can reach.
_VENDORED = "triton_kernels_ref"


def _other_arg(call: ast.Call):
    for kw in call.keywords:
        if kw.arg == "other":
            return kw.value
    return None


#: Calls that return one value, never a tensor. `tl.full` and `tl.zeros`
#: produce a tensor and are deliberately absent.
_SCALAR_CALLS = {"float", "int", "_get_finfo_val", "tl.math.inf"}


def _resolve(name: str, tree) -> ast.AST | None:
    """The LAST assignment to `name` anywhere in the module, or None.

    Deliberately not scope-aware: a kernel that assigns `min_val` twice with
    two different kinds of value would come back as whichever is last, and the
    verdict below then says `unresolved` rather than claiming a scalar. Under-
    claiming is the safe direction here.
    """
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    found = node.value
    return found


def _classify_other(node, tree, depth: int = 0):
    """`scalar`, `per_element`, or `unresolved` — following the VALUE.

    The first version of this scan printed `other=min_value` and stopped,
    leaving a reader to decide whether that name held a float or a tensor.
    That is the detector that reads where a value is NAMED rather than where
    it is WRITTEN — the class the vacuous-guard register carries as its fourth
    form, committed here in the very tool built to measure a refusal. It is
    resolved, not listed.
    """
    if depth > 6:
        return "unresolved", "chase too deep"
    if isinstance(node, ast.Constant):
        return "scalar", ast.unparse(node)
    if isinstance(node, ast.UnaryOp):
        return _classify_other(node.operand, tree, depth + 1)
    if isinstance(node, ast.Call):
        fn = ast.unparse(node.func)
        if fn.split(".")[-1] == "constexpr" and len(node.args) == 1:
            # `tl.constexpr(X)` is X, boxed. Chase X rather than trust the
            # wrapper: `tl.constexpr` does not itself make a value scalar.
            verdict, written = _classify_other(node.args[0], tree, depth + 1)
            return verdict, f"tl.constexpr({written})"
        if fn.split(".")[-1] in {c.split(".")[-1] for c in _SCALAR_CALLS}:
            return "scalar", f"{fn}(...)"
        return "unresolved", f"call {fn}(...)"
    if isinstance(node, ast.IfExp):
        a, wa = _classify_other(node.body, tree, depth + 1)
        b, wb = _classify_other(node.orelse, tree, depth + 1)
        if a == b == "scalar":
            return "scalar", f"{wa} / {wb}"
        return ("per_element" if "per_element" in (a, b) else "unresolved",
                f"{wa} / {wb}")
    if isinstance(node, ast.Subscript):
        return "per_element", ast.unparse(node)[:50]   # `x[None, :]` and kin
    if isinstance(node, ast.Name):
        src = _resolve(node.id, tree)
        if src is None:
            # A kernel PARAMETER: a runtime argument, which Triton passes as a
            # scalar and lowers to `tt.splat`. It cannot be a tensor -- a
            # @triton.jit signature takes pointers and scalars, not arrays.
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef):
                    names = [a.arg for a in
                             list(fn.args.args) + list(fn.args.kwonlyargs)]
                    if node.id in names:
                        return "scalar", f"parameter of {fn.name}()"
            return "unresolved", f"`{node.id}` never assigned in this file"
        verdict, written = _classify_other(src, tree, depth + 1)
        return verdict, f"{node.id} = {written}"
    return "unresolved", ast.unparse(node)[:50]


def scan(verbose: bool = False) -> int:
    """Every `tl.load(..., other=X)` in our tree where X is not a literal.

    An UPPER BOUND over the population, and nothing better. It cannot tell a
    per-element array from a scalar the kernel computed once — `other=acc_zero`
    where `acc_zero = 0.0` reads exactly like `other=offs.to(tl.float32)` in
    the source, and only the compiled IR separates them. Its use is to say how
    LARGE the question is before the census answers it: if the tree holds no
    non-literal `other` at all, the refusal cannot reach anything and the
    census is a formality.
    """
    total_loads = 0
    files_read = 0
    jit_files = 0
    hits: dict[str, list] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if _VENDORED in path.parts:
            continue
        try:
            src_text = path.read_text()
            tree = ast.parse(src_text)
        except SyntaxError:
            continue
        files_read += 1
        if "@triton.jit" in src_text:
            jit_files += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not ast.unparse(node.func).endswith("load"):
                continue
            other = _other_arg(node)
            if other is None:
                continue
            total_loads += 1
            verdict, written = _classify_other(other, tree)
            if verdict == "scalar":
                continue
            rel = path.relative_to(PACKAGE.parents[1])
            hits.setdefault(str(rel), []).append(
                (other.lineno, ast.unparse(other)[:60], verdict, written))
    flat = [(f, *row) for f, rows in hits.items() for row in rows]
    per_el = [r for r in flat if r[3] == "per_element"]
    unres = [r for r in flat if r[3] == "unresolved"]
    print(f"files read under {PACKAGE.name}/ (minus the vendored tree) : {files_read}")
    print(f"  of which carry @triton.jit               : {jit_files}")
    print(f"`tl.load(..., other=)` sites in the tree : {total_loads}")
    print(f"resolved to a SCALAR                     : {total_loads - len(flat)}")
    print(f"resolved to a PER-ELEMENT array          : {len(per_el)}")
    print(f"NOT resolved by this scan                : {len(unres)}")
    print()
    if per_el or unres or verbose:
        for f, lineno, src, verdict, written in sorted(flat):
            print(f"  [{verdict}] {f}:{lineno}  other={src}")
            print(f"          written: {written}")
        print()
    print("The scan follows each `other` to where its value is WRITTEN, not to")
    print("where it is named: `other=min_value` says nothing until `min_value`")
    print("is resolved. Anything it cannot chase is reported as unresolved, and")
    print("never as a scalar -- under-claiming is the safe direction, because")
    print("a missed per-element `other` becomes a landed refusal breaking a")
    print("model with nothing in the count having warned.")
    print()
    print("Still an upper bound on the REFUSAL: a per-element `other` is only")
    print("refused if it also feeds a cooperative staged fill, which only the")
    print("lowerer decides. Zero here settles it without needing the marker.")
    return 0


def census(model: str, arm: str, extra) -> int:
    seen = compile_census(model, arm, per_element_other, extra)
    rc = refuse_if_empty(seen, model, arm,
                         "the per-element `other` of a masked load")
    if rc is not None:
        return rc
    carriers = {k: v for k, v in seen.items() if v["findings"]}
    refused = {k: v for k, v in seen.items() if v.get("refused")}
    print()
    # `len(seen)` counts distinct NAMES, and a name is not a kernel: whisper
    # compiles dozens of kernels all called `kernel`, which collapsed into one
    # row and made the denominator read 2. The compilations are the population
    # actually inspected, so they are what is printed first.
    compilations = sum(v["compilations"] for v in seen.values())
    print(f"compilations INSPECTED for {model} ({arm}) : {compilations}")
    print(f"  distinct kernel names                   : {len(seen)}")
    print(f"  names that also REFUSED at least once   : {len(refused)}")
    print(f"with a masked load whose `other`     : {len(carriers)}")
    print(f"  is a per-element array")
    for name in sorted(refused):
        print(f"  [refused, IR not read] {name}")
    for name, row in sorted(carriers.items()):
        print(f"  {name}  ({row['compilations']} compilation(s))")
        for f in row["findings"]:
            print(f"      other {f['other']} produced by {f['produced_by']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", metavar="MODEL")
    ap.add_argument("--arm", default="triton")
    ap.add_argument("--selfproof", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("rest", nargs="*", help="arguments passed to `neurobrix run`")
    args = ap.parse_args()
    if args.selfproof:
        return selfproof()
    if args.scan:
        return scan(args.verbose)
    if args.census:
        return census(args.census, args.arm, args.rest or None)
    ap.error("give --scan, --census MODEL or --selfproof")


if __name__ == "__main__":
    sys.exit(main())
