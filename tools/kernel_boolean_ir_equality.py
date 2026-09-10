#!/usr/bin/env python3
"""Cells A and B of the kernel-boolean validation plan: prove the nineteen
`and`/`or` → `&`/`|` edits changed nothing, and prove this instrument can see
the one fault that would have made them change something.

WHY AN IR TEST AND NOT A NUMERICAL ONE

The thesis to prove is "the edit is neutral". A numerical test can only ever
support that thesis weakly — it agrees with the oracle before and after, which
is also what a *pair* of compensating errors looks like. Compiling both forms
and comparing the Triton IR proves it directly, and proves it for the three
sites (`aten::argmin`, `aten::min`, `aten::var`) that no container in the local
cache reaches, where a model run would be green having exercised nothing.

WHAT EACH CELL ASSERTS

  A  before-TTIR == after-TTIR, byte for byte, for every kernel carrying an
     edit. On Triton 3.6.0 this must hold: `visit_BoolOp` lowers `and`/`or` on
     tensors to `logical_and`/`logical_or`, which bitcast each operand to int1
     and apply the bitwise op — and every operand at these sites is already
     int1, so the bitcasts are no-ops and the lowering *is* `&`.

  B  a deliberately UNPARENTHESISED variant produces a DIFFERENT TTIR. `&`
     binds tighter than `<`, so `col_offset < N & row_mask` parses as
     `col_offset < (N & row_mask)`, i.e. `col_offset < (N & 1)` — one column of
     N is read and the norm comes out wrong by roughly sqrt(N) in silence. Cell
     A is worth nothing until it has been seen catching this.

The AST gate (tests/unit/kernels/test_jit_bodies_have_no_python_booleans.py)
catches the deprecated form; it cannot catch cell B's fault, because there is no
boolean operator left to see. The two instruments are complementary, not
redundant.

NO SILENT SKIP. A kernel whose signature does not compile is a FAILURE, not an
omission: the run asserts it compiled exactly the expected number of kernels.
That rule is the reason this file exists rather than a shell loop.

DEVICE — A DOOR, NOT A CENSUS

The target is passed explicitly, so no device of that kind need be present: the
same mechanism reported 98,304 bytes of shared memory for sm_70 and 164,352 for
sm_86 on a rig that holds only V100s. But "it did not touch the rig" is a claim
about one run, and this tool is meant to run beside a timed campaign.

So it does not measure that it took nothing. It REFUSES TO RUN unless the
environment has already made that impossible: `CUDA_VISIBLE_DEVICES` must be set
and empty. With no device visible, no context can be created on a real card and
no byte of its memory can be taken, whatever the stack underneath decides to do.

    A census says "not this time". A door says "never".

That is the general rule, and it is doctrine now, not a preference here: when
you are unsure whether a thing can do harm, put it in a state where it cannot,
rather than measuring that it did not. See `docs/reference/proving-by-doors.md`.

If the compilation succeeds behind that door, we have proved more than we asked
for. If it fails, the answer is just as clear — it needs a context, and it waits
for the campaign to close.

The nvidia-smi census stays available and costs nothing; it confirms the door,
it does not replace it.

Usage:
    CUDA_VISIBLE_DEVICES= python tools/kernel_boolean_ir_equality.py
    CUDA_VISIBLE_DEVICES= python tools/kernel_boolean_ir_equality.py --census
    CUDA_VISIBLE_DEVICES= python tools/kernel_boolean_ir_equality.py --target 86
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPS = REPO / "src" / "neurobrix" / "kernels" / "ops"

# The commit that carried the nineteen edits; its parent is the "before".
EDIT_COMMIT = "30c695c"

# One entry per kernel that CARRIES an edited line. Kernels reached only as a
# `combine_fn` (reduce_all, reduce_any) are compiled inside their caller and are
# named here so the count below is honest about what is covered.
#
#   signature: Triton type string per argument, "constexpr" for constexpr args
#   constexprs: their values — powers of two, tl.arange requires it
POINTER_F32, POINTER_I64, POINTER_I8, POINTER_I1 = "*fp32", "*i64", "*i8", "*i1"

SPEC: dict[str, list[tuple[str, dict, dict, tuple[str, ...]]]] = {
    "all_reduce.py": [(
        "all_kernel_dim",
        {"inp": POINTER_F32, "out": POINTER_I1, "M": "i32", "N": "i32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("all_reduce.py:14 (reduce_all, via tl.reduce)", "all_reduce.py:78", "all_reduce.py:81"),
    )],
    "any_reduce.py": [(
        "any_kernel_dim",
        {"inp": POINTER_F32, "out": POINTER_I1, "M": "i32", "N": "i32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("any_reduce.py:14 (reduce_any, via tl.reduce)", "any_reduce.py:78", "any_reduce.py:81"),
    )],
    "argmin.py": [(
        "argmin_kernel",
        {"inp": POINTER_F32, "out_index": POINTER_I64, "M": "i32", "N": "i32", "K": "i32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("argmin.py:80",),
    )],
    "max_reduce.py": [(
        "max_kernel",
        {"inp": POINTER_F32, "out_value": POINTER_F32, "out_index": POINTER_I64,
         "M": "i32", "N": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("max_reduce.py:78",),
    )],
    "min_reduce.py": [(
        "min_kernel",
        {"inp": POINTER_F32, "out_value": POINTER_F32, "out_index": POINTER_I64,
         "M": "i32", "N": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("min_reduce.py:78",),
    )],
    "prod.py": [(
        "prod_kernel",
        {"inp": POINTER_F32, "out": POINTER_F32, "M": "i32", "N": "i32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("prod.py:73",),
    )],
    "tril.py": [
        ("tril_kernel",
         {"X": POINTER_F32, "Y": POINTER_F32, "M": "i32", "N": "i32", "diagonal": "i32",
          "M_BLOCK_SIZE": "constexpr", "N_BLOCK_SIZE": "constexpr"},
         {"M_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 64},
         ("tril.py:32",)),
        ("tril_batch_kernel",
         {"X": POINTER_F32, "Y": POINTER_F32, "batch": "i32", "MN": "i32", "N": "i32",
          "diagonal": "i32", "BATCH_BLOCK_SIZE": "constexpr", "MN_BLOCK_SIZE": "constexpr"},
         {"BATCH_BLOCK_SIZE": 8, "MN_BLOCK_SIZE": 128},
         ("tril.py:59",)),
    ],
    "triu.py": [
        ("triu_kernel",
         {"X": POINTER_F32, "Y": POINTER_F32, "M": "i32", "N": "i32", "diagonal": "i32",
          "M_BLOCK_SIZE": "constexpr", "N_BLOCK_SIZE": "constexpr"},
         {"M_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 64},
         ("triu.py:30",)),
        ("triu_batch_kernel",
         {"X": POINTER_F32, "Y": POINTER_F32, "batch": "i32", "MN": "i32", "N": "i32",
          "diagonal": "i32", "BATCH_BLOCK_SIZE": "constexpr", "MN_BLOCK_SIZE": "constexpr"},
         {"BATCH_BLOCK_SIZE": 8, "MN_BLOCK_SIZE": 128},
         ("triu.py:57",)),
    ],
    "var.py": [(
        "var_welford_kernel",
        {"X": POINTER_F32, "Var": POINTER_F32, "M": "i32", "N": "i32", "correction": "fp32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        {"BLOCK_M": 32, "BLOCK_N": 64},
        ("var.py:120",),
    )],
    "weight_norm.py": [
        ("weight_norm_kernel_first",
         {"output_ptr": POINTER_F32, "norm_ptr": POINTER_F32, "v_ptr": POINTER_F32,
          "g_ptr": POINTER_F32, "M": "i32", "N": "i32", "eps": "fp32",
          "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
         {"BLOCK_M": 32, "BLOCK_N": 64},
         ("weight_norm.py:43", "weight_norm.py:54")),
        ("weight_norm_kernel_last",
         {"output_ptr": POINTER_F32, "norm_ptr": POINTER_F32, "v_ptr": POINTER_F32,
          "g_ptr": POINTER_F32, "M": "i32", "N": "i32", "eps": "fp32",
          "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
         {"BLOCK_M": 32, "BLOCK_N": 64},
         ("weight_norm.py:90", "weight_norm.py:101")),
    ],
    "where.py": [(
        "where_forward_kernel",
        {"cond_ptr": POINTER_I8, "x_ptr": POINTER_F32, "y_ptr": POINTER_F32,
         "output_ptr": POINTER_F32, "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
        {"BLOCK_SIZE": 1024},
        ("where.py:26",),
    )],
}

EXPECTED_KERNELS = sum(len(v) for v in SPEC.values())
EXPECTED_SITES = sum(len(e[3]) for v in SPEC.values() for e in v)


def _load(source: str, name: str, tmp: Path):
    """Import a module from source text.

    Written to a real file on purpose: @triton.jit reads its function's source
    through `inspect.getsource`, which has nothing to read for exec'd code.
    """
    path = tmp / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_LOC_DEF = re.compile(r"^#loc\d* = loc\(.*\)$")
_LOC_USE = re.compile(r"\s+loc\((?:#loc\d*|unknown)\)")


def _structure(ttir: str) -> str:
    """The IR with every debug-location reference removed.

    THE FIRST VERSION OF THIS FILE COMPARED THE RAW TTIR AND WAS VACUOUS. The
    two variants are written to different temporary files, so every `#loc`
    carries a different path and all fourteen kernels reported DIFFERS —
    including kernels whose bodies are provably identical. Worse, cell B was
    green for that same wrong reason: it would have announced "the gate bites"
    even if the precedence trap compiled to the same instructions.

    Line and column cannot be kept either: `mask = (a < N) & m` and
    `mask = a < N & m` put their operators at different columns, so a
    comparison that keeps them differs on the punctuation rather than on the
    computation, and cell B passes for the wrong reason a second time.

    What "semantically neutral" means is that the OPERATIONS are the same. So
    the comparison is the `tt.*` body with all locations stripped, and cell B
    is the proof that this still sees a real structural change.
    """
    kept = []
    for line in ttir.splitlines():
        if _LOC_DEF.match(line.strip()):
            continue
        kept.append(_LOC_USE.sub("", line).rstrip())
    return "\n".join(kept)


def _ttir(module, kernel: str, signature: dict, constexprs: dict, target) -> str:
    import triton
    from triton.compiler import ASTSource

    fn = getattr(module, kernel)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return _structure(triton.compile(src, target=target).asm["ttir"])


def _census(label: str) -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"  [census {label}] {out or 'no compute process'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=70, help="compute capability, e.g. 70 or 86")
    ap.add_argument("--census", action="store_true", help="print the rig's compute processes around the run")
    ap.add_argument("--allow-devices", action="store_true",
                    help="open the door: run with the rig's cards visible. Only for a "
                         "machine with nothing in flight, and never to get past the refusal.")
    args = ap.parse_args()

    # -- the door ----------------------------------------------------------
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.allow_devices and visible != "":
        print(
            "REFUSED: this tool compiles, it does not run, and it is meant to be safe\n"
            "beside a timed campaign. Make that structural rather than hoped for:\n\n"
            "    CUDA_VISIBLE_DEVICES= python tools/kernel_boolean_ir_equality.py\n\n"
            f"CUDA_VISIBLE_DEVICES is currently {visible!r}. With no device visible no\n"
            "context can be created on a real card and no byte of its memory can be\n"
            "taken, whatever the stack underneath does. A census would only say 'not\n"
            "this time'; this says 'never'.\n\n"
            "--allow-devices opens it deliberately, for a machine with nothing in flight.",
            file=sys.stderr)
        return 2
    if args.allow_devices:
        print("  !! running with the rig's cards VISIBLE (--allow-devices)")

    from triton.backends.compiler import GPUTarget
    target = GPUTarget("cuda", args.target, 32)

    if args.census:
        _census("before")

    failures: list[str] = []
    compiled = 0
    covered_sites: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- cell A: before == after -------------------------------------
        for filename, entries in SPEC.items():
            rel = f"src/neurobrix/kernels/ops/{filename}"
            before_src = subprocess.run(
                ["git", "-C", str(REPO), "show", f"{EDIT_COMMIT}^:{rel}"],
                capture_output=True, text=True, check=True,
            ).stdout
            after_src = (OPS / filename).read_text()
            if before_src == after_src:
                failures.append(f"{filename}: before and after are the SAME TEXT — "
                                f"{EDIT_COMMIT} did not touch this file, the spec is wrong")
                continue

            stem = filename[:-3]
            mod_before = _load(before_src, f"_before_{stem}", tmp)
            mod_after = _load(after_src, f"_after_{stem}", tmp)

            for kernel, signature, constexprs, sites in entries:
                try:
                    ir_before = _ttir(mod_before, kernel, signature, constexprs, target)
                    ir_after = _ttir(mod_after, kernel, signature, constexprs, target)
                except Exception as exc:  # a signature that does not compile is a FAILURE
                    failures.append(f"{filename}::{kernel} did not compile — {type(exc).__name__}: {exc}")
                    continue
                compiled += 1
                covered_sites.extend(sites)
                if ir_before != ir_after:
                    failures.append(
                        f"{filename}::{kernel} — TTIR DIFFERS before/after. The edit was NOT "
                        f"neutral on this Triton; sites {', '.join(sites)} need a numerical "
                        f"verdict against the oracle before main is measured.")
                else:
                    print(f"  A  {filename}::{kernel:<26} identical TTIR   ({', '.join(sites)})")

        # ---- cell B: the instrument must be seen catching the trap -------
        # NOTE: Triton caches compiled artefacts on disk by source hash, so a
        # warning-based check would fire only on the first run of a given
        # source and read as silence afterwards. The IR comparison is immune —
        # a cached artefact carries the same TTIR the codegen produced — which
        # is a second reason to test structure rather than diagnostics.
        botched = (OPS / "weight_norm.py").read_text().replace(
            "mask = (col_offset < N) & row_mask",
            "mask = col_offset < N & row_mask",
        )
        if "mask = col_offset < N & row_mask" not in botched:
            failures.append("cell B: could not build the unparenthesised variant — the line "
                            "weight_norm.py:43 no longer reads as this cell expects")
        else:
            mod_botched = _load(botched, "_botched_weight_norm", tmp)
            spec = SPEC["weight_norm.py"][0]
            try:
                ir_botched = _ttir(mod_botched, spec[0], spec[1], spec[2], target)
                ir_good = _ttir(_load((OPS / "weight_norm.py").read_text(),
                                      "_good_weight_norm", tmp),
                                spec[0], spec[1], spec[2], target)
            except Exception as exc:
                failures.append(f"cell B did not compile — {type(exc).__name__}: {exc}")
            else:
                if ir_botched == ir_good:
                    failures.append(
                        "cell B: the unparenthesised variant produced the SAME TTIR. This "
                        "instrument cannot see the precedence trap, so cell A's greens prove "
                        "nothing. Do not report cell A until this is understood.")
                else:
                    print("  B  weight_norm.py unparenthesised   TTIR DIFFERS — the gate bites")

    if args.census:
        _census("after")

    print()
    print(f"kernels compiled : {compiled} / {EXPECTED_KERNELS} expected")
    print(f"sites covered    : {len(covered_sites)} / {EXPECTED_SITES} expected")

    if compiled != EXPECTED_KERNELS:
        failures.append(f"only {compiled} of {EXPECTED_KERNELS} kernels compiled — a green "
                        f"verdict here would be a gate that measured nothing")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nCELLS A AND B PASS. The nineteen edits are neutral on this Triton, and the "
          "instrument was seen catching the precedence trap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
