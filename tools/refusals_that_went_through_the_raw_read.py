#!/usr/bin/env python3
"""Which recorded refusals could be artefacts of the oracle's raw bf16 read.

`screen_oracle._to_f64` reads a bf16 operand by RAW POINTER --
`ctypes.string_at(t.data_ptr(), n)` -- while every other dtype goes through
`t.numpy()`. If that read is wrong, the oracle is systematically wrong for any
comparison involving a bf16 operand, and a systematically wrong reference
CONTRADICTS EVERY CORRECT ANSWER exactly as loudly as a wrong one.

A refusal is believed. That is what makes an unexamined one dangerous: the
empty red of the vacuous-guard register. So before the read is repaired, every
refusal that could have come through it is listed -- because if the read is
broken, each of them is a claim that must be withdrawn, including the ones
already cited outside this repository.

The test is the presence of a bf16 OPERAND in the key, not the file's dtype
label: `addmm_kernel.fp32` carries keys whose bias is bf16, and those are
exactly the ones at issue.

    tools/refusals_that_went_through_the_raw_read.py <log-or-dir> [...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

#: A refusal the screen or the certification recorded, and what it refused.
_SCREEN = re.compile(
    r"NeuroBrix autotune screen: (\w+) at key \(([^)]*(?:\)[^)]*)*?)\)\s*—\s*(.+)")
_CERTIFY = re.compile(r"\[certify\] (\w+) (\w+) ([^:]+): FAILED — (.+)")

#: The line that carries the OPERAND dtypes, emitted for the same kernel just
#: before the screen runs. Without it a screen refusal cannot be classified.
_CONTEXT = re.compile(
    r"no certified setting for (\w+) \w+ \(([^)]*?)((?:fp\d+|bf16|int\d+)(?:,(?:fp\d+|bf16|int\d+))+)\)")

#: bf16 anywhere in the key's dtype list.
_BF16 = re.compile(r"\bbf16\b|bfloat16")


_TWO_GIB = 2 * 1024 ** 3

#: The operand dtype list is carried on the context line, captured into the
#: `where` label. bf16 sits at a known position for the kernels seen so far;
#: the shape comes from the M/N/K on that same line.
_MNK = re.compile(r"M=(\d+)\s+N=(\d+)\s+K=(\d+)")
_OPERANDS = re.compile(r"operands:\s*([a-z0-9,]+)")


def _bf16_operand_bytes(key: str, where: str):
    """Bytes the bf16 operand occupies, or None when it cannot be derived.

    None is NOT zero: a refusal whose buffer size is unknown cannot be cleared
    of the size mechanism, and saying so is the point.
    """
    m_ops = _OPERANDS.search(where)
    if not m_ops or "bf16" not in m_ops.group(1):
        return None
    dtypes = m_ops.group(1).split(",")
    m = _MNK.search(key) or _MNK.search(where)
    if not m:
        return None
    M, N, K = (int(x) for x in m.groups())
    # addmm: (a=MxK, b=KxN, bias=N, out=MxN). The bf16 one decides the size.
    idx = dtypes.index("bf16")
    elems = {0: M * K, 1: K * N, 2: N, 3: M * N}.get(idx)
    return None if elems is None else elems * 2


def _logs(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from sorted(p.rglob("*.log"))
        elif p.exists():
            yield p


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    screen_hits, certify_hits, other = [], [], []
    files = 0
    for path in _logs(sys.argv[1:]):
        files += 1
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        # The screen's refusal line names the kernel and its CONSTEXPR key --
        # `EPILOGUE`, `IEEE_PRECISION` -- and carries no dtypes at all. The
        # dtypes are on the `[autotune] no certified setting for <kernel>
        # <dtype> (... fp32,fp32,bf16,fp32)` line that precedes it for the
        # same kernel. Reading only the refusal line reported ZERO screen
        # refusals with a bf16 operand while one was known by hand -- the
        # instrument answering about the syntax of one line instead of about
        # the operands of the event.
        dtypes_by_kernel = {}
        for line in text.splitlines():
            m_ctx = _CONTEXT.search(line)
            if m_ctx:
                # Keep the SHAPE as well as the dtypes: without M/N/K the
                # buffer size cannot be derived, and "size not determinable"
                # for all eighteen is a non-answer wearing an honest word.
                shape = _MNK.search(m_ctx.group(2))
                dtypes_by_kernel[m_ctx.group(1)] = (
                    m_ctx.group(3),
                    f"M={shape.group(1)} N={shape.group(2)} K={shape.group(3)}"
                    if shape else "")
            m = _SCREEN.search(line)
            if m:
                kernel, key, why = m.group(1), m.group(2), m.group(3)
                operands, shape = dtypes_by_kernel.get(kernel, ("", ""))
                has_bf16 = bool(_BF16.search(line) or _BF16.search(operands))
                where = (f"{path.name} (operands: {operands or 'unknown'}"
                         + (f" {shape}" if shape else "") + ")")
                (screen_hits if has_bf16 else other).append(
                    (where, kernel, key[:90], why[:90]))
                continue
            m = _CERTIFY.search(line)
            if m:
                kernel, dtype, key, why = m.groups()
                (certify_hits if _BF16.search(line) else other).append(
                    (path.name, f"{kernel}.{dtype}", key[:90], why[:90]))

    # Carrying a bf16 operand is not enough: the question is whether the
    # ORACLE RAN. A certification that failed on a key mismatch never reached
    # a comparison, so the raw read cannot have decided it -- listing it beside
    # a verdict the oracle actually gave would inflate what must be withdrawn,
    # and an inflated withdrawal is as unusable as none.
    _ORACLE_SPOKE = ("oracle contradicts", "deviation", "oracle refused")
    rests_on_read = [h for h in screen_hits + certify_hits
                     if any(t in h[3] for t in _ORACLE_SPOKE)]
    oracle_silent = [h for h in screen_hits + certify_hits
                     if h not in rests_on_read]

    print(f"logs read                                      : {files}")
    print(f"refusals carrying a bf16 operand               : "
          f"{len(screen_hits) + len(certify_hits)}")
    print(f"  of which the ORACLE actually decided          : {len(rests_on_read)}")
    print(f"  of which failed BEFORE any comparison         : {len(oracle_silent)}")
    print(f"refusals without a bf16 operand (unaffected)   : {len(other)}")
    print()
    # TWO mechanisms produce "the oracle contradicts everything", and the
    # inventory must not merge them either. Addressability is one. The other,
    # measured on the other machine 2026-09-07: `ctypes.string_at(ptr, n)`
    # hands `n` to `PyBytes_FromStringAndSize` as a C int, so any buffer of
    # 2 GiB or more returns "Negative size" -- seven census shapes reported
    # "no config could run" while every configuration was fine.
    #
    # They are told apart by SIZE, and nothing else. So the bf16 operand's
    # byte count is computed for every refusal: if the refused lean large the
    # mechanism is the size, and the small ones remain valid refusals.
    print("REST ON THE RAW READ -- and the size that tells the two mechanisms apart:")
    print(f"  ({_TWO_GIB} bytes is where `string_at` starts returning "
          f"'Negative size')")
    big = small = unknown = 0
    for where, kernel, key, why in rests_on_read:
        nbytes = _bf16_operand_bytes(key, where)
        if nbytes is None:
            tag, unknown = "taille inconnue", unknown + 1
        elif nbytes >= _TWO_GIB:
            tag, big = f"{nbytes} o — AU-DESSUS du seuil", big + 1
        else:
            tag, small = f"{nbytes} o — sous le seuil", small + 1
        print(f"  [{where}] {kernel}  <{tag}>")
        print(f"      key: {key}")
        print(f"      why: {why}")
    print()
    print(f"  bf16 operand at or above 2 GiB : {big}")
    print(f"  below it                       : {small}")
    print(f"  size not determinable          : {unknown}")
    if big == 0 and unknown == 0:
        print("  -> the size mechanism cannot explain ANY of these. If the raw")
        print("     read is at fault here it is for addressability or staleness,")
        print("     and the two must not be withdrawn together.")
    print()
    print("THE ORACLE NEVER RAN on these; they carry bf16 and are a different")
    print("defect. They are listed so the two are not confused:")
    for where, kernel, key, why in oracle_silent:
        print(f"  [{where}] {kernel}  -- {why[:70]}")
    print()
    print("A refusal is believed. That is what makes an unexamined one the empty")
    print("red of the register, and why the withdrawal list is drawn BEFORE the")
    print("read is repaired rather than after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
