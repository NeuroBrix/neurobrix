"""Every flat-indexed kernel computes its element offsets in 64-bit.

The int32 form `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps past
2^31 elements; the project fixed it by hand in `add` (Sana 4K VAE, 2026-05)
and `silu`, and met it again in the GEMM epilogue (Mochi, register 58). On
2026-09-14 129 sites in 90 kernels still carried the int32 form; all were
promoted at once. This gate refuses the form coming back — a door, not a
census: the pattern is read from the source, so a kernel added tomorrow with
the int32 form is red before it meets its first large tensor.

Two forms are refused, because the first fix chose the wrong one: casting
AFTER the product — `(pid * BLOCK_SIZE + tl.arange(...)).to(tl.int64)` — widens
a sum that has already wrapped (pid * BLOCK_SIZE is int32 * constexpr). The
gate's first form accepted it, and the 2.25e9-element fill still faulted
(2026-09-14 06:31, card 2): a gate that reads the source proves the text,
not the arithmetic — the GPU test beside it is the one that proves the
arithmetic. The form that holds widens the program id itself:
`pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` (FlagGems #6083).

Injection: one line of `gelu.py` restored to the int32 form made this RED;
one line restored to the cast-after form made this RED; restored, green.
"""
import re
from pathlib import Path

OPS = Path(__file__).resolve().parents[3] / "src/neurobrix/kernels/ops"
INT32_FORM = re.compile(r"^\s*\w+ = \(?pid\w* \* (BLOCK_SIZE|BLOCK) \+ tl\.arange\(0, (BLOCK_SIZE|BLOCK)\)\)?(\.to\(tl\.int64\))?\s*(#.*)?$", re.M)


def test_no_flat_offset_is_computed_in_int32():
    hits = []
    for f in sorted(OPS.glob("*.py")):
        for m in INT32_FORM.finditer(f.read_text(encoding="utf-8")):
            hits.append(f"{f.name}: {m.group(0).strip()}")
    assert not hits, ("flat offsets whose product is computed in int32 — it wraps past 2^31 elements; write "
                      "`pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` (widen the program id):\n  " + "\n  ".join(hits))


def test_the_gate_reads_the_form_it_claims_to(tmp_path):
    assert INT32_FORM.search("    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n")
    assert INT32_FORM.search("    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # x\n")
    assert INT32_FORM.search("    offset = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)\n"), "cast-after wraps too"
    assert not INT32_FORM.search("    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n")
