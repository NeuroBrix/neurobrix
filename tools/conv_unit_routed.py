"""Which conv2d_forward_kernel rows of an epoch re-prove ledger may be RETIRED: a key no call can form on the profile.

A conv key is formed by `launch_keys.conv2d_launches` (the census and `conv_wrappers.conv2d_wrapper` decide with the
same functions): a launch the profile's matrix unit takes (`matrix_unit_conv_operands`) forms no key; any other forms
`(N, in_c, in_h, IW, out_c, out_h, OW, kh, kw, ...)`. A ONE-ROW convolution keys its width on the profile's W ladder
(`conv_width_key`): every input width of the bucket forms the key of the bucket's top, and the unit's decision depends
on the output width of EACH (M = N * out_w). So this tool walks EVERY width the key's bucket holds, as
gemm_row_representative walks every M of a GEMM bucket, and calls the library's own `conv2d_launches` at each, with the
depthwise route on and off. A 2-D convolution keeps its exact extents: its one width is the key's (a band of a taller
input forms the key at the key's own rows, so the same M). The key is RETIRED only when no width forms it.

Verdicts, by the ledger's reason (`never_reached`: "the wrapper never reached the autotuner"; `unreachable_key`: the
wrapper computed another key; `ok`: re-proven):
  retired_unit_routed       never_reached, and no width of the bucket forms the key;
  failed_reachable          never_reached, yet some width forms it (FMA kernel): NOT retired, it stays failed;
  reproven_live             ok, and the key is formed;
  named_apart_width_bucket  unreachable_key, and no width forms it (the key's width is no ladder top): stays failed;
  EXCEPTION                 anything else (counted on stderr; must be 0).

Usage: python tools/conv_unit_routed.py <profile.yml> <ledger.tsv> > <out.tsv>   (run with the library's python)
"""
import ast
import sys
from pathlib import Path

import yaml

COLS = ["file", "memory_class", "outcome", "reason_class", "unit_routed", "verdict", "M", "N", "kh", "kw", "x", "w",
        "formed_at", "key"]
KERNEL = "/conv2d_forward_kernel."
_BAND = 1 << 62            # no band streaming: a one-row conv cannot be cut in rows, a 2-D band keys its own rows


def bind_profile(yml):
    """Put the profile file in force for the library (its ladders and its matrix unit) and return the unit."""
    from nbxtensor.ops import _configs
    path = Path(yml)
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(yaml.safe_load(path.read_text()))
    _configs._ACTIVE_PROFILE.update({"_vendor": path.parent.name, "_profile": path.stem, "_exact": True})
    return _configs.matrix_unit()


def bucket_widths(key):
    """Every input width that keys `key`'s width: the whole W bucket of a one-row conv (none when the key's width is
    no ladder top), the key's own width for a 2-D conv."""
    from nbxtensor.ops.autotune_bucket import bucket_of
    in_h, iw, out_h = int(key[2]), int(key[3]), int(key[5])
    if not (in_h == 1 and out_h == 1):
        return [iw]
    out, w = [], iw
    while w >= 1 and bucket_of("W", w) == iw:
        out.append(w)
        w -= 1
    return out


def _dtype(tag_text):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    return next(d for d in (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32) if LK.tag(d) == tag_text)


def walk(key, unit):
    """(widths that form the key, the unit takes every width, (min M, max M)) — through `conv2d_launches`."""
    from nbxtensor.ops import launch_keys as LK
    N, ic, ih, _iw, oc, _oh, _ow, kh, kw, sh, sw, ph, pw, dh, dw, g = (int(v) for v in key[:16])
    x, w, out = _dtype(key[17]), _dtype(key[18]), _dtype(key[19])
    formed, unit_all, ms = [], True, []
    for width in bucket_widths(key):
        oh, ow = LK.conv_out_hw(ih, width, kh, kw, sh, sw, ph, pw, dh, dw)
        ms.append(N * oh * ow)
        unit_all &= LK.matrix_unit_conv_operands(x, w, N * oh * ow, oc // max(1, g), kh, kw, unit) is not None
        if any((LK.CONV2D, tuple(key)) in LK.conv2d_launches(N, ic, ih, width, oc, kh, kw, sh, sw, ph, pw, dh, dw,
                                                                g, x, w, out, _BAND, dep, unit)
               for dep in (True, False)):
            formed.append(width)
    return formed, (unit_all if ms else None), ((min(ms), max(ms)) if ms else None)


def reason_class(outcome, reason):
    if "never reached the autotuner" in reason:
        return "never_reached"
    if reason.startswith("unreachable"):
        return "unreachable_key"
    return "ok" if outcome == "reproven" else "other"


def verdict(rc, formed):
    return {("never_reached", False): "retired_unit_routed", ("never_reached", True): "failed_reachable",
            ("ok", True): "reproven_live", ("unreachable_key", False): "named_apart_width_bucket"
            }.get((rc, bool(formed)), "EXCEPTION")


def _span(v):
    return "-" if not v else (f"{min(v)}..{max(v)}" if len(v) > 1 else str(v[0]))


def rows(ledger, unit):
    """One output row per conv2d_forward_kernel line of the ledger, in ledger order."""
    seen = {}
    for line in open(ledger):
        f = line.rstrip("\n").split("\t")
        if len(f) < 6 or KERNEL not in f[0]:
            continue
        key = ast.literal_eval(f[1])
        if key not in seen:
            seen[key] = walk(key, unit)
        formed, unit_all, m = seen[key]
        rc = reason_class(f[3], f[5])
        yield [f[0], f[2], f[3], rc, unit_all, verdict(rc, formed), _span(list(m) if m else None),
               int(key[4]) // max(1, int(key[15])), key[7], key[8], key[17], key[18], _span(formed), f[1]]


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    unit = bind_profile(a[0])
    print("\t".join(COLS))
    bad = 0
    for r in rows(a[1], unit):
        bad += r[5] == "EXCEPTION"
        print("\t".join(map(str, r)))
    print(f"# exceptions {bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
