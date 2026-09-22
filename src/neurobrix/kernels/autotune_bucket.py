"""The bucket of a request-dependent dimension in an autotune key.

The owner's decision (2026-09-21): request-dependent dimensions — a prefill's token count,
a decode's key length, a convolution's spatial extents and batch — are BUCKETED in the
launcher's key. The bucket selects the configuration; the kernel still runs the true size
with its masks. Without it a census can only cover the requests it is shown, a 57-token
prompt meets a key nobody certified, and zero miss at verification can never be true.

The ladder is DATA, on the vendor profile (`autotune.buckets`), chosen by measurement the
way the tile unit was (`tools/bucket_loss.py`, owed-proofs 2026-09-21): on Volta, both
memory classes, matmul M and the SDPA scores' batched GEMM — exact under 64 (the batched
GEMM's optimum moves with every size there: up to 16.7 % lost inside a 16-wide bucket),
then 16-step to 256, 32-step to 1024, 128-step to 8192, 512-step beyond: 0.0 % median and
maximum loss against the per-size optimum, where powers of two lost up to 26.9 %.

A profile that is BOUND but declares no ladder is REFUSED, not quietly bucketed at step 1.
Answering "exact" there is a silent degradation dressed as a default: it produces a key
explosion with no error, and it hid for a day that the ladder had been added to
nvidia/volta.yml alone while all twenty-six other profiles carried none (found 2026-09-22 by
the schema gate, not by a run). An UNBOUND profile — `{}`, no target matched, the census
behind its door before `install()` — is a different condition and still keys exact; that one
is covered by binding the shadow to its profile, not by a refusal here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: A ladder is a list of [upper_bound, step] rows in ascending bound order: a value v takes
#: the smallest multiple of `step` >= v within the first row whose bound is >= v. A step of
#: 1 is "exact". The last row's bound may be null (open).
Ladder = Sequence[Tuple[Optional[int], int]]


_PARSED: Dict[str, Ladder] = {}


def parse_ladder(rows: Any) -> Ladder:
    """Parsed once per distinct row list: the key of every launch reads the ladder (32 616
    parses in one chatterbox shadow, 2026-09-22)."""
    memo = repr(rows)
    got = _PARSED.get(memo)
    if got is not None:
        return got
    out = _parse_ladder(rows)
    _PARSED[memo] = out
    return out


def _parse_ladder(rows: Any) -> Ladder:
    out: List[Tuple[Optional[int], int]] = []
    for r in rows or []:
        if isinstance(r, dict):
            bound, step = r.get("up_to"), r.get("step")
        else:
            bound, step = r[0], r[1]
        if step is None or int(step) < 1:
            raise ValueError(f"ZERO FALLBACK: a bucket ladder row needs a step >= 1: {r!r}")
        out.append((None if bound is None else int(bound), int(step)))
    for i in range(1, len(out)):
        if out[i - 1][0] is None or (out[i][0] is not None and out[i][0] <= out[i - 1][0]):
            raise ValueError(f"ZERO FALLBACK: bucket ladder bounds must ascend and only the last may be open: {rows!r}")
    return out


def bucket(value: int, ladder: Ladder) -> int:
    """The bucket's TOP for `value`: the smallest multiple of the row's step that is >= value."""
    v = int(value)
    if v <= 0:
        return v
    for bound, step in ladder:
        if bound is None or v <= bound:
            return -(-v // step) * step
    bound, step = ladder[-1]
    return -(-v // step) * step


def ladder_for(dim: str, profile: Optional[Dict[str, Any]] = None) -> Ladder:
    """The profile's ladder for `dim` ("M", "N", "batch", "height", "width"), else exact."""
    if profile is None:
        from neurobrix.kernels.ops._configs import active_vendor_profile
        profile = active_vendor_profile()
    bound = bool(profile)
    spec = ((profile or {}).get("autotune") or {}).get("buckets") or {}
    rows = spec.get(dim) if isinstance(spec, dict) else None
    if rows is None:
        rows = spec.get("default") if isinstance(spec, dict) else None
    if rows:
        return parse_ladder(rows)
    if bound:
        # ZERO FALLBACK: a real profile that forgot the ladder is a defect in the profile,
        # and keying exact would hide it behind a plausible-looking answer.
        raise RuntimeError(
            "ZERO FALLBACK: the bound hardware profile "
            f"{(profile or {}).get('architecture') or '<unnamed>'!r} declares no "
            f"autotune.buckets ladder, so dimension {dim!r} has no bucket.\n"
            "  Add `autotune.buckets.default` to its vendor YAML (see nvidia/volta.yml), "
            "or the census keys this dimension EXACTLY and every request length becomes "
            "its own key.\n"
            "  A step of 1 is a legitimate ladder and must be written down as one."
        )
    return [(None, 1)]        # no target bound: the census's own door, unchanged


def bucket_of(dim: str, value: int, profile: Optional[Dict[str, Any]] = None) -> int:
    return bucket(value, ladder_for(dim, profile))
