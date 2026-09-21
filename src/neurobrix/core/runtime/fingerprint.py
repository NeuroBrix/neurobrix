"""How many bytes of an op output the fingerprint hashes — one door for both engines.

The two fingerprint paths (the ATen sequential loop in graph_executor and the Triton
sequence) hashed the first 8 192 bytes of each output by default and named the first
op whose PREFIX differed. On 2026-09-20 that named a layer-norm mean for Kokoro while
the layer-norm's input already differed past the prefix; the real levers were two
`aten::pow` commits, found by bisection. Every attribution made on the prefix is
invalid (the owner, 2026-09-21). The default is now the whole tensor; a cap is an
explicit, visible choice (`NBX_OP_FINGERPRINT_CAP=<bytes>`) and the record still says
how many bytes it hashed.
"""
from __future__ import annotations

import os

ENV_CAP = "NBX_OP_FINGERPRINT_CAP"


def hashed_span(nbytes: int, cap: int | None = None) -> int:
    """The number of bytes to hash for an output of `nbytes`: all of them unless a cap
    (the env, or the argument) says fewer. A cap of 0 means the whole tensor."""
    if cap is None:
        cap = int(os.environ.get(ENV_CAP, "0") or 0)
    return nbytes if cap <= 0 else min(nbytes, cap)
