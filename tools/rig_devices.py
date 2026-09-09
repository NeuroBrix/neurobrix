"""Device indices, resolved through the pin a job actually runs under.

ONE brick, imported by every harness that places work on a card. It exists
because the same defect was written twice, independently, in the two tools that
produce this rig's numbers:

    env["CUDA_VISIBLE_DEVICES"] = str(gpu)      # tools/precision_zoo_campaign.py
    env["CUDA_VISIBLE_DEVICES"] = args.gpu      # benchmarks/harness/bench_row.py

`CUDA_VISIBLE_DEVICES` makes device indices RELATIVE: under an outer pin of "2",
index 0 IS card 2. Writing the index straight back discards the pin, so a job the
scheduler declared on one card runs on another. Two consequences, and the second
is the one that corrupts measurements rather than merely misplacing them:

  * the work lands on a card nobody chose — on 2026-09-09 a 30 GB row declared on
    a 32 GB card ran on a 16 GB one and its arm was recorded FAILED;
  * the scheduler's own accounting goes wrong. It believes the declared card is
    busy and the real one free, so it can start an untimed job right beside a
    timed measurement while reporting the machine exclusive — the exact
    falsification `7759b3f` was written to end.

`nvidia-smi -i N` always speaks REAL indices, whatever the pin, so a harness that
locks clocks or reads state must use the resolved physical index for those too,
or it watches one card while computing on another.
"""
from __future__ import annotations

import os


def visible_card(gpu) -> str:
    """The physical card `gpu` names, resolved through a pin we inherited.

    An index the pin does not expose is REFUSED, never guessed at: a device a
    harness cannot resolve is a placement nobody decided.
    """
    inherited = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not inherited:
        return str(gpu)
    cards = [c.strip() for c in inherited.split(",") if c.strip()]
    try:
        return cards[int(gpu)]
    except (ValueError, IndexError):
        raise SystemExit(
            f"--gpu {gpu} names no card inside the pin CUDA_VISIBLE_DEVICES="
            f"{inherited!r}, which exposes {len(cards)}: {cards}. A device that "
            f"cannot be resolved is never guessed at."
        )


def gate_card(default_env: str = "NBX_GATE_GPU", default: str = "1") -> str:
    """The card a reference gate runs on.

    The default names a physical card of this rig ("never the condemned card").
    Under an inherited pin the caller has already chosen, so there is nothing for
    that default to choose between: take the first card we were given rather than
    reach outside the pin.
    """
    inherited = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if inherited:
        return inherited.split(",")[0].strip()
    return os.environ.get(default_env, default)
