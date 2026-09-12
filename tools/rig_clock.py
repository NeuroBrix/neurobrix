"""The rig runs at the protocol clock, or the measurement does not start.

ONE brick, imported by every harness whose numbers are timings -- campaigns,
benchmark rows, and the autotune certification, which picks a configuration by
timing candidates and is therefore a measurement like any other.

WHY IT IS A DOOR AND NOT A REPORT
---------------------------------
A census answers for one run: it says the clocks were right this time, and every
later run re-opens the question. This machine has no UPS and loses mains
(2026-09-11: two boots nine minutes apart), and application clocks DO NOT survive
a reboot -- each card returns to its own factory default. So the harmful state
recurs by itself, on a schedule nobody controls. Making it unreachable at entry
is the only form that holds: a number produced behind this door was produced at
the protocol clock, and that is a claim about every future run, not about one.

WHY IT READS EVERY CARD
-----------------------
This rack is heterogeneous -- 16 GB V100s on 0 and 1, 32 GB on 2 and 3 -- and the
two SKUs carry DIFFERENT factory defaults: 1312 MHz and 1290 MHz. The protocol
value is 1290. So after a reboot half the rig is already at the protocol value by
pure manufacturer coincidence, and the other half is not.

A check that samples one card goes green on a rig that is half wrong. A check
that samples card 2 or 3 goes green ALWAYS, and would never once have fired. That
is why this reads every card the driver reports and names each divergence with
its own value -- and why reading ZERO cards is itself a refusal, because an
instrument that examined nothing must never be mistaken for one that found
nothing wrong.

The two SKUs advertise byte-identical supported-clock lists, so no capability
query reveals the disagreement. Only the reading does.

WHICH CLOCK, AND THE DIVISION OF LABOUR
---------------------------------------
This door reads the APPLICATION clock (`clocks.applications.graphics`, set by
`nvidia-smi -ac`) -- the frequency the card will run work AT. It deliberately
does not read the current clock, which sits at 135 MHz on an idle card and would
make an entry check meaningless.

Holding the clock DURING a run is a separate job, already done: `bench_row.py`
locks with `-lgc` for the duration and `ClockWatch` samples `clocks.sm` every two
seconds, marking a rep contaminated on any excursion. This door is the entry
condition; that sampler is the ongoing one. Neither replaces the other -- a
transient `-lgc` lock is released in a `finally` and never survives the reboot
this door exists for.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROTOCOL_FILE = Path(__file__).resolve().parent / "rig_protocol.json"
OPT_OUT = "--allow-off-protocol-clock"


class OffProtocol(SystemExit):
    """Raised as a refusal: the rig is not in the state the protocol names."""


def protocol_clock() -> tuple[int, int]:
    """The protocol's (graphics, memory) MHz, read from the authority.

    A missing or unreadable authority is a REFUSAL, never a default. A harness
    that falls back to a built-in number stops citing the protocol and starts
    inventing one, and the divergence between the two is silent by construction.
    """
    try:
        clock = json.loads(PROTOCOL_FILE.read_text())["clock"]
        return (int(clock["application_graphics_mhz"]),
                int(clock["application_memory_mhz"]))
    except Exception as exc:
        raise OffProtocol(
            f"REFUSED: cannot read the clock protocol from {PROTOCOL_FILE} "
            f"({exc}). The protocol value is not optional and has no default: "
            f"restore the file rather than running without it.")


def rig_clocks() -> list[dict]:
    """Every physical card the driver reports, with its application clocks.

    `nvidia-smi` speaks real indices and ignores CUDA_VISIBLE_DEVICES, which is
    what this needs: a card at the wrong frequency beside a measurement is a
    fact about the rig whether or not the job was pinned away from it.
    """
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,clocks.applications.graphics,"
         "clocks.applications.memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise OffProtocol(
            f"REFUSED: cannot read the rig's clocks "
            f"({(r.stderr or r.stdout).strip() or 'nvidia-smi failed'}). A "
            f"measurement whose conditions cannot be read is not a measurement.")
    cards = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        idx, name, gfx, mem = [c.strip() for c in line.split(",")]
        cards.append({"index": idx, "name": name,
                      "graphics_mhz": int(gfx), "memory_mhz": int(mem)})
    return cards


def require_protocol_clock(allow_off_protocol: bool = False, say=print) -> dict:
    """Refuse to proceed unless EVERY card sits at the protocol clock.

    Returns the reading on success, so a caller can stamp its report with the
    conditions it actually ran under rather than with the ones it asked for.
    """
    want_gfx, want_mem = protocol_clock()
    cards = rig_clocks()

    if not cards:
        raise OffProtocol(
            "REFUSED: the driver reported ZERO cards. This check examined "
            "nothing, and a check that examined nothing must not be read as a "
            "check that found nothing wrong.")

    off = [c for c in cards
           if c["graphics_mhz"] != want_gfx or c["memory_mhz"] != want_mem]

    if not off:
        say(f"[rig] {len(cards)} card(s) read, all at the protocol clock "
            f"{want_gfx}/{want_mem} MHz")
        return {"protocol_mhz": [want_gfx, want_mem], "cards": cards,
                "cards_read": len(cards), "off_protocol": []}

    lines = [f"the rig is NOT at the protocol clock: {len(off)} of "
             f"{len(cards)} card(s) diverge (protocol {want_gfx}/{want_mem} MHz)"]
    for c in off:
        lines.append(f"    card {c['index']}  {c['name']}  "
                     f"{c['graphics_mhz']}/{c['memory_mhz']} MHz")
    for c in cards:
        if c not in off:
            lines.append(f"    card {c['index']}  {c['name']}  "
                         f"{c['graphics_mhz']}/{c['memory_mhz']} MHz  (at protocol)")
    lines.append("")
    lines.append("  Application clocks do not survive a reboot and each SKU "
                 "returns to its OWN factory default, so half a heterogeneous "
                 "rack can sit at the protocol value by coincidence.")
    lines.append("")
    lines.append("  Restore every card, then re-run:")
    lines.append("      for i in " + " ".join(c["index"] for c in cards) +
                 f"; do sudo nvidia-smi -i $i -ac {want_mem},{want_gfx}; done")

    if allow_off_protocol:
        say("[rig] WARNING, " + lines[0])
        for line in lines[1:]:
            say("[rig] " + line)
        say(f"[rig] proceeding anyway on {OPT_OUT}: every number produced by "
            f"this run was taken off protocol and may not be compared with one "
            f"that was not.")
        return {"protocol_mhz": [want_gfx, want_mem], "cards": cards,
                "cards_read": len(cards), "off_protocol": off,
                "waived": True}

    raise OffProtocol("REFUSED: " + "\n".join(lines) +
                      f"\n\n  To measure off protocol deliberately, pass "
                      f"{OPT_OUT} — the run then says so in its own output.")


def add_argument(parser) -> None:
    """Wire the one deliberate opening, spelled the same way everywhere."""
    parser.add_argument(
        OPT_OUT, action="store_true",
        help="run even though a card is off the protocol clock (the run says "
             "so, and its numbers are not comparable with on-protocol ones)")


if __name__ == "__main__":
    import sys
    try:
        state = require_protocol_clock(
            allow_off_protocol=OPT_OUT in sys.argv)
    except OffProtocol as exc:
        print(exc)
        sys.exit(1)
    print(json.dumps(state, indent=2))
