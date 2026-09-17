"""A chain that names ITSELF as the checkpointer's producer can never finish.

The checkpointer's contract is that it holds its producers and exits when the
last is gone, "so a chain can wait on IT". Naming the chain as the producer
inverts that into a mutual wait.

Measured, 2026-09-17: `reproof_t38_v2.sh` passed `--producer-pid $$` and then
ran `wait` over its own job table. All four certifiers finished — card 3 at
02:53 and card 1 at 02:57, card 2 at 04:13, 4172 and 5942 shapes proven — and
the chain never wrote `== reproof t38 done`. It sat in `wait` for the eight
hours after its work was complete. Nothing starved on it only by luck; a waiter
on that marker would have waited for ever behind a chain that had SUCCEEDED,
which is register 55 from the other side: *not yet* and *never* reading alike.
"""

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "certified_checkpoint.py"
_spec = importlib.util.spec_from_file_location("certified_checkpoint", TOOL)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

refuse = cc.refuse_a_producer_that_will_wait_for_us


def test_the_launching_shell_as_producer_is_refused():
    msg = refuse([4242], 4242)
    assert msg, "producer == our own parent is the mutual wait"
    assert "4242" in msg


def test_the_refusal_says_what_to_do_instead():
    # A door refuses at entry WITH the command that satisfies it.
    msg = refuse([99], 99)
    assert "--allow-parent-as-producer" in msg
    assert "doing the WORK" in msg


def test_real_producers_are_not_refused():
    # The normal shape: the certifiers' own pids, none of them our parent.
    assert refuse([111, 222], 999) == ""


def test_the_parent_hidden_among_several_producers_is_still_caught():
    assert refuse([111, 999, 222], 999) != "", "one bad pid in the list is enough"


def test_no_producers_at_all_is_not_a_mutual_wait():
    assert refuse([], 999) == ""


def test_the_door_has_a_deliberate_opening():
    """The escape exists and reads as deliberate — a shell that launches the
    checkpointer and then exits without waiting is a legitimate caller."""
    import argparse
    src = TOOL.read_text()
    assert '"--allow-parent-as-producer"' in src
    assert "action=\"store_true\"" in src
    # and it is an opening, not a silent bypass: the refusal names it.
    assert "--allow-parent-as-producer" in refuse([7], 7)


def test_the_door_is_WIRED_not_merely_correct(tmp_path):
    """Register 17: a helper whose every test passes can still have no seam.

    This runs the tool for real, from a shell that names ITSELF as the
    producer — the exact shape that deadlocked — and requires the refusal to
    come out of `main()`, not just out of the function above.
    """
    import subprocess
    import sys

    script = (
        "import os, subprocess, sys\n"
        f"r = subprocess.run([sys.executable, {str(TOOL)!r}, '--repo', {str(tmp_path)!r},\n"
        "                    '--producer-pid', str(os.getpid()), '--once'],\n"
        "                   capture_output=True, text=True)\n"
        "print(r.returncode); print(r.stderr)\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    first = out.stdout.splitlines()[0] if out.stdout else ""
    assert first == "2", f"main() must refuse the mutual wait, got rc={first!r}\n{out.stdout}{out.stderr}"
    assert "REFUSED" in out.stdout
