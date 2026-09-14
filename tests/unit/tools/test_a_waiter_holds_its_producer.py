"""Register entry 55 — a waiter that watches the result and never the producer.

The shape it must refuse: the marker's writer dies AFTER its job succeeded and
BEFORE writing the marker. A bare `until grep -q` waits for ever (four cards
idle 17:12→21:25 on 2026-09-13). `tools/wait_for.py` holds the producer's pid
and refuses the moment it is gone without its marker.

Injection: the producer is a `sleep`, killed by the test. Seen RED on
2026-09-13 22:20 UTC against a waiter with no liveness — `producer_alive`
forced to True, which is the bare `until grep -q` behaviour — the
dead-producer case returned 5 (the test's own timeout, 3 s) instead of 3:
without the bound the waiter starves. With the brick, 3 in under 0.1 s.
"""
import os
import subprocess
import sys
import time
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import wait_for as WF  # noqa: E402


def _producer():
    return subprocess.Popen(["sleep", "60"])


def test_marker_present_returns_zero_at_once(tmp_path):
    rec = tmp_path / "RUN.md"
    rec.write_text("== rejeu 2 termine 17:12:00 ==\n")
    p = _producer()
    try:
        assert WF.wait_for(str(rec), r"== rejeu 2 termine", p.pid, poll=0.05, timeout=2) == 0
    finally:
        p.kill(); p.wait()


def test_a_dead_producer_without_its_marker_is_refused_not_awaited(tmp_path):
    """The injection: the producer is killed after the record is complete
    but before the marker — exactly the 2026-09-13 17:12 state."""
    rec = tmp_path / "RUN.md"
    rec.write_text("8 failed, 1 passed in 1030.02s\n")   # the job's output, complete
    p = _producer()
    p.kill(); p.wait()                                    # the marker-writer is gone
    t0 = time.monotonic()
    rc = WF.wait_for(str(rec), r"== rejeu 2 termine", p.pid, poll=0.05, timeout=30)
    assert rc == 3, "a gone producer with the marker absent must be a REFUSAL (3), not a wait"
    assert time.monotonic() - t0 < 5, "the refusal is immediate, not the timeout"


def test_a_living_producer_is_awaited_until_it_writes(tmp_path):
    rec = tmp_path / "RUN.md"
    rec.write_text("")
    p = _producer()
    try:
        # write the marker from a side thread after 0.3 s, while the producer lives
        import threading
        def later():
            time.sleep(0.3)
            with open(rec, "a") as f:
                f.write("== rejeu 2 termine 21:25:39 ==\n")
        threading.Thread(target=later, daemon=True).start()
        assert WF.wait_for(str(rec), r"== rejeu 2 termine", p.pid, poll=0.05, timeout=5) == 0
    finally:
        p.kill(); p.wait()


def test_a_stale_heartbeat_is_refused(tmp_path):
    rec = tmp_path / "RUN.md"; rec.write_text("")
    hb = tmp_path / "heartbeat"; hb.write_text("x")
    old = time.time() - 100
    os.utime(hb, (old, old))
    p = _producer()
    try:
        rc = WF.wait_for(str(rec), r"never", p.pid, poll=0.05, heartbeat=str(hb), stale_after=10, timeout=5)
        assert rc == 4
    finally:
        p.kill(); p.wait()


def test_the_cli_refuses_with_the_producer_named(tmp_path):
    rec = tmp_path / "RUN.md"; rec.write_text("done\n")
    p = _producer(); p.kill(); p.wait()
    r = subprocess.run([sys.executable, str(TOOLS / "wait_for.py"), "--file", str(rec),
                        "--marker", "== fin ==", "--producer-pid", str(p.pid), "--poll", "0.05",
                        "--timeout", "10"], capture_output=True, text=True)
    assert r.returncode == 3
    assert "REFUSED" in r.stderr and str(p.pid) in r.stderr and "last write" in r.stderr


def test_a_marker_already_in_the_record_is_not_this_waits_marker(tmp_path):
    """Seen on 2026-09-13 22:30: a waiter for `mochi rc=` returned 0 at once on the
    afternoon's perturbed run's line, written hours before the run it watched.
    Injection = the default (any line, the only behaviour before the fix) →
    returns 0 on the stale line; `from_now=True` ignores it and waits for a NEW
    line. The default stays "any line": a waiter armed after its producer
    already wrote the marker must see it (it would otherwise refuse a finished
    job), so a record that accumulates markers is watched with `--from-now`."""
    rec = tmp_path / "RUN.md"
    rec.write_text("   mochi rc=1 16:05:00; sanitizer: 2 lines\n########## mochi again 21:26:07\n")
    p = _producer()
    try:
        assert WF.wait_for(str(rec), r"mochi rc=", p.pid, poll=0.05, timeout=2) == 0, "any line: the stale line is accepted (what happened at 22:30)"
        assert WF.wait_for(str(rec), r"mochi rc=", p.pid, poll=0.05, timeout=0.5, from_now=True) == 5, "from now: the stale line does not count, the wait goes on"
        import threading
        def later():
            time.sleep(0.3)
            with open(rec, "a") as f:
                f.write("   mochi rc=124 23:26:07; sanitizer: 0 lines\n")
        threading.Thread(target=later, daemon=True).start()
        assert WF.wait_for(str(rec), r"mochi rc=", p.pid, poll=0.05, timeout=5, from_now=True) == 0
    finally:
        p.kill(); p.wait()
