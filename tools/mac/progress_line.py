"""Rewrites the report's `CHAIN PROGRESS` line from a running campaign's own files, every 15 min until the chain's
process is gone (supervisor 2026-10-22 12:13 / 16:36: the report follows every step, not only the agent's wakes).
Read-only on the campaign; one line of `<communication>/_report.md` replaced in place. Paths from atelier.json.
usage: progress_line.py <campaign dir> <process pattern> [<new keys to certify>]"""
import os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atelier import root, writable  # noqa: E402

R = writable(sys.argv[1]); PAT = sys.argv[2]; NEW_KEYS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
REP = writable(root("communication") / "_report.md")


def line():
    st = (R / "status").read_text().splitlines() if (R / "status").exists() else []
    phase = st[-1][11:120] if st else "?"
    cert = sum(1 for f in R.glob("certify*.log") for l in open(f, errors="replace") if re.match(r"\[certify\] [a-z_0-9]+_kernel ", l))
    ends = [l for l in open(R / "chain.out", errors="replace")] if (R / "chain.out").exists() else []
    ends = [l for l in ends if l.startswith("CELL ") and " END " in l]
    ok = lambda l: " rc=0 " in l and "zero_miss=PASS" in l
    red = [l.split()[1] + "/" + l.split()[2] for l in ends if not ok(l)]
    keys = f"certify {cert}/{NEW_KEYS} new keys; " if NEW_KEYS else ""
    return (f"CHAIN PROGRESS {time.strftime('%H:%M')} (auto, from the chain's files): last status '{phase}'; {keys}"
            f"matrix cells ended {len(ends)} ({sum(map(ok, ends))} green, red: {', '.join(red) or 'none'}); see point 4 for verdicts.")


while True:
    s = REP.read_text().split("\n")
    idx = [i for i, x in enumerate(s) if x.startswith("CHAIN PROGRESS ")]
    if idx:
        s[idx[0]] = line()
    else:
        s.insert(1, line())
    REP.write_text("\n".join(s))
    if subprocess.run(["pgrep", "-f", PAT], capture_output=True).returncode != 0:
        break
    time.sleep(900)
