#!/usr/bin/env python3
"""Run a chain script from a FROZEN copy, so editing the script can never change a running chain.

Bash reads a script by byte offset while it runs: it parses one compound command (a whole loop),
runs it, then reads on from where it stopped in the FILE. Edit the file meanwhile and the next read
lands at the old offset in the new text. Measured 2026-09-29: merge-queue-16's 32a gate ran
`gate.sh`, the script was rewritten in place at 16:28 (464 bytes longer), and when the loop ended
bash resumed at byte 1 280 of the new file, mid-line — `echo "gate 32a done"` never ran, the chain
behind it (`after3`) waited for that marker and refused at 17:21, and the four cards stood idle
until the next session. "A running bash script is never edited" was a rule; this is the door.

    python tools/run_frozen.py <script.sh> [args...]

copies the script to `<script dir>/.frozen/<name>.<YYYYmmdd-HHMMSS>.<pid>.sh`, read-only, and
`exec`s bash on the COPY (the same pid, the same arguments, the same environment and cwd). The
source can then be edited, rewritten or deleted; the running chain keeps reading the bytes it
started with. The copy's path and sha256 are said on stderr, so a log names exactly what ran.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import time
from pathlib import Path


def frozen_copy(script: Path) -> Path:
    """A read-only copy of `script` beside it, under `.frozen/`, named by time and pid."""
    script = script.resolve()
    if not script.is_file():
        raise SystemExit(f"run_frozen: {script}: no such script")
    d = script.parent / ".frozen"
    d.mkdir(exist_ok=True)
    dst = d / f"{script.stem}.{time.strftime('%Y%m%d-%H%M%S')}.{os.getpid()}{script.suffix or '.sh'}"
    shutil.copyfile(script, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return dst


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: run_frozen.py <script.sh> [args...]")
    dst = frozen_copy(Path(argv[0]))
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
    print(f"[run_frozen] {argv[0]} -> {dst} (sha256 {digest})", file=sys.stderr, flush=True)
    bash = shutil.which("bash") or "/bin/bash"
    os.execv(bash, [bash, str(dst), *argv[1:]])
    return 0                                        # not reached


if __name__ == "__main__":
    sys.exit(main())
