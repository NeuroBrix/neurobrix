"""`neurobrix drift` — the drift-site detector on one request: the same
request on the ATen oracle (sequential, op by op) and on the Triton engine,
both writing their per-op record (NBX_DUMP_TIDS), then the first op in the
oracle's order whose window departs beyond the bound.

Each engine runs in its own process (the ATen branch loads torch; the
Triton branch must not), through the same CLI as a user's request. The
report is written beside the dumps (`<out>/drift.json`, `drift.txt`)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _passthrough(args, exclude: set) -> list:
    """The request arguments as they were given, for the two child runs."""
    out = []
    argv = list(sys.argv[1:])
    # drop the subcommand and the detector's own options
    if argv and argv[0] == "drift":
        argv = argv[1:]
    skip = 0
    for i, a in enumerate(argv):
        if skip:
            skip -= 1
            continue
        key = a.split("=", 1)[0]
        if key in exclude:
            if "=" not in a and key not in ("--sweep",):
                skip = 1
            continue
        out.append(a)
    return out


def cmd_drift(args) -> int:
    from neurobrix.core.dtype import drift as D
    if args.model is None:
        print("ERROR: --model is required for drift.")
        return 2
    out = Path(args.out or (Path.home() / ".neurobrix" / "drift" / args.model))
    out.mkdir(parents=True, exist_ok=True)
    request = _passthrough(args, {"--out", "--bound", "--top", "--oracle", "--sequential", "--triton",
                                  "--triton-sequential", "--compiled", "--output", "--sweep"})
    nbx = [sys.executable, "-m", "neurobrix"] if os.environ.get("NBX_DRIFT_MODULE") else [sys.argv[0]]
    if not Path(nbx[0]).exists() or nbx[0].endswith("__main__.py"):
        nbx = [sys.executable, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())"]
    arms = (("oracle", [f"--{args.oracle}"], out / "oracle.jsonl"),
            ("engine", ["--triton"] + (["--sweep"] if getattr(args, "sweep", False) else []), out / "engine.jsonl"))
    print("=" * 70)
    print(f"NeuroBrix Drift — {args.model}: the ATen oracle ({args.oracle}) against the Triton engine, per op")
    print(f"   request: {' '.join(request)}")
    print(f"   dumps and report: {out}")
    print("=" * 70)
    for name, flags, dump in arms:
        if dump.exists():
            dump.unlink()
        env = {**os.environ, "NBX_DUMP_TIDS": str(dump)}
        output = out / f"{name}_output"
        cmd = nbx + ["run", "--model", args.model] + request + flags + ["--output", str(output)]
        log = out / f"{name}.log"
        with open(log, "w") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[drift] {name} arm exited {rc}; dump {dump} ({dump.stat().st_size if dump.exists() else 0} bytes) — log {log}")
        if rc:
            print(f"[drift] the {name} arm failed — no site named")
            return int(rc)
    report = D.detect(arms[0][2], arms[1][2], bound=args.bound, top=args.top)
    text = D.describe(report)
    print(text)
    (out / "drift.txt").write_text(text + "\n")
    (out / "drift.json").write_text(json.dumps({"model": args.model, "oracle": args.oracle, "request": request,
                                                **report.to_dict()}, indent=1))
    return 0
