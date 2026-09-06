"""`neurobrix autotune` — the certified autotune directory's tool.

    neurobrix autotune certify --profile <profile> [--vendor <vendor>] [--census PATH]
                               [--out DIR] [--kernels a,b] [--limit N] [--only-missing]
    neurobrix autotune check   [--dir DIR]          # the directory's gate, file by file
    neurobrix autotune status                       # what the profile in force would be served
"""
from __future__ import annotations

import json
from pathlib import Path


def cmd_autotune(args) -> int:
    from neurobrix.kernels import autotune_certified as C
    action = getattr(args, "action", None)
    if action == "certify":
        if not getattr(args, "profile", None):
            print("ERROR: --profile is required: a certification names the profile it was measured on.")
            return 2
        from neurobrix.kernels.autotune_certify import certify
        kernels = [k for k in (args.kernels or "").split(",") if k] or None
        print("=" * 70)
        print(f"NeuroBrix autotune certify — profile {args.vendor + '/' if args.vendor else ''}{args.profile}")
        print("=" * 70)
        try:
            summary = certify(args.profile, vendor=args.vendor, census_path=args.census, out=args.out,
                              kernels=kernels, limit=args.limit, only_missing=args.only_missing)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(json.dumps({k: v for k, v in summary.items() if k != "started"}, indent=1))
        # The gate, on what was just written: a file whose proof does not re-read is not left behind.
        bad = 0
        for path in C.files(Path(summary["directory"])):
            doc = json.loads(path.read_text(encoding="utf-8"))
            problems = C.validate(doc, path)
            if problems:
                bad += 1
                print(f"GATE: {path}: {problems[0]}")
        print(f"[certify] {summary['certified']} shape(s) certified, {summary['excluded_configs']} config(s) excluded, "
              f"{summary['failed']} failed; directory gate: {'every file re-reads' if not bad else f'{bad} file(s) refused'}")
        return 0 if not bad and not summary["failed"] else 1
    if action == "check":
        root = Path(args.dir) if getattr(args, "dir", None) else C.directory()
        n = bad = 0
        for path in C.files(root):
            n += 1
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                bad += 1; print(f"REFUSED {path}: unreadable ({exc})"); continue
            problems = C.validate(doc, path)
            if problems:
                bad += 1; print(f"REFUSED {path}: " + "; ".join(problems[:3]))
            else:
                print(f"ok      {path} ({len(doc.get('entries') or {})} shape(s))")
        print(f"{n} file(s), {bad} refused")
        return 0 if not bad else 1
    if action == "status":
        prof = C.active_profile()
        print(f"profile in force: {prof[0] + '/' + prof[1] if prof else 'none resolved'}")
        print(f"directory: {C.directory()} ({'on' if C.enabled() else 'OFF (NBX_AUTOTUNE_CERTIFIED=off)'})")
        root = C.directory() / prof[0] / prof[1] if prof else None
        files = list(C.files()) if root else []
        mine = [p for p in files if root and p.parent == root]
        print(f"files for this profile: {len(mine)}; shapes: "
              f"{sum(len((json.loads(p.read_text(encoding='utf-8')).get('entries') or {})) for p in mine)}")
        return 0
    print("usage: neurobrix autotune {certify,check,status} …")
    return 2
