"""`neurobrix autotune` — the certified autotune directory's tool.

    neurobrix autotune certify --profile <profile> [--vendor <vendor>] [--census PATH]
                               [--out DIR] [--kernels a,b] [--limit N] [--only-missing]
    neurobrix autotune check   [--dir DIR]          # the directory's gate, file by file
    neurobrix autotune status                       # what the profile in force would be served
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

print = functools.partial(print, flush=True)   # a run of hours is read while it runs


def _repo_root() -> Path:
    """The repository this file lives in: commands -> cli -> neurobrix -> src -> repo.

    Named rather than spelled inline because the door below is only as good as this number:
    one short and it looks for a checkpointer holding `src/`, finds none however many are
    running, and refuses every certification on the machine. It had that bug for five
    minutes on 2026-09-22."""
    return Path(__file__).resolve().parents[4]


def _checkpointer_holds(repo: Path) -> bool:
    """True when a `certified_checkpoint.py` process holds THIS repository.

    Read from /proc rather than through `pgrep`, so the check cannot match its own command
    line — the self-match that has cost this session three shells (exit 144)."""
    import os
    target = str(repo)
    if os.path.isdir("/proc"):
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as fh:
                    argv = fh.read().split(b"\0")
            except OSError:
                continue
            text = [a.decode("utf-8", "replace") for a in argv if a]
            if len(text) < 2 or "certified_checkpoint.py" not in " ".join(text[:2]):
                continue
            if target in text:
                return True
        return False
    # No /proc (macOS, the BSDs). `os.listdir("/proc")` used to raise here, and the certifier
    # died with "[Errno 2] No such file or directory: '/proc'" the moment a checkpointer was
    # finally holding the repository — the door could not see, so nothing could be certified
    # on this platform at all (2026-09-22).
    #
    # `ps` is the only process table there is here, so the self-match the /proc walk was
    # chosen to avoid is handled directly: THIS process's pid is skipped. Nothing else on the
    # machine carries `certified_checkpoint.py` in its first two arguments unless it IS one.
    import subprocess
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,args="], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    me = os.getpid()
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        if not head.isdigit() or int(head) == me:
            continue
        text = rest.split()
        if len(text) < 2 or "certified_checkpoint.py" not in " ".join(text[:2]):
            continue
        if target in text:
            return True
    return False


def _refuse_without_a_checkpointer(allowed: bool) -> str:
    """The refusal, or "" — a DOOR, not a report (`docs/reference/proving-by-doors.md`).

    The harmful STATE is a certification writing proofs into the directory while nothing
    carries them to a remote. On 2026-09-22 eleven hours of stage two — 2 497 entries across
    seven files — stood in the working tree alone with no checkpoint commit and no checkpoint
    ref on either remote, which is precisely the loss `tools/certified_checkpoint.py` was
    written against after the mains cuts of 2026-09-11, 09-12 and 09-13. It did not run
    because nothing STARTED it, and a brick that must be remembered is a brick that will be
    forgotten; so the certifier refuses instead, and the opening is named
    (`--allow-uncheckpointed`) rather than silent."""
    if allowed:
        return ""
    repo = _repo_root()
    if _checkpointer_holds(repo):
        return ""
    py = "python"
    return (
        "REFUSED: no certified checkpointer holds this repository, so a cut would cost this\n"
        "whole pass — the certifier writes its proofs entry by entry and nothing else carries\n"
        "them anywhere (2026-09-11, 09-12, 09-13, and again 09-22 with 2 497 entries).\n"
        "Start one and re-run:\n"
        f"  {py} tools/certified_checkpoint.py --repo {repo} \\\n"
        "      --dir src/neurobrix/config/autotune --interval 600 &\n"
        "Or say so deliberately with --allow-uncheckpointed (a sweep that certifies nothing,\n"
        "a laptop with no remote)."
    )


def cmd_autotune(args) -> int:
    from neurobrix.kernels import autotune_certified as C
    action = getattr(args, "action", None)
    if action == "certify":
        if not getattr(args, "profile", None):
            print("ERROR: --profile is required: a certification names the profile it was measured on.")
            return 2
        refusal = _refuse_without_a_checkpointer(getattr(args, "allow_uncheckpointed", False))
        if refusal:
            print(refusal)
            return 3
        from neurobrix.kernels.autotune_certify import certify
        kernels = [k for k in (args.kernels or "").split(",") if k] or None
        print("=" * 70)
        print(f"NeuroBrix autotune certify — profile {args.vendor + '/' if args.vendor else ''}{args.profile}")
        print("=" * 70)
        try:
            summary = certify(args.profile, vendor=args.vendor, census_path=args.census, out=args.out,
                              kernels=kernels, limit=args.limit, only_missing=args.only_missing,
                              reprove_unclocked=getattr(args, "reprove_unclocked", False),
                              reprove_generator=getattr(args, "reprove_generator", False),
                              allow_off_protocol=getattr(args, "allow_off_protocol_clock", False))
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(json.dumps({k: v for k, v in summary.items() if k != "started"}, indent=1))
        if summary.get("aborted"):
            print(f"ABORTED: the CUDA context died at {summary['aborted']['key']} — {summary['aborted']['reason'][:160]}; "
                  f"{summary['certified']} shape(s) certified before it stand, nothing after it was measured.")
            return 1
        # The gate, on what was just written: a file whose proof does not re-read is not left behind.
        bad = 0
        for path in C.files(Path(summary["directory"])):
            doc = json.loads(path.read_text(encoding="utf-8"))
            problems = C.validate(doc, path)
            if problems:
                bad += 1
                print(f"GATE: {path}: {problems[0]}")
        unreachable = summary.get("unreachable", 0)
        print(f"[certify] {summary['certified']} shape(s) certified, {summary['excluded_configs']} config(s) excluded, "
              f"{summary['failed']} failed, {unreachable} unreachable; "
              f"directory gate: {'every file re-reads' if not bad else f'{bad} file(s) refused'}")
        if unreachable:
            # Said in clear, and NOT folded into the exit code. The census
            # accumulates across engine versions and a key recorded under an
            # older rule can never be presented again, so refusing it is the
            # correct outcome, not a fault. Conflating the two made this command
            # exit 1 on every run of a healthy directory — a status that cries
            # wolf is a status nobody reads on the day it is right.
            print(f"[certify] {unreachable} census key(s) are unreachable to this engine — the debt "
                  f"D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE, not a failure. Each is named above "
                  f"with both keys, and nothing was certified for it.")
        return 0 if not bad and not summary["failed"] else 1
    if action == "check":
        root = Path(args.dir) if getattr(args, "dir", None) else C.directory()
        n = bad = 0
        _json = getattr(args, "json", False)
        _files = []
        for path in C.files(root):
            n += 1
            if getattr(args, "restamp", False):
                try:
                    changed = C.restamp(path)
                except (OSError, ValueError) as exc:
                    changed = None; print(f"restamp {path}: unreadable ({exc})")
                if changed:
                    print(f"restamp {path}: format claim repaired to {changed} (entries untouched)")
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                bad += 1; _files.append({"path": str(path), "ok": False, "problems": [f"unreadable ({exc})"]})
                if not _json: print(f"REFUSED {path}: unreadable ({exc})")
                continue
            problems = C.validate(doc, path)
            if problems:
                bad += 1; _files.append({"path": str(path), "ok": False, "problems": problems[:10]})
                if not _json: print(f"REFUSED {path}: " + "; ".join(problems[:3]))
            else:
                _files.append({"path": str(path), "ok": True, "shapes": len(doc.get("entries") or {})})
                if not _json: print(f"ok      {path} ({len(doc.get('entries') or {})} shape(s))")
        if _json:
            from neurobrix.cli.json_out import emit
            emit("autotune.check", {"directory": str(root), "files": _files, "refused": bad})
        else:
            print(f"{n} file(s), {bad} refused")
        return 0 if not bad else 1
    if action == "status":
        prof = C.active_profile()
        if getattr(args, "json", False):
            root = C.directory() / prof[0] / prof[1] if prof else None
            files = list(C.files()) if root else []
            mine = [p for p in files if root and p.parent == root]
            docs = [json.loads(p.read_text(encoding='utf-8')).get('entries') or {} for p in mine]
            by_class = {}; unknown = 0; by_backend = {}
            for entries in docs:
                for entry in entries.values():
                    classes = C.covered_memory_classes(entry)
                    if not classes: unknown += 1
                    for c in classes: by_class[c] = by_class.get(c, 0) + 1
                    for lab in (C.proof_backends(entry) or {"unknown"}): by_backend[lab] = by_backend.get(lab, 0) + 1
            here = C.executing_memory_class()
            from neurobrix.cli.json_out import emit
            emit("autotune.status", {"profile": f"{prof[0]}/{prof[1]}" if prof else None, "directory": str(C.directory()),
                                     "enabled": bool(C.enabled()), "files": len(mine), "shapes": sum(len(e) for e in docs),
                                     "served_by_memory_class_gb": {str(k): v for k, v in sorted(by_class.items())},
                                     "proven_on_unknown_card": unknown, "this_card_class_gb": here,
                                     "proofs_by_backend": dict(sorted(by_backend.items())),
                                     "would_be_served_here": by_class.get(here, 0) if here is not None else None})
            return 0
        print(f"profile in force: {prof[0] + '/' + prof[1] if prof else 'none resolved'}")
        print(f"directory: {C.directory()} ({'on' if C.enabled() else 'OFF (NBX_AUTOTUNE_CERTIFIED=off)'})")
        root = C.directory() / prof[0] / prof[1] if prof else None
        files = list(C.files()) if root else []
        mine = [p for p in files if root and p.parent == root]
        docs = [json.loads(p.read_text(encoding='utf-8')).get('entries') or {} for p in mine]
        total = sum(len(e) for e in docs)
        here = C.executing_memory_class()
        by_class = {}
        unknown = 0
        by_backend = {}
        for entries in docs:
            for entry in entries.values():
                classes = C.covered_memory_classes(entry)
                if not classes:
                    unknown += 1
                for c in classes:
                    by_class[c] = by_class.get(c, 0) + 1
                for lab in (C.proof_backends(entry) or {"unknown"}):
                    by_backend[lab] = by_backend.get(lab, 0) + 1
        print(f"files for this profile: {len(mine)}; shapes: {total}")
        print("proven under (a setting is proven for one code generator; a Triton upgrade re-proves): "
              + (", ".join(f"{k}: {n}" for k, n in sorted(by_backend.items())) or "none"))
        print("served by memory class (an entry serves only the class it was proven on): "
              + (", ".join(f"{c} GB: {n}" for c, n in sorted(by_class.items())) or "none")
              + f"; proven on an unknown card (served to no card until re-proven): {unknown}")
        print(f"this process's card class: {f'{here} GB' if here is not None else 'unknown (several classes visible, or no profile)'}"
              + (f" — {by_class.get(here, 0)} shape(s) would be served" if here is not None else ""))
        return 0
    print("usage: neurobrix autotune {certify,check,status} …")
    return 2
