#!/usr/bin/env python3
"""Refuse to measure on an environment that has silently degraded.

Written 2026-09-10 after six test failures and one collection error were
traced not to the code under test but to an external clone that the system's
periodic /private/tmp cleaner had half-deleted: 245 tracked files gone,
`triton_msl/__init__.py` among them, the directory still present so nothing
looked wrong. The measurement said "nine pre-existing failures". There were
three.

Run this before any suite or campaign whose result will be written down.
Exit 0 = the environment is what the measurement assumes. Non-zero = say so
and stop; a degraded environment does not produce a weaker number, it
produces a wrong one.
"""
from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


def _editable_targets(module: str) -> list[Path]:
    """Directories an editable install points at, read from its finder."""
    spec = importlib.util.find_spec(f"__editable___{module}_0_2_0_finder")
    if spec is None or spec.origin is None:
        return []
    ns: dict = {}
    exec(compile(Path(spec.origin).read_text(), spec.origin, "exec"), ns)
    return [Path(p) for p in (ns.get("MAPPING") or {}).values()]


def check_importable(module: str) -> list[str]:
    try:
        __import__(module)
        return []
    except Exception as exc:
        targets = _editable_targets(module)
        detail = "".join(
            f"\n    {t} : {'present' if t.exists() else 'GONE'}"
            f"{'' if (t / '__init__.py').exists() else '  (no __init__.py)'}"
            for t in targets)
        return [f"{module} does not import: {exc}{detail}"]


def check_worktree_intact(path: Path) -> list[str]:
    """A checkout with files deleted underneath it is not a checkout."""
    if not (path / ".git").exists():
        return [f"{path} is not a git checkout"]
    out = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    deleted = [l[3:] for l in out.splitlines() if l.startswith(" D")]
    if not deleted:
        return []
    return [f"{path}: {len(deleted)} tracked files deleted from the working "
            f"tree (e.g. {', '.join(deleted[:3])}). "
            f"Restore with: git -C {path} checkout -- ."]


# Directory prefixes the operating system may clear without asking. macOS
# cleans /private/tmp by ACCESS time on a periodic schedule, and /var/folders
# is the per-user temporary tree with the same contract.
_EPHEMERAL_PREFIXES = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp",
                       "/var/folders", "/private/var/folders")


#: `mount` names the filesystem type differently per platform, and the first
#: version of this parser matched neither: macOS ends the line with `(nfs)`
#: and it looked for `" nfs "`, so it found nothing and the guard passed in
#: silence — a detector that finds nothing is worse than no detector, which is
#: why this is a named function with a test rather than a line inside one.
_NFS_LINE = re.compile(
    r"^(?P<host>\d{1,3}(?:\.\d{1,3}){3}):\S*\s+on\s+\S+.*?"
    r"(?:\((?:[^)]*,)?nfs[,)]|type\s+nfs\b)", re.IGNORECASE)


def parse_nfs_servers(mount_output: str) -> list[str]:
    """The NFS servers named in `mount` output, in order, without repeats."""
    out: list[str] = []
    for line in mount_output.splitlines():
        m = _NFS_LINE.match(line.strip())
        if m and m.group("host") not in out:
            out.append(m.group("host"))
    return out


def _nfs_servers_and_local_addresses() -> tuple[list[str], list[str]]:
    """(servers this machine has mounted, addresses its interfaces hold).

    The expected network is not written here. It is READ from the mount table:
    whatever NFS servers this machine has actually mounted are the network it
    belongs to, so the day the lab moves, this moves with it and nobody has to
    remember to edit a constant.
    """
    servers: list[str] = []
    mounts = subprocess.run(["mount"], capture_output=True, text=True)
    servers.extend(parse_nfs_servers(mounts.stdout))

    local: list[str] = []
    ifc = subprocess.run(["ifconfig"], capture_output=True, text=True)
    for line in ifc.stdout.splitlines():
        m = re.search(r"^\s+inet (\d{1,3}(?:\.\d{1,3}){3})", line)
        if m and not m.group(1).startswith("127."):
            local.append(m.group(1))
    return servers, local


def check_on_the_measurement_network() -> list[str]:
    """A campaign that starts off the lab network reads nothing and refuses.

    This cost a whole model on 2026-09-10: the machine had joined
    192.168.1.0/24 instead of the lab's 10.0.0.0/24, so traffic to the file
    server went to the internet gateway and a 12 GB stage died at 2 GB. It was
    not an export failure and not NFS — it was the machine on the wrong
    network, and it was found AFTER the copy, from its corpse. A machine that
    joins the wrong network once will join it again, so this is a check rather
    than a habit.

    The test is SUBNET MEMBERSHIP, not reachability, and that choice was
    forced by measurement. The first version connected to each server's nfsd
    with a 2 s timeout; run while a 12 GB stage was saturating the link, every
    connect timed out and the guard refused a machine that was demonstrably on
    the right network — the ports were open and the mounts readable seconds
    later. A guard that refuses because a copy is in flight is worse than no
    guard. Subnet membership answers the question that was actually asked and
    cannot be perturbed by load; reachability is kept as a warning, where a
    transient costs nothing.

    Nothing about the lab is written here: the expected network is READ from
    the mount table, so the day it moves, this moves with it.
    """
    servers, local = _nfs_servers_and_local_addresses()
    if not servers:
        return []                       # nothing mounted: nothing to be off

    def net24(addr: str) -> str:
        return addr.rsplit(".", 1)[0]

    local_nets = {net24(a) for a in local}
    off = [h for h in servers if net24(h) not in local_nets]
    if not off:
        return []
    return [f"not on the measurement network: this machine holds "
            f"{', '.join(local) or 'no routable address'}, and none of them is "
            f"on the network of the server(s) it has mounted "
            f"({', '.join(f'{h} (expected {net24(h)}.0/24)' for h in off)}). "
            f"A campaign that starts here reads nothing — check which network "
            f"the machine joined before blaming the exports."]


def warn_servers_answer() -> list[str]:
    """The mounted servers answer on nfsd. A WARNING, never a refusal: a
    server can be busy or briefly unreachable while the machine is perfectly
    on the right network, and that is the transient the subnet check above
    exists to not confuse with a real one."""
    servers, _ = _nfs_servers_and_local_addresses()
    silent = []
    for host in servers:
        for _ in range(2):              # one retry: a saturated link is not an outage
            sock = socket.socket()
            sock.settimeout(6.0)
            try:
                sock.connect((host, 2049))      # nfsd
                break
            except OSError:
                continue
            finally:
                sock.close()
        else:
            silent.append(host)
    if not silent:
        return []
    return [f"mounted server(s) not answering on nfsd right now: "
            f"{', '.join(silent)} — the machine is on their network, so this "
            f"is a busy or absent server rather than a wrong subnet"]


def check_package_is_durable(module: str) -> list[str]:
    """Where the measured object actually LIVES, read from the import itself.

    Not from the install metadata and not from a configured path: from
    `module.__file__` after import, because that is the file the measurement
    executes. A package on a volatile path is not a slower measurement, it is
    a measurement whose object cannot be guaranteed — the same object may not
    be there on the next run, and was not necessarily whole on the last one.
    """
    import importlib, os
    try:
        m = importlib.import_module(module)
    except Exception:
        return []          # the import check above already reported this
    path = os.path.realpath(getattr(m, "__file__", "") or "")
    if not path:
        return [f"{module} imports but names no file; its location cannot be "
                f"established."]
    if path.startswith(_EPHEMERAL_PREFIXES):
        return [f"{module} is imported from {path}, which is on a path the "
                f"system clears ({', '.join(_EPHEMERAL_PREFIXES)}). Every "
                f"number measured against it is a number whose object may "
                f"already have changed. Move the checkout somewhere durable "
                f"and reinstall the venv from there."]
    return []


def check_profile_matches_hardware() -> tuple[list[str], list[str]]:
    """A tightened budget changes which STRATEGY runs, so it changes what is
    measured — silently, because these profiles are gitignored and `git
    status` never mentions them.

    Measured 2026-09-10: a probe left `memory_mb` at 1000 after being killed
    by memory pressure. Every run afterwards chose `layer_streaming` instead
    of `single_gpu`, including four cold runs on which a conclusion was
    written and a whole campaign that was supposed to measure the ordinary
    path. Nothing in the environment said so.

    Tightening is a legitimate technique — it is how the layer rung is
    exercised. So this does not forbid it: it requires that it be DECLARED,
    with NBX_PROFILE_TIGHTENED=1, and then says so on every run.
    """
    import glob, os, re
    problems: list[str] = []
    notes: list[str] = []
    try:
        import Metal
        dev = Metal.MTLCreateSystemDefaultDevice()
        hw_mb = int(dev.recommendedMaxWorkingSetSize()) // (1024 * 1024)
    except Exception:
        return [], []          # not an Apple device: nothing to compare against
    declared = os.environ.get("NBX_PROFILE_TIGHTENED") == "1"
    for f in glob.glob("src/neurobrix/config/hardware/default-*.yml"):
        m = re.search(r"^\s*memory_mb:\s*(\d+)", Path(f).read_text(), re.M)
        if not m:
            continue
        prof_mb = int(m.group(1))
        if prof_mb == hw_mb:
            continue
        line = (f"{f}: memory_mb is {prof_mb} where the device reports "
                f"{hw_mb}. A budget below the device's changes which Prism "
                f"strategy runs, so it changes what is measured.")
        if declared:
            notes.append(line + " (declared: NBX_PROFILE_TIGHTENED=1)")
        else:
            problems.append(line + " Set NBX_PROFILE_TIGHTENED=1 if this is "
                                   "deliberate, or restore the detected value.")
    return problems, notes


def check_worktrees_are_durable() -> list[str]:
    """No git worktree may live on a path the system clears.

    Three did — `converge-mac`, `before-levers`, `mfl-before` — under
    /private/tmp beside a 1.6 GB scratchpad. Their commits were reachable
    from remotes, so no history was at risk, but the measurement TREES would
    have vanished at the next sweep and the work would have restarted. The
    same cleaner had already eaten 245 tracked files and loose git objects
    from an external clone.
    """
    out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                         capture_output=True, text=True)
    bad = []
    for line in out.stdout.splitlines():
        if line.startswith("worktree "):
            path = line.split(" ", 1)[1].strip()
            if os.path.realpath(path).startswith(_EPHEMERAL_PREFIXES):
                bad.append(path)
    if not bad:
        return []
    return [f"{len(bad)} git worktree(s) on a path the system clears: "
            f"{', '.join(bad)}. Remove them with `git worktree remove` — never "
            f"rm — and create them under a durable root."]


def check_output_dirs_are_durable(paths) -> list[str]:
    """A campaign's output directory is not scratch: it is the measurement."""
    bad = [p for p in paths
           if os.path.realpath(p).startswith(_EPHEMERAL_PREFIXES)]
    if not bad:
        return []
    return [f"output directory on a path the system clears: {', '.join(bad)}. "
            f"A campaign writes where its results survive it."]


def check_object_store(path: Path) -> list[str]:
    """A checkout whose history is unreadable is eroding, not merely dirty.

    The working-tree check above missed this: `git checkout -- .` restored
    every file, and `git log -S` still answered `fatal: unable to read tree`,
    because the cleaner had eaten loose objects too. A measurement can be
    sound on such a repo (the tree at HEAD is complete) while every question
    about *when* something changed is unanswerable — and the next thing the
    cleaner eats may be the tree itself.
    """
    if not (path / ".git").exists():
        return []
    out = subprocess.run(["git", "-C", str(path), "fsck", "--no-progress",
                          "--connectivity-only"],
                         capture_output=True, text=True, timeout=300)
    bad = [l for l in (out.stdout + out.stderr).splitlines()
           if l.startswith(("missing ", "broken link")) or "unable to read" in l]
    if not bad:
        return []
    return [f"{path}: object store is incomplete ({len(bad)} problems, "
            f"e.g. {bad[0]}). History is partly unreadable; the tree at HEAD "
            f"may still be sound, but this repo is being eroded."]


def check_branch_is_recoverable(path: Path) -> list[str]:
    """Work that exists only here is one cleaner pass from gone."""
    if not (path / ".git").exists():
        return []
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(path), "rev-parse",
                             "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    remotes = subprocess.run(["git", "-C", str(path), "remote"],
                             capture_output=True, text=True).stdout.split()
    for r in remotes:
        ls = subprocess.run(["git", "-C", str(path), "ls-remote", "--heads",
                             r], capture_output=True, text=True, timeout=120)
        if head in ls.stdout:
            return []
    return [f"{path}: HEAD {head[:12]} ({branch}) is on no remote. If this "
            f"checkout lives under a directory the system cleans, the work "
            f"exists in exactly one place."]


def main() -> int:
    # Two classes, and the line between them is the same one this repo draws
    # everywhere else: REFUSE where the measurement would be wrong, SAY IT
    # LOUDLY where it would not.
    #
    #   * a deleted working-tree file changes what is compiled -> refuse;
    #   * an incomplete object store leaves the tree at HEAD sound, so the
    #     measurement stands — but the repo is eroding and every question
    #     about when something changed is already unanswerable -> warn.
    problems: list[str] = []
    warnings: list[str] = []
    problems += check_importable("triton_msl")
    problems += check_package_is_durable("triton_msl")
    problems += check_on_the_measurement_network()
    problems += check_worktrees_are_durable()
    problems += check_output_dirs_are_durable(sys.argv[1:])
    _prof_problems, _prof_notes = check_profile_matches_hardware()
    problems += _prof_problems
    warnings += _prof_notes
    warnings += warn_servers_answer()
    for target in _editable_targets("triton_msl"):
        # the clone root is the parent of the package directory
        clone = target.parent
        problems += check_worktree_intact(clone)
        warnings += check_object_store(clone)
        warnings += check_branch_is_recoverable(clone)
        break

    for w in warnings:
        print(f"  ! {w}")
    if problems:
        print("REFUSING TO MEASURE — the environment has degraded:")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("environment sound for measuring: working tree complete, "
          "triton_msl imports, on the measurement network, no worktree "
          "or output dir on a cleared path"
          + (f" ({len(warnings)} warning(s) above)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
