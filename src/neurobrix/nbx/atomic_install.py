"""Installing a container into a cache that another machine may be reading.

The cache is frequently a mounted export shared by several machines. Two of them
installing the same model into it, or one installing while the other runs it, met
three separate races — all three written twice, once in `cli/commands/registry.py`
and once in `nbx/cache.py`:

1. **The staging name was shared.** `<name>.installing` carries no host and no pid,
   so two installers used the SAME directory, each extracting members into the
   other's tree, and each removing it on entry as "a previous import that died".
2. **The live tree was removed before the rename.** `rmtree(cache_path)` then
   `os.replace(staging, cache_path)` leaves the model absent for the whole duration
   of the rmtree — seconds for a small container, minutes for a sixty-gigabyte one —
   and destroys the tree a reader on the other machine is loading from.
3. **`NBXCache.extract` never staged at all**: it removed the live tree and then
   extracted IN PLACE at the final name. A `manifest.json` appears from the first
   member on, so any concurrent reader saw a half-written model as a complete one.
   That is exactly the defect `registry.py` had already been fixed for.

The fix is one brick with one contract, used by both call sites, and it is a door
rather than a check: the harmful states are made unreachable rather than detected
afterwards.

* **A per-model lock**, taken with `os.mkdir`, which is atomic on a local
  filesystem and on NFS (the server serialises MKDIR and answers EEXIST). A second
  installer does not wait and does not break it — it REFUSES, naming the host, the
  pid and the age of the holder.
* **A staging directory unique to this host and this pid**, so no installer can
  ever remove another's work, whatever it believes about it.
* **Two renames instead of a removal.** POSIX `rename(2)` refuses to replace a
  non-empty directory, which is why both call sites removed the live tree first.
  So the live tree is renamed ASIDE, the staging directory is renamed in, and the
  aside is removed afterwards. The window in which the model is absent shrinks from
  the length of a recursive delete to two syscalls, and a reader that already opened
  files under the old tree keeps reading them: POSIX keeps an open descriptor valid
  across a rename, and NFS does the same through its silly-rename.

Nothing here is model-specific, family-specific or vendor-specific: it is a
filesystem contract.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

#: Suffixes an installation leaves beside the final name. Model discovery must skip
#: every one of them: each can carry a `manifest.json` while being incomplete or
#: on its way out.
INSTALL_SUFFIXES = (".installing", ".lock", ".replaced")


class InstallHeldByAnother(RuntimeError):
    """Another installer holds this model's lock.

    Raised instead of waiting or breaking: an install that proceeds beside another
    install of the same model is the state this module exists to make unreachable.
    """


def is_install_artifact(name: str) -> bool:
    """True for the directories an installation leaves beside the final name.

    Discovery keys on `<dir>/manifest.json`, and a staging tree carries one from
    its first member on, so these names must never be offered as models. The test
    is on a SEGMENT, not on the end of the string: a staging directory is
    `<name>.installing.<host>.<pid>`, and an aside is
    `<name>.replaced.<host>.<pid>.<stamp>`.
    """
    return any(f"{suffix}." in name or name.endswith(suffix)
               for suffix in INSTALL_SUFFIXES)


def _tag() -> str:
    """The mark of this installer: a host and a pid, both needed.

    The host alone cannot distinguish two imports on one machine; the pid alone
    collides across machines, which is how two hosts came to share one staging
    directory.
    """
    return f"{socket.gethostname()}.{os.getpid()}"


def lock_path(cache_path: Path) -> Path:
    """The lock directory for a model, beside the model's own name."""
    cache_path = Path(cache_path)
    return cache_path.with_name(cache_path.name + ".lock")


def staging_path(cache_path: Path) -> Path:
    """Where THIS installer extracts: a name no other installer can produce."""
    cache_path = Path(cache_path)
    return cache_path.with_name(f"{cache_path.name}.installing.{_tag()}")


def _read_owner(lock: Path) -> dict:
    try:
        return json.loads((lock / "owner.json").read_text())
    except (OSError, ValueError):
        # A lock whose owner file is unreadable is still a lock. Saying "unknown"
        # is honest; inventing an owner to make the message tidy is not.
        return {}


def _is_our_dead_lock(owner: dict) -> bool:
    """True only for a lock this host left behind and whose process is gone.

    A lock held by ANOTHER host can never be judged stale from here — we cannot
    see that machine's process table, and guessing is how a live installation gets
    its tree deleted. So the answer for a foreign lock is always "no".
    """
    if owner.get("host") != socket.gethostname():
        return False
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False       # it exists and belongs to someone else
    return False


def _describe(owner: dict) -> str:
    host = owner.get("host", "an unknown host")
    pid = owner.get("pid", "?")
    started = owner.get("started")
    age = ""
    if isinstance(started, (int, float)):
        age = f", holding it for {int(time.time() - started)} s"
    return f"{host} (pid {pid}{age})"


#: `renameat2(2)` with `RENAME_EXCHANGE` swaps two paths in ONE syscall, so there is
#: no instant at which either name is missing. The two-rename form is correct but not
#: instantaneous, and its window is real rather than theoretical: the gate
#: `test_the_live_tree_is_readable_at_every_instant_of_a_reinstall` caught it on
#: 2026-09-18 with a single observation of `manifest.json absent` out of thousands of
#: polls on a loaded machine. One observation is enough — the claim is "at every instant".
_AT_FDCWD = -100
_RENAME_EXCHANGE = 2
_SYS_renameat2 = 316          # x86_64


def _exchange(a: Path, b: Path) -> bool:
    """Swap two existing paths atomically. False if the platform cannot.

    Returns False rather than raising on ENOSYS (a kernel older than the call),
    EINVAL/EOPNOTSUPP (a filesystem without the flag — several network filesystems
    included) and on any non-x86_64 build, so the caller keeps its portable path
    instead of an install failing on a machine whose kernel simply predates this.
    """
    import ctypes
    if os.uname().machine != "x86_64":
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return False
    ctypes.set_errno(0)
    rc = libc.syscall(ctypes.c_long(_SYS_renameat2),
                      ctypes.c_int(_AT_FDCWD), ctypes.c_char_p(str(a).encode()),
                      ctypes.c_int(_AT_FDCWD), ctypes.c_char_p(str(b).encode()),
                      ctypes.c_uint(_RENAME_EXCHANGE))
    if rc == 0:
        return True
    err = ctypes.get_errno()
    if err in (errno.ENOSYS, errno.EINVAL, errno.ENOTTY, errno.EOPNOTSUPP):
        return False
    raise OSError(err, os.strerror(err), str(a), None, str(b))


@contextmanager
def installing(cache_path, *, label: str = "install",
               allow_break: bool = False) -> Iterator[Path]:
    """Extract into a private staging tree, then swap it in atomically.

    Yields the staging directory to extract into. On a clean exit the staging tree
    replaces `cache_path` through two renames; on any exception the staging tree is
    removed and `cache_path` is left exactly as it was — an install that fails
    must not cost the copy that was already working.

    Args:
        cache_path: the final directory the model is to be visible at.
        label: what is installing, for the lock's owner record.
        allow_break: break a lock held by another host. The one deliberate
            opening, and it is never taken by the engine itself.

    Raises:
        InstallHeldByAnother: another installer holds this model's lock.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path(cache_path)

    try:
        os.mkdir(lock)
    except FileExistsError:
        owner = _read_owner(lock)
        if _is_our_dead_lock(owner):
            print(f"[install] clearing a lock left by pid {owner.get('pid')} on this "
                  f"host, whose process is gone: {lock}")
            shutil.rmtree(lock, ignore_errors=True)
            os.mkdir(lock)
        elif allow_break:
            print(f"[install] breaking the lock held by {_describe(owner)} "
                  f"(explicitly allowed): {lock}")
            shutil.rmtree(lock, ignore_errors=True)
            os.mkdir(lock)
        else:
            raise InstallHeldByAnother(
                f"{cache_path.name} is being installed by {_describe(owner)}.\n"
                f"   Two installs into one cache overwrite each other's files and can "
                f"destroy the copy a run on the other machine is reading.\n"
                f"   Wait for it to finish, or — only if you know that installer is "
                f"dead — remove {lock} by hand.")

    (lock / "owner.json").write_text(json.dumps({
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started": time.time(),
        "label": label,
        "target": str(cache_path),
    }, indent=2))

    staging = staging_path(cache_path)
    shutil.rmtree(staging, ignore_errors=True)   # ours alone: the tag says so
    staging.mkdir(parents=True)

    aside: Optional[Path] = None
    exchanged = False
    try:
        yield staging

        # The swap. ONE syscall where the kernel and filesystem allow it, so the
        # model is never absent for any instant; two renames otherwise, which is
        # the best a portable path can do. No recursive delete before either.
        if cache_path.exists() and _exchange(staging, cache_path):
            # `cache_path` now holds the new tree and `staging` the old one, and
            # no instant existed in which either name was missing. The old tree is
            # handed to the cleanup below under the name it now occupies.
            aside, exchanged = staging, True
        else:
            if cache_path.exists():
                aside = cache_path.with_name(
                    f"{cache_path.name}.replaced.{_tag()}.{int(time.time())}")
                os.replace(cache_path, aside)
            try:
                os.replace(staging, cache_path)
            except OSError as e:
                # Put the working copy back before surfacing: a failed install
                # must never be the reason a model that worked is gone.
                if aside is not None and not cache_path.exists():
                    os.replace(aside, cache_path)
                    aside = None
                if e.errno == errno.EXDEV:
                    raise OSError(
                        errno.EXDEV,
                        f"staging and cache are on different filesystems "
                        f"({staging} -> {cache_path}); the swap cannot be atomic. "
                        f"Point the staging directory at the cache's own filesystem."
                    ) from e
                raise
    except BaseException:
        # After an exchange, `staging` IS the model that is now live: removing it
        # here would delete what was just installed.
        if not exchanged:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)
        shutil.rmtree(lock, ignore_errors=True)
