"""The engine says when it is behind: at start, one short notice on stderr when the installed neurobrix is older than
the latest release on the package index, or its nbxtensor does not match the engine's own pin, with the command that
updates it for the way it was installed. It warns and never refuses.

Where and when: `neurobrix.cli.main`, once the arguments are read and the census shadow installed, before the
command. The notice is drawn from the cache on disk and the local package metadata only: no network on the command's
path. When the cache is a day old (or absent), the time of the attempt is written first (one attempt a day whatever
happens, offline included), then ONE detached child process (this interpreter, its own session, no stdin/stdout/stderr,
no handle kept) asks the index (PyPI's JSON, `cli.utils.PYPI_INDEX_URL`, the one source), writes the answer atomically
and ends by itself at TIMEOUT_S. The command never waits for it and it outlives the command: a command that ends at
once still gets the answer, shown by the next one. A failed or wrong answer is silence and a debug line, never a guess.
Nothing at all in a census shadow process, nor in a `serve` child another neurobrix launched (its launcher said it).
Cache: `<engine home>/update_check.json`, {"checked_at": unix seconds, "latest": {"neurobrix": "x.y.z", ...}}.
Switch: NEUROBRIX_NO_UPDATE_CHECK=1 — no call, no notice. Under --json no notice either (a machine reads it)."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("neurobrix.cli.update_check")

SWITCH = "NEUROBRIX_NO_UPDATE_CHECK"
CHECK_EVERY_S = 24 * 3600            # the owner's order (17/11): at most one network check a day
TIMEOUT_S = 3.0                      # the child's whole life: both answers, or it ends by itself
TEMP_GLOB = "update_check.json.*.tmp"  # write_cache's own temp names, the only files the sweep removes
ENGINE = "neurobrix"
LIBRARY = "nbxtensor"


def disabled() -> bool:
    return os.environ.get(SWITCH, "").strip() not in ("", "0")


def cache_path() -> Path:
    from neurobrix.core.paths import engine_home
    return engine_home() / "update_check.json"


def read_cache() -> dict:
    """The cache's record, or {} when it is absent or not the record this module writes (then a check is due)."""
    try:
        record = json.loads(cache_path().read_text())
    except (OSError, ValueError) as e:
        log.debug(f"update check: no usable cache ({e})")
        return {}
    ok = isinstance(record, dict) and isinstance(record.get("checked_at"), (int, float))
    return record if ok and isinstance(record.get("latest"), dict) else {}


def write_cache(record: dict) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as f:
        f.write(json.dumps(record))         # its own file per writer: the child and a command race safely
    os.replace(f.name, path)


def sweep_temps() -> None:
    """Remove the temp files a writer that died between its write and its os.replace left. Only this module's own temp
    names, and only those older than a child's whole life: a younger one may be a live writer's."""
    for temp in cache_path().parent.glob(TEMP_GLOB):
        try:
            if time.time() - temp.stat().st_mtime > 2 * TIMEOUT_S:
                temp.unlink()
        except OSError as e:                # another sweep took it first, or the home is read-only
            log.debug(f"update check: temp {temp.name} not removed ({e})")


def due(record: dict) -> bool:
    return not record or time.time() - record["checked_at"] >= CHECK_EVERY_S


def index_url() -> str:
    from neurobrix.cli.utils import PYPI_INDEX_URL
    return PYPI_INDEX_URL.rstrip("/")


def fetch_index(package: str, timeout: float = TIMEOUT_S):
    """The index's JSON record of one package (PyPI's `/pypi/<name>/json`); raises what the network raises."""
    import urllib.request
    with urllib.request.urlopen(f"{index_url()}/{package}/json", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _published(body):
    """The release number in an index record, or None when the record does not hold one."""
    from packaging.version import InvalidVersion, Version
    info = body.get("info") if isinstance(body, dict) else None
    text = info.get("version") if isinstance(info, dict) else None
    try:
        return str(Version(text)) if isinstance(text, str) else None
    except InvalidVersion:
        return None


def mark_attempt() -> dict:
    """The time of today's attempt, written before the ask: whatever becomes of the ask, the next is a day later."""
    record = read_cache() or {"latest": {}}
    record["checked_at"] = time.time()
    write_cache(record)
    return record


def refresh(fetch=fetch_index, deadline=None) -> None:
    """The ask (in the child): each package's latest release, each answer that is one kept, all within the deadline."""
    deadline = time.monotonic() + TIMEOUT_S if deadline is None else deadline
    try:
        record = read_cache() or mark_attempt()
        for package in (ENGINE, LIBRARY):
            left = deadline - time.monotonic()
            if left <= 0:
                log.debug(f"update check: no time left for {package}; nothing said")
                break
            try:
                latest = _published(fetch(package, left))
            except (OSError, ValueError) as e:      # offline, refused, timed out, not JSON
                latest = None
                log.debug(f"update check: the index did not answer for {package} ({e})")
            if latest is None:
                log.debug(f"update check: no release number for {package}; nothing said")
                continue
            record["latest"][package] = latest
            write_cache(record)
    except OSError as e:
        log.debug(f"update check: the cache could not be written ({e})")


# The child's program: this interpreter, isolated (-I: no current directory, no PYTHON* variable on its path), given the
# command's own module path so it imports this very neurobrix.
CHILD_CODE = ("import sys, json; sys.path[:] = json.loads(sys.argv[1]); "
              "from neurobrix.cli.update_check import child_main; child_main()")


def _start_refresh() -> None:
    """Start the detached child that asks the index; nothing of it is kept, waited for or read."""
    import subprocess
    detach = ({"creationflags": 0x00000008 | 0x00000200} if os.name == "nt"     # DETACHED_PROCESS | NEW_PROCESS_GROUP
              else {"start_new_session": True})
    subprocess.Popen([sys.executable, "-I", "-c", CHILD_CODE, json.dumps(sys.path)], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, **detach)


def child_main() -> None:
    """The detached child: the ask runs in a daemon thread waited for TIMEOUT_S at most, then the child exits whatever
    blocks (a name lookup has no timeout of its own); an abandoned daemon thread does not hold the exit."""
    import threading
    ask = threading.Thread(target=refresh, kwargs={"deadline": time.monotonic() + TIMEOUT_S}, daemon=True)
    ask.start()
    ask.join(TIMEOUT_S)


def installed_version(name: str):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def nbxtensor_requirement():
    """This neurobrix's requirement on nbxtensor, read from its installed metadata; None when it has none."""
    from importlib.metadata import PackageNotFoundError, requires
    from packaging.requirements import Requirement
    try:
        reqs = requires(ENGINE) or []
    except PackageNotFoundError:
        return None
    found = [Requirement(r) for r in reqs if Requirement(r).name == LIBRARY and not Requirement(r).marker]
    return found[0] if found else None


def library_commit():
    """The commit the installed nbxtensor was built from, as pip recorded it (`direct_url.json`): the commit of a git
    install, "" for a published wheel (no record: not any commit), None when it cannot say (a local tree)."""
    import json
    from importlib.metadata import PackageNotFoundError, distribution
    try:
        raw = distribution(LIBRARY).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not raw:
        return ""
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


def _install_kind():
    """How this neurobrix was installed, as far as its metadata and its environment say: (kind, source directory)."""
    from importlib.metadata import PackageNotFoundError, distribution
    try:
        dist = distribution(ENGINE)
    except PackageNotFoundError:
        return "pip", None
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable") and str(direct.get("url", "")).startswith("file://"):
        from urllib.parse import unquote, urlparse
        return "editable", unquote(urlparse(direct["url"]).path)
    parts = Path(sys.prefix).parts
    if "pipx" in parts and "venvs" in parts:
        return "pipx", None
    installer = (dist.read_text("INSTALLER") or "").strip()
    if installer == "uv":
        return ("uv-tool" if "uv" in parts and "tools" in parts else "uv"), None
    return "pip", None


def _quoted(text: str) -> str:
    return f'"{text}"' if " " in text else text


def update_command(package: str, spec: str = "") -> str:
    """The exact command that updates `package` (to `spec`, a requirement, when given) for this install."""
    kind, source = _install_kind()
    python = _quoted(sys.executable)
    target = _quoted(f"{package}{spec}") if spec else f"-U {package}"
    if kind == "editable" and package == ENGINE:
        return f"git -C {_quoted(source)} pull && {python} -m pip install -e {_quoted(source)}"
    if kind in ("pipx", "uv-tool") and not spec:
        return f"{'pipx' if kind == 'pipx' else 'uv tool'} upgrade {ENGINE}"
    if kind in ("pipx", "uv-tool"):
        return f"{'pipx inject' if kind == 'pipx' else 'uv tool install --with'} {_quoted(package + spec)} {ENGINE}"
    if kind == "uv":
        return f"uv pip install --python {python} {target}"
    return f"{python} -m pip install {target}"


def notice_lines(record=None) -> list:
    """What the person reads: the local pin check, then what the cached index answer says. [] = nothing to say."""
    from packaging.version import Version
    record = read_cache() if record is None else record
    latest = record.get("latest", {})
    lines = []
    have_engine, have_lib, pin = installed_version(ENGINE), installed_version(LIBRARY), nbxtensor_requirement()
    # between releases the engine pins its library by COMMIT (`nbxtensor @ git+<repository>@<sha40>`): a version says
    # nothing of it, the commit pip recorded does (supervisor 23/11 01:01)
    want = pin.url.rsplit("@", 1)[-1] if pin is not None and pin.url else None
    have_commit = library_commit() if want and have_lib else None
    if want and have_commit is not None and have_commit != want:
        lines.append(f"nbxtensor {have_lib} ({f'commit {have_commit[:12]}' if have_commit else 'a published release'}) "
                     f"is installed but neurobrix {have_engine} needs {LIBRARY} at commit {want[:12]}: "
                     f"{update_command(LIBRARY, f' @ {pin.url}')}")
    elif not want and pin and have_lib and not pin.specifier.contains(have_lib, prereleases=True):
        lines.append(f"nbxtensor {have_lib} is installed but neurobrix {have_engine} needs {LIBRARY}{pin.specifier}: "
                     f"{update_command(LIBRARY, str(pin.specifier))}")
    if have_engine and latest.get(ENGINE) and Version(latest[ENGINE]) > Version(have_engine):
        lines.append(f"neurobrix {latest[ENGINE]} is published (you have {have_engine}): {update_command(ENGINE)}")
    new_lib = latest.get(LIBRARY)
    if have_lib and new_lib and Version(new_lib) > Version(have_lib) and (pin is None or (not pin.url and pin.specifier.contains(new_lib))):
        lines.append(f"nbxtensor {new_lib} is published (you have {have_lib}): {update_command(LIBRARY)}")
    if lines:
        lines.append(f"({SWITCH}=1 turns this check off)")
    return lines


def silent_process(args) -> bool:
    """A process that must say and start nothing: a census shadow (kernels/census.py), or a `serve` child another
    neurobrix launched (its launcher passed where to signal ready, and says the notice itself)."""
    launched = getattr(args, "ready_fd", None) is not None or getattr(args, "ready_file", None) is not None
    return os.environ.get("NBX_CENSUS") == "1" or launched


def warn(args) -> list:
    """The start-of-command check: the notice from the cache and the metadata on stderr, the detached ask when the
    cache is a day old. Never raises, never waits on the network. Returns the lines said."""
    if disabled() or silent_process(args):
        return []
    try:
        sweep_temps()
        record = read_cache()
        if due(record):
            record = mark_attempt()
            _start_refresh()
        lines = [] if getattr(args, "json", False) else notice_lines(record)
    except Exception as e:      # advice only: a defect here must never stop a command
        log.debug(f"update check skipped: {type(e).__name__}: {e}")
        return []
    for line in lines:
        print(line, file=sys.stderr)
    return lines
