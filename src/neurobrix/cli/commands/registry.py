"""
neurobrix import/list/remove/clean/hub — Registry and model management commands.

All commands that manage ~/.neurobrix/store/ and ~/.neurobrix/cache/.
"""

import os
import sys
import json
import re

from neurobrix import __version__
from neurobrix.cli.utils import (
    STORE_DIR, CACHE_DIR, REGISTRY_URL, format_size,
)


# Headroom kept free after an import, so a disk is never handed back at 100 %.
IMPORT_DISK_MARGIN = 2 * 1024 ** 3


def _nearest_existing(path):
    """The closest existing ancestor — `disk_usage` needs a path that exists,
    and the store or cache may not have been created yet."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return str(p)


def import_peak_bytes(archive_bytes: int, keep: bool = True) -> int:
    """Bytes the disk must hold at the WORST moment of an import.

    The archive is written to the store, then extracted to the cache, and only
    deleted afterwards. So the peak is store + cache — about twice the archive —
    and `--no-keep` does not lower it: that flag frees space AFTER the
    extraction, not during it. Reporting the archive's size as the requirement
    would understate the moment that actually fails by half.
    """
    return int(archive_bytes) * 2


def disk_refusal(needed_bytes, free_bytes):
    """Why this import must not start, or None if it may.

    A user lost a 30.6 GB import at 98 % on a full disk: nothing had checked.
    The check goes BEFORE, and it declares BOTH numbers — what it needs and what
    there is — because a refusal that names only one of them cannot be acted on.

    An unknown size refuses nothing: the hub does not always publish one, and
    blocking on a number nobody has would stop imports that would have worked.
    Same rule as every other guard here.

    This is a check, NOT a resume. An import that dies at 98 % for another
    reason still loses everything; that is D-IMPORT-RESUMABLE-DOWNLOAD and it is
    a separate piece of work.
    """
    if not needed_bytes:
        return None
    needed, free = int(needed_bytes), int(free_bytes)
    if free >= needed + IMPORT_DISK_MARGIN:
        return None
    return (f"this import needs about {format_size(needed)} at its peak "
            f"(the archive is extracted before it is deleted, so the store and "
            f"the cache hold it at once) plus a {format_size(IMPORT_DISK_MARGIN)} "
            f"margin, and the disk has {format_size(free)} free")


# The mode an extracted container carries, whoever imported it.
CONTAINER_FILE_MODE = 0o644


class IncompleteDownload(RuntimeError):
    """The stream ended before the announced size — kept as a partial, never renamed."""


def partial_path(dest):
    """Where an unfinished download lives: beside its destination, named so that
    nothing reads it as a container (`.nbx` is what `import` extracts)."""
    from pathlib import Path
    dest = Path(dest)
    return dest.with_name(dest.name + ".part")


def download_resumable(url, dest, total_hint=0, desc="", get=None, progress=None, chunk_size=1 << 20):
    """Download `url` to `dest`, resuming a partial left by an earlier attempt.

    D-IMPORT-RESUMABLE-DOWNLOAD (2026-09-13). A 127 GB container took 86 minutes to
    move on this rack's own network, and an import that died at 98 % lost all of
    it: the stream was written straight to its final name and unlinked on any
    error. Now:

      * bytes go to `<dest>.part`; the final name appears only when the size the
        server announced has been reached (or, when it announced none, when the
        stream ended cleanly) — so a file named `.nbx` is never a fragment;
      * a `.part` found at entry is resumed with `Range: bytes=<have>-`; a server
        that answers 206 with a matching `Content-Range` gets appended to, one
        that answers 200 (no range support) is read from zero and the partial
        replaced — never appended to, which would splice two streams;
      * a stream that ends short of the announced size raises
        IncompleteDownload and KEEPS the partial, so the next run resumes;
      * a partial larger than the announced size is not trusted: it is dropped
        and the download restarts (a stale partial from another version of the
        file cannot be told from a good one by its length).

    `get` and `progress` are injected so the brick is testable against a local
    server and quiet in tests; the import command passes `requests.get` and
    `tqdm`. Returns the final size in bytes.
    """
    import os
    from pathlib import Path
    if get is None:
        import requests
        get = requests.get
    dest = Path(dest)
    part = partial_path(dest)
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    resp = get(url, stream=True, timeout=30, headers=headers)
    resp.raise_for_status()
    total = 0
    mode = "wb"
    if have and resp.status_code == 206:
        content_range = resp.headers.get("Content-Range", "")
        # bytes <start>-<end>/<total>; the start must be exactly what we have
        try:
            spec = content_range.split(" ", 1)[1]
            start = int(spec.split("-", 1)[0])
            total = int(spec.rsplit("/", 1)[1]) if "/" in spec and not spec.endswith("*") else 0
        except (IndexError, ValueError):
            start, total = -1, 0
        if start != have:
            raise IncompleteDownload(f"server resumed at byte {start}, partial holds {have}")
        mode = "ab"
    else:
        # 200 — the server ignored the range (or there was none): start over.
        have = 0
        total = int(resp.headers.get("content-length", 0) or 0) or int(total_hint or 0)
    if total and have > total:
        have, mode = 0, "wb"
    bar = progress(total=total or None, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
                   desc=desc) if progress else None
    written = have
    try:
        with open(part, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if bar:
                    bar.update(len(chunk))
    except Exception as exc:
        # Whatever broke the stream (a reset, a chunked-encoding tear, a timeout)
        # is one fact to the caller: the bytes so far are on disk and the next
        # run resumes. The original is chained for the log.
        raise IncompleteDownload(f"stream broke at {written} bytes: {exc}") from exc
    finally:
        if bar:
            bar.close()
    if total and written < total:
        raise IncompleteDownload(f"stream ended at {written} of {total} bytes")
    if total and written > total:
        part.unlink()
        raise IncompleteDownload(f"stream delivered {written} bytes for an announced {total}: not a resume")
    os.replace(part, dest)
    return dest.stat().st_size
CONTAINER_DIR_MODE = 0o755


def is_installed_model_dir(d) -> bool:
    """A cache entry that counts as a model: a directory with a manifest that is
    none of the things an install leaves beside the final name — a staging tree,
    a lock, or a tree renamed aside during the swap. Each of those can carry a
    `manifest.json` while being incomplete or on its way out, so the test is
    `is_install_artifact`, never an equality against one suffix."""
    from pathlib import Path
    from neurobrix.nbx.atomic_install import is_install_artifact
    d = Path(d)
    return d.is_dir() and not is_install_artifact(d.name) and (d / "manifest.json").exists()


def installing_path(cache_path) -> "Path":
    """Where an import extracts before the final name exists, beside it.

    The name carries THIS host and THIS pid — `<name>.installing.<host>.<pid>` —
    because the old shared `<name>.installing` let two machines extract into one
    directory and each remove the other's tree as "a previous import that died".
    Model discovery keys on `<dir>/manifest.json`, and a staging tree carries one
    from its first member on, so the suffix is also the mark every reader skips.
    """
    from neurobrix.nbx.atomic_install import staging_path
    return staging_path(cache_path)


def extract_container(store_path, cache_path):
    """Extract a `.nbx` and give the result a mode that does not depend on the
    importer's environment.

    `zipfile.extractall` does NOT apply the permission bits stored in the
    archive: every file takes the umask of whatever process ran the import.
    Measured on this machine — two models imported in May carried their whole
    content owner-only (manifest, profile, weights index, topology, twelve
    files) while a third imported in August was world-readable. Same engine,
    same archives, different shell. In an engine that sells determinism, the
    state on disk of an artefact must not depend on who unpacked it.

    The member path check that was already here is kept and runs FIRST: a member
    resolving outside the cache directory is refused before anything is written.
    """
    import zipfile

    cache_path = Path(cache_path)
    root = os.path.realpath(str(cache_path))
    with zipfile.ZipFile(store_path, "r") as zf:
        for member in zf.namelist():
            resolved = os.path.realpath(os.path.join(str(cache_path), member))
            if not resolved.startswith(root + os.sep) and resolved != root:
                raise ValueError(
                    f"Security: path traversal detected in archive member: {member}")
        cache_path.mkdir(parents=True, exist_ok=True)
        zf.extractall(str(cache_path))

    # umask applies at creation, so the mode is set afterwards — on the tree as
    # it now stands, directories included.
    for dirpath, dirnames, filenames in os.walk(str(cache_path)):
        os.chmod(dirpath, CONTAINER_DIR_MODE)
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), CONTAINER_FILE_MODE)
    os.chmod(str(cache_path), CONTAINER_DIR_MODE)


def _ev(args, name: str, **fields) -> None:
    """One lifecycle event of `import` under --json (nothing otherwise)."""
    from neurobrix.cli.json_out import wants_json, event
    if wants_json(args):
        event("import", name, fields)


def _die(args, *lines: str, code: int = 1):
    """Refuse: the human lines (stderr under --json), one `error` event
    carrying the first line, then exit. Every refusal of `import` ends here
    so a client always reads a terminal event."""
    for line in lines:
        print(line)
    _ev(args, "error", message=lines[0] if lines else "refused")
    sys.exit(code)


def cmd_import(args):
    """Download model from NeuroBrix registry and extract to local cache.
    Under --json: one NDJSON event per phase on stdout (`info`, `license`,
    `download` with real byte counts, `downloaded`, `extracting`, `installed`,
    `done` — or `error`), every human line on stderr (Studio requests 5, 6)."""
    from neurobrix.cli.json_out import wants_json, human_lines_to_stderr
    with human_lines_to_stderr(wants_json(args)):
        _import_body(args)


def _import_body(args):
    import requests
    from tqdm import tqdm
    from neurobrix.cli.json_out import wants_json, NdjsonProgress

    registry = args.registry or REGISTRY_URL
    model_ref = args.model_ref

    if "/" not in model_ref:
        _die(args, f"ERROR: Invalid model reference '{model_ref}'",
             "Expected format: org/name (e.g., pixart/sigma-xl-1024)")

    org, name = model_ref.split("/", 1)

    print("=" * 70)
    print("NeuroBrix Import")
    print("=" * 70)
    print(f"Model: {org}/{name}")
    print(f"Registry: {registry}")

    # Check if already cached
    cache_path = CACHE_DIR / name
    if cache_path.exists() and (cache_path / "manifest.json").exists() and not args.force:
        print(f"\nModel already installed: {cache_path}")
        print("Use --force to re-download.")
        _ev(args, "installed", model=f"{org}/{name}", cache=str(cache_path), already=True)
        _ev(args, "done", model=f"{org}/{name}", cache=str(cache_path))
        sys.exit(0)

    # 1. Get model metadata
    print(f"\n[1/4] Fetching model info...")
    try:
        resp = requests.get(f"{registry}/api/models/{org}/{name}", timeout=10)
        resp.raise_for_status()
        model_info = resp.json()
    except requests.ConnectionError:
        _die(args, f"ERROR: Cannot connect to registry at {registry}",
             "Check your network connection or use --registry to specify a different URL.")
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        if status == 404:
            _die(args, f"ERROR: Model '{org}/{name}' not found on registry.",
                 f"Browse available models: {registry}")
        _die(args, f"ERROR: Registry returned {status}: {e}")

    model_data = model_info.get("model", model_info)
    file_size = int(model_data.get("fileSize", 0))
    category = model_data.get("category", "unknown")
    description = model_data.get("description", "")
    license_id = model_data.get("license", "unknown")
    license_name = model_data.get("licenseName", license_id)
    license_url = model_data.get("licenseUrl", "")
    is_gated = model_data.get("gated", False)

    print(f"   Category: {category}")
    if description:
        print(f"   Description: {description[:80]}")
    if file_size > 0:
        print(f"   Size: {format_size(file_size)}")
    print(f"   License: {license_name}")
    _ev(args, "info", model=f"{org}/{name}", category=category, bytes=file_size,
        license=license_id, license_name=license_name, gated=bool(is_gated))

    # Before a byte is fetched. A 30.6 GB import died at 98 % on a full disk
    # because nothing looked first; the peak is store + cache, about twice the
    # archive, and --no-keep does not lower it. Both numbers are declared.
    if file_size > 0:
        import shutil as _shutil
        _need = import_peak_bytes(file_size, keep=not getattr(args, "no_keep", False))
        _free = min(
            _shutil.disk_usage(_p).free
            for _p in {_nearest_existing(STORE_DIR), _nearest_existing(CACHE_DIR)})
        _refusal = disk_refusal(_need, _free)
        if _refusal:
            _die(args, f"ERROR: not enough disk for this import — {_refusal}.",
                 "       Free space, or import to a device that has it.")
        print(f"   Disk: {format_size(_need)} needed at peak, "
              f"{format_size(_free)} free")

    # License acceptance — hub is the source of truth
    if is_gated and not _is_license_accepted(org, name):
        full_url = ""
        if license_url:
            full_url = license_url if license_url.startswith("http") else f"{registry}{license_url}"

        print(f"\n{'=' * 70}")
        print("LICENSE NOTICE")
        print("=" * 70)
        print(f"   Vendor: {org}")
        print(f"   License: {license_name}")
        if full_url:
            print(f"   Full text: {full_url}")
        print()
        print("   This model is distributed under the license above.")
        print("   NeuroBrix does not modify the original license terms.")
        print("   You are responsible for complying with the license.")
        print()
        print("   THIS LICENSE REQUIRES EXPLICIT ACCEPTANCE.")
        print("=" * 70)

        # Non-interactive acceptance: --accept-license flag or NBX_ACCEPT_LICENSE=1
        accepted_via = None
        if getattr(args, "accept_license", False):
            accepted_via = "--accept-license"
        elif os.environ.get("NBX_ACCEPT_LICENSE") == "1":
            accepted_via = "NBX_ACCEPT_LICENSE=1"

        if accepted_via:
            _record_license_acceptance(org, name, license_id)
            print(f"   License accepted via {accepted_via}.\n")
            _ev(args, "license", license=license_id, accepted_via=accepted_via)
        else:
            # A client never answers a prompt: under --json the licence is
            # an explicit parameter (--accept-license) or a refusal that
            # names it — the prompt is for a person at a terminal.
            if wants_json(args) or not sys.stdin.isatty():
                _print_noninteractive_license_help(model_ref, license_name, full_url)
                _die(args, f"ERROR: license '{license_id}' requires explicit acceptance: "
                     f"re-run with --accept-license (full text: {full_url or 'see the hub'})")

            try:
                reply = input("\n   Accept license terms? [yes/No]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                _print_noninteractive_license_help(model_ref, license_name, full_url)
                sys.exit(1)

            if reply != "yes":
                print("   License declined. Download cancelled.")
                sys.exit(1)

            _record_license_acceptance(org, name, license_id)
            print("   License accepted.\n")
            _ev(args, "license", license=license_id, accepted_via="prompt")

    # 2. Get signed download URL
    print(f"\n[2/4] Getting download URL...")
    try:
        headers = {}
        if is_gated:
            headers["X-License-Accepted"] = "true"
        resp = requests.get(f"{registry}/api/models/{org}/{name}/download", headers=headers, timeout=10)
        resp.raise_for_status()
        download_info = resp.json()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        server_error = None
        if e.response is not None:
            try:
                body = e.response.json()
                if isinstance(body, dict):
                    server_error = body
            except ValueError:
                server_error = None

        if server_error is not None:
            code = server_error.get("code", "")
            message = server_error.get("message") or server_error.get("error") or ""
            if code == "LICENSE_LOGIN_REQUIRED":
                _die(args,
                     f"ERROR: {message}" if message else
                     "ERROR: License acceptance on the hub is required for this model.",
                     "The hub requires a logged-in license acceptance for this model.",
                     f"Log in on {registry}, accept the license for {org}/{name}, then retry the import.")
            if code or message:
                _die(args, f"ERROR: Failed to get download URL ({status}): {message or code}",
                     *( [f"   Server error code: {code}"] if code and message else [] ))

        _die(args, f"ERROR: Failed to get download URL: {e}")

    download_url = download_info.get("url")
    file_name = download_info.get("fileName", f"{name}.nbx")

    if not download_url:
        _die(args, "ERROR: Registry did not return a download URL.",
             "The model may not have a .nbx file uploaded yet.")

    # 3. Download .nbx to store/
    print(f"\n[3/4] Downloading {file_name}...")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    store_path = STORE_DIR / file_name

    try:
        actual_size = download_resumable(download_url, store_path, total_hint=file_size,
                                         desc=file_name, get=requests.get,
                                         progress=NdjsonProgress if wants_json(args) else tqdm)
        print(f"   Saved: {store_path} ({format_size(actual_size)})")
        _ev(args, "downloaded", file=file_name, store=str(store_path), bytes=actual_size)

    except requests.HTTPError as e:
        _die(args, f"ERROR: Download failed: {e}")
    except (requests.ConnectionError, requests.Timeout, IncompleteDownload) as e:
        part = partial_path(store_path)
        have = part.stat().st_size if part.exists() else 0
        _die(args, f"ERROR: Connection lost during download ({e}).",
             f"   {format_size(have)} are kept in {part}; re-run the same command to resume "
             f"from there.")

    # 4. Extract to cache/ — into a staging directory beside the final name,
    # renamed in one motion at the end: a model is visible (to `list`, to
    # `run`, to a client) only after its extraction succeeded, never as a
    # half-written directory that already carries a manifest (Studio request 6).
    print(f"\n[4/4] Extracting to cache...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _ev(args, "extracting", store=str(store_path), cache=str(cache_path))

    import zipfile
    if not zipfile.is_zipfile(store_path):
        _die(args, "ERROR: Downloaded file is not a valid .nbx (ZIP) archive.")
    # The staging tree, the lock and the two-rename swap are one brick, shared
    # with `NBXCache.extract`: the cache is often a mounted export, and the old
    # code here removed the live tree BEFORE renaming staging over it, leaving
    # the model absent for the length of a recursive delete and destroying the
    # copy a run on another machine was loading from.
    from neurobrix.nbx.atomic_install import installing, InstallHeldByAnother
    try:
        with installing(cache_path, label=f"import {org}/{name}") as staging:
            extract_container(store_path, staging)
    except InstallHeldByAnother as e:
        _die(args, f"ERROR: {e}")
    print(f"   Extracted: {cache_path}")
    _ev(args, "installed", model=f"{org}/{name}", cache=str(cache_path), already=False)

    # Delete .nbx from store if --no-keep
    if args.no_keep:
        store_path.unlink()
        print(f"   Store: deleted (--no-keep)")
    else:
        print(f"   Store: {store_path} (kept — use --no-keep to save {format_size(actual_size)})")

    print(f"\n{'=' * 70}")
    print("IMPORT COMPLETE")
    print("=" * 70)
    print(f"Model: {org}/{name}")
    print(f"Cache: {cache_path}")
    print(f"\nRun with: {_suggest_run_command(name, cache_path)}")
    _ev(args, "done", model=f"{org}/{name}", cache=str(cache_path),
        store=None if args.no_keep else str(store_path))


def _suggest_run_command(name: str, cache_path) -> str:
    """The closing line of `import`, built from the family's declared inputs.

    It used to assume every model is prompt-driven and printed
    `--prompt "..."` for all of them, so the last line a newcomer read before
    their first run was a command that could not work — an upscaler answered
    it with `ZERO FALLBACK: family 'upscaler' requires --input-image`
    (hub walkthrough, 2026-09-03).

    The family YAML already declares `inputs.required`, so the suggestion is
    read from there. No family cascade: adding a family to the taxonomy must
    not require touching this function (R32).
    """
    try:
        import json as _json

        manifest = _json.loads((Path(cache_path) / "manifest.json").read_text())
        family = manifest.get("family")
        if not family:
            raise ValueError("no family in manifest")

        from neurobrix.core.runtime.output_dispatch import get_family_config

        required = (get_family_config(family).get("inputs") or {}).get("required") or []
    except Exception:
        # Unknown family or unreadable manifest: fall back to the bare command
        # rather than inventing flags the model may not accept.
        return f"neurobrix run --model {name}"

    parts = [f"neurobrix run --model {name}"]
    for flag in required:
        # Placeholder derived mechanically from the flag, so a new required
        # input needs no table here: --input-image -> IMAGE, --audio -> AUDIO.
        placeholder = flag.lstrip("-").split("-")[-1].upper()
        parts.append(f'{flag} "..."' if placeholder == "PROMPT"
                     else f"{flag} <{placeholder}>")
    return " ".join(parts)


def _strip_build_stamp(stem: str) -> str:
    """`Model.20260828T183831` -> `Model`.

    Store filenames carry a build stamp; extracted cache directories do not.
    Anything comparing the two must normalise first.
    """
    return re.sub(r"\.\d{8}T\d{6}$", "", stem)


def list_record(args) -> dict:
    """The installed models and the store, as one record (the same walk `list` prints)."""
    store_files = {}
    if STORE_DIR.exists():
        for nbx_file in STORE_DIR.glob("*.nbx"):
            store_files[nbx_file.stem] = nbx_file.stat().st_size
    models = []
    if CACHE_DIR.exists():
        for model_dir in sorted(CACHE_DIR.iterdir()):
            if not is_installed_model_dir(model_dir):
                continue
            try:
                manifest = json.loads((model_dir / "manifest.json").read_text())
            except (json.JSONDecodeError, OSError):
                manifest = {}
            models.append({"name": model_dir.name, "family": manifest.get("family", "?"),
                           "size_bytes": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()),
                           "license": manifest.get("license", "") or "",
                           "in_store": any(model_dir.name in stem for stem in store_files)})
    installed = {m["name"] for m in models}
    store_only = [{"name": stem, "size_bytes": size} for stem, size in sorted(store_files.items())
                  if not any(name in stem for name in installed)]
    return {"models": models, "store_only": store_only,
            "store": {"path": str(STORE_DIR), "files": [{"name": k + ".nbx", "size_bytes": v} for k, v in sorted(store_files.items())],
                      "bytes": sum(store_files.values())}}


def cmd_list(args):
    from neurobrix.cli.json_out import wants_json, emit
    if wants_json(args):
        emit("list", list_record(args))
        return
    """List installed models (cache) and downloaded archives (store)."""

    # --store: show store contents only
    if args.store:
        print("=" * 70)
        print("NeuroBrix Store (~/.neurobrix/store/)")
        print("=" * 70)

        if not STORE_DIR.exists() or not any(STORE_DIR.glob("*.nbx")):
            print("\nStore is empty.")
            print("  .nbx files are kept here after import (use --no-keep to skip)")
            return

        print(f"\n{'FILE':<45} {'SIZE':>10}")
        print("-" * 57)
        total_size = 0
        count = 0
        for nbx_file in sorted(STORE_DIR.glob("*.nbx")):
            size = nbx_file.stat().st_size
            total_size += size
            count += 1
            print(f"{nbx_file.name:<45} {format_size(size):>10}")

        print(f"\nTotal: {count} file(s), {format_size(total_size)}")
        print(f"Path:  {STORE_DIR}")
        print(f"\nFree space: neurobrix clean --store")
        return

    # Default: show cache (installed models) with store indicator
    print("=" * 70)
    print("NeuroBrix Models")
    print("=" * 70)

    # Index store .nbx files for cross-reference
    store_files = {}
    if STORE_DIR.exists():
        for nbx_file in STORE_DIR.glob("*.nbx"):
            store_files[nbx_file.stem] = nbx_file.stat().st_size

    models = []

    if CACHE_DIR.exists():
        for model_dir in sorted(CACHE_DIR.iterdir()):
            if not is_installed_model_dir(model_dir):
                continue
            manifest_path = model_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    family = manifest.get("family", "?")
                    license_id = manifest.get("license", "")
                except (json.JSONDecodeError, OSError):
                    family = "?"
                    license_id = ""

                total_size = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )

                # Check if .nbx backup exists in store
                has_store = any(model_dir.name in stem for stem in store_files)

                models.append({
                    "name": model_dir.name,
                    "family": family,
                    "size": total_size,
                    "store": has_store,
                    "license": license_id,
                })

    # Also show store-only entries (downloaded but not extracted).
    #
    # The store filename carries a build stamp — "Swin2SR-Classical-x4
    # .20260828T183831.nbx" — while the extracted cache directory is the bare
    # model name. Comparing the two directly never matched, so a model that
    # HAD just been extracted was listed as installed and as "store only (not
    # extracted)" on the same screen, with an instruction to import it again
    # (hub walkthrough, 2026-09-03). Strip the stamp before comparing.
    cached_names = {m["name"] for m in models}
    store_only = []
    for stem, size in store_files.items():
        if _strip_build_stamp(stem) not in cached_names:
            store_only.append({"name": stem, "size": size})

    if not models and not store_only:
        print("\nNo models installed.")
        print(f"\n  Install from registry: neurobrix import <org>/<model>")
        print(f"  Browse hub:            neurobrix hub")
        return

    if models:
        print(f"\n{'MODEL':<35} {'FAMILY':<10} {'SIZE':>10} {'LICENSE':>16} {'STORE':>8}")
        print("-" * 82)
        for m in models:
            store_str = ".nbx" if m["store"] else "-"
            lic = m.get("license", "") or "-"
            print(f"{m['name']:<35} {m['family']:<10} {format_size(m['size']):>10} {lic:>16} {store_str:>8}")
        print(f"\nInstalled: {len(models)} model(s)")

    if store_only:
        print(f"\nStore only (not extracted):")
        for s in store_only:
            print(f"  {s['name']}.nbx  ({format_size(s['size'])})")
        # `<org>` was printed as a literal, so the suggested command could not
        # be pasted. The org is not recoverable from the store filename, so
        # point at the command that finds it instead of inventing a value.
        _first = _strip_build_stamp(store_only[0]["name"])
        print(f"  Find its org:  neurobrix hub --search {_first}")
        print(f"  Then extract:  neurobrix import <org>/{_first} --force")

    if store_files:
        total_store = sum(store_files.values())
        print(f"\nStore usage: {format_size(total_store)} ({len(store_files)} file(s))")
        print(f"  View: neurobrix list --store")
        print(f"  Free: neurobrix clean --store")


def _find_store_files(model_name):
    """Find .nbx files in store matching a model name."""
    matches = []
    if STORE_DIR.exists():
        for nbx_file in STORE_DIR.glob("*.nbx"):
            if model_name in nbx_file.stem:
                matches.append(nbx_file)
    return matches


def cmd_remove(args):
    """Remove a model from cache, store, or both. Under --json one record:
    what was removed (kind, path, bytes) and whether the model was found."""
    from neurobrix.cli.json_out import wants_json, human_lines_to_stderr, emit
    record = {"model": args.model_name, "removed": []}
    with human_lines_to_stderr(wants_json(args)):
        try:
            _remove_body(args, record)
        finally:
            if wants_json(args):
                record["found"] = bool(record["removed"])
                emit("remove", record)


def _remove_body(args, record):
    import shutil

    model_name = args.model_name
    cache_path = CACHE_DIR / model_name
    do_store = args.store or getattr(args, 'all', False)
    do_cache = not args.store or getattr(args, 'all', False)

    print("=" * 70)
    print("NeuroBrix Remove")
    print("=" * 70)

    removed = False

    # Remove from cache
    if do_cache and cache_path.exists():
        total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        shutil.rmtree(cache_path)
        print(f"Removed cache: {cache_path} ({format_size(total_size)} freed)")
        record["removed"].append({"kind": "cache", "path": str(cache_path), "bytes": total_size})
        removed = True
    elif do_cache and not cache_path.exists():
        if not do_store:
            # Cache-only mode but nothing in cache — check if in store
            store_matches = _find_store_files(model_name)
            if store_matches:
                print(f"Model '{model_name}' not in cache, but found in store:")
                for f in store_matches:
                    print(f"  {f.name} ({format_size(f.stat().st_size)})")
                print(f"\nTo remove from store: neurobrix remove {model_name} --store")
                sys.exit(1)

    # Remove from store
    if do_store:
        store_matches = _find_store_files(model_name)
        for nbx_file in store_matches:
            size = nbx_file.stat().st_size
            nbx_file.unlink()
            print(f"Removed store: {nbx_file.name} ({format_size(size)} freed)")
            record["removed"].append({"kind": "store", "path": str(nbx_file), "bytes": size})
            removed = True

        if not store_matches and not removed:
            print(f"No .nbx file found for '{model_name}' in store.")

    if not removed:
        print(f"Model '{model_name}' not found.")
        # Show what's available
        available = []
        if CACHE_DIR.exists():
            available += [f"{d.name} (cache)" for d in CACHE_DIR.iterdir()
                         if is_installed_model_dir(d)]
        for nbx_file in (STORE_DIR.glob("*.nbx") if STORE_DIR.exists() else []):
            available.append(f"{nbx_file.stem} (store)")
        if available:
            print(f"Available: {', '.join(available)}")
        sys.exit(1)

    print("\nDone.")


def cmd_clean(args):
    """Wipe all downloaded models from store and/or cache."""
    import shutil

    do_store = args.all or args.store
    do_cache = args.all or args.cache

    if not do_store and not do_cache:
        print("ERROR: Specify --store, --cache, or --all")
        print("  neurobrix clean --store   # Delete all .nbx files")
        print("  neurobrix clean --cache   # Delete all extracted models")
        print("  neurobrix clean --all     # Delete both")
        sys.exit(1)

    print("=" * 70)
    print("NeuroBrix Clean")
    print("=" * 70)

    store_size = 0
    store_count = 0
    cache_size = 0
    cache_count = 0

    if do_store and STORE_DIR.exists():
        for f in STORE_DIR.glob("*.nbx"):
            store_size += f.stat().st_size
            store_count += 1
        if store_count:
            print(f"  Store: {store_count} file(s), {format_size(store_size)}  ({STORE_DIR})")

    if do_cache and CACHE_DIR.exists():
        for d in CACHE_DIR.iterdir():
            if is_installed_model_dir(d):
                cache_size += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                cache_count += 1
        if cache_count:
            print(f"  Cache: {cache_count} model(s), {format_size(cache_size)}  ({CACHE_DIR})")

    total = store_count + cache_count
    if total == 0:
        print("Nothing to clean.")
        return

    total_size = (store_size + cache_size) / (1024**3)
    print(f"\n  Total: {total_size:.2f} GB will be freed")

    if not args.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply != 'y':
            print("Cancelled.")
            return

    freed = 0
    if do_store and STORE_DIR.exists():
        for f in STORE_DIR.glob("*.nbx"):
            freed += f.stat().st_size
            f.unlink()
        print(f"Wiped store: {store_count} file(s) removed")

    if do_cache and CACHE_DIR.exists():
        for d in list(CACHE_DIR.iterdir()):
            if d.is_dir():
                shutil.rmtree(d)
        print(f"Wiped cache: {cache_count} model(s) removed")

    print(f"\nDone. {(freed + cache_size) / (1024**3):.2f} GB freed.")


def cmd_hub(args):
    """Browse models available on the NeuroBrix registry."""
    from neurobrix.cli.json_out import wants_json as _wj, human_lines_to_stderr
    with human_lines_to_stderr(_wj(args)):
        return _cmd_hub(args)


def _cmd_hub(args):
    """Browse models available on the NeuroBrix registry."""
    # `requests`, like every other registry call in this file, and NOT raw
    # urllib. urllib verifies against OpenSSL's default CA file, and on a
    # python.org macOS install that file does not exist until the user runs
    # `Install Certificates.command` — measured here 2026-09-18:
    #
    #   ssl default verify paths: .../Python.framework/.../etc/openssl/cert.pem
    #   that file exists: False
    #   plain urlopen    : CERTIFICATE_VERIFY_FAILED
    #   requests         : HTTP 200
    #
    # So `neurobrix hub` reported "Cannot connect to registry" on a machine
    # whose network was fine and whose registry answered every other command,
    # because this one function reached for a different HTTP client. requests
    # carries certifi and is already a declared dependency.
    import requests
    import urllib.parse

    registry = args.registry or REGISTRY_URL

    # Build query parameters
    params = {"limit": "100"}
    if args.category:
        params["category"] = args.category.upper()
    if args.search:
        params["q"] = args.search

    url = f"{registry}/api/models?{urllib.parse.urlencode(params)}"

    print("=" * 70)
    print("NeuroBrix Hub")
    print("=" * 70)
    print(f"Registry: {registry}")
    if args.category:
        print(f"Category: {args.category.upper()}")
    if args.search:
        print(f"Search: {args.search}")

    # Fetch model list (no auth required)
    try:
        resp = requests.get(url, timeout=10, headers={
            "Accept": "application/json",
            "User-Agent": f"neurobrix-cli/{__version__}",
        })
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        # HTTPError subclasses URLError, so it used to land in the branch
        # below and every rejected request was reported as a connectivity
        # failure — sending a user who had merely mistyped a category off to
        # check their firewall. The registry answered, and it answered
        # usefully: the body carries the reason and the vocabulary.
        detail = {}
        code = e.response.status_code if e.response is not None else 0
        try:
            detail = e.response.json()
        except Exception:
            pass
        if code < 500 and detail:
            print(f"\nERROR: {detail.get('error', e)}")
            valid = detail.get("validCategories")
            if valid:
                print(f"  Valid categories: {', '.join(valid)}")
            sys.exit(2)
        print(f"\nERROR: registry returned HTTP {code} for {url}")
        print(f"  {e}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"\nERROR: Cannot connect to registry at {registry}")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    remote_models = data.get("models", [])
    total_count = data.get("total", len(remote_models))
    from neurobrix.cli.json_out import wants_json as _wj, emit as _emit
    if _wj(args):
        installed_now = set()
        if CACHE_DIR.exists():
            installed_now = {d.name for d in CACHE_DIR.iterdir() if d.is_dir() and (d / "manifest.json").exists()}
        rows = []
        for rm in remote_models:
            slug = rm.get("slug", f"{rm.get('org', '?')}/{rm.get('name', '?')}")
            name = rm.get("name", slug.split("/")[-1])
            rows.append({"slug": slug, "name": name, "category": rm.get("category", "?"),
                         "size_bytes": int(rm.get("fileSize", 0) or 0), "license": rm.get("license") or None,
                         "downloads": rm.get("downloadCount", 0), "visibility": (rm.get("visibility") or "PUBLIC").upper(),
                         "installed": name in installed_now})
        _emit("hub", {"registry": registry, "query": {"category": args.category, "search": args.search},
                      "total": total_count, "models": rows})
        return
    if not remote_models:
        print("\nNo models found.")
        if args.category or args.search:
            print("Try without filters: neurobrix hub")
        sys.exit(0)

    # Check which are already installed locally
    installed = set()
    if CACHE_DIR.exists():
        for d in CACHE_DIR.iterdir():
            if is_installed_model_dir(d):
                installed.add(d.name)

    # Display
    print(f"\n{'MODEL':<30} {'CATEGORY':<10} {'SIZE':>10} {'LICENSE':>16} {'DL':>6}  STATUS")
    print("-" * 90)

    for rm in remote_models:
        slug = rm.get("slug", f"{rm.get('org', '?')}/{rm.get('name', '?')}")
        category = rm.get("category", "?")
        file_size = int(rm.get("fileSize", 0))
        downloads = rm.get("downloadCount", 0)
        lic = rm.get("license", "-") or "-"
        name = rm.get("name", slug.split("/")[-1])
        # The visibility badge must reach the user BEFORE the download:
        # a non-public state (DEPRECATED) outranks the local "installed"
        # marker in this column. The stated reason is returned by the
        # download API and shown at import time.
        visibility = (rm.get("visibility") or "PUBLIC").upper()
        if visibility != "PUBLIC":
            status = visibility
        elif name in installed:
            status = "installed"
        else:
            status = ""

        print(f"{slug:<30} {category:<10} {format_size(file_size):>10} {lic:>16} {downloads:>6}  {status}")

    print(f"\nTotal: {total_count} model(s) on registry")

    if installed:
        print(f"Installed locally: {len(installed)}")

    print(f"\nInstall: neurobrix import <org>/<model>")


# ============================================================================
# LICENSE ACCEPTANCE CACHE
# ============================================================================
# Stored in ~/.neurobrix/license_acceptances.json
# Hub is the source of truth for gating. This file only records user consent.

from pathlib import Path
from datetime import datetime, timezone

_ACCEPTANCES_FILE = Path.home() / ".neurobrix" / "license_acceptances.json"


def _print_noninteractive_license_help(model_ref: str, license_name: str, full_url: str) -> None:
    """Explain how to accept a gated-model license without a terminal prompt."""
    print("\n   ERROR: This model requires explicit license acceptance, but no")
    print("   interactive terminal is available to prompt for it.")
    print()
    print(f"   License: {license_name}")
    if full_url:
        print(f"   Full text: {full_url}")
    print()
    print("   To accept the license non-interactively, re-run with either:")
    print(f"     neurobrix import {model_ref} --accept-license")
    print(f"     NBX_ACCEPT_LICENSE=1 neurobrix import {model_ref}")


def _is_license_accepted(org: str, name: str) -> bool:
    """Check if user has already accepted the license for org/name.

    Acceptances are keyed by "org/name". Legacy entries keyed by the bare
    model name are honored and migrated to the org/name key on first read.
    """
    if not _ACCEPTANCES_FILE.exists():
        return False
    try:
        data = json.loads(_ACCEPTANCES_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    key = f"{org}/{name}"
    if key in data:
        return True
    if name in data:
        # Legacy bare-name entry: migrate to the org/name key.
        data[key] = data.pop(name)
        _ACCEPTANCES_FILE.write_text(json.dumps(data, indent=2))
        return True
    return False


def _record_license_acceptance(org: str, name: str, license_id: str) -> None:
    """Record license acceptance locally so user isn't asked again."""
    _ACCEPTANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if _ACCEPTANCES_FILE.exists():
        try:
            data = json.loads(_ACCEPTANCES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    data[f"{org}/{name}"] = {
        "license": license_id,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }
    _ACCEPTANCES_FILE.write_text(json.dumps(data, indent=2))
