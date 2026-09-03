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


def cmd_import(args):
    """Download model from NeuroBrix registry and extract to local cache."""
    import requests
    from tqdm import tqdm

    registry = args.registry or REGISTRY_URL
    model_ref = args.model_ref

    if "/" not in model_ref:
        print(f"ERROR: Invalid model reference '{model_ref}'")
        print("Expected format: org/name (e.g., pixart/sigma-xl-1024)")
        sys.exit(1)

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
        sys.exit(0)

    # 1. Get model metadata
    print(f"\n[1/4] Fetching model info...")
    try:
        resp = requests.get(f"{registry}/api/models/{org}/{name}", timeout=10)
        resp.raise_for_status()
        model_info = resp.json()
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to registry at {registry}")
        print("Check your network connection or use --registry to specify a different URL.")
        sys.exit(1)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        if status == 404:
            print(f"ERROR: Model '{org}/{name}' not found on registry.")
            print(f"Browse available models: {registry}")
        else:
            print(f"ERROR: Registry returned {status}: {e}")
        sys.exit(1)

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
        else:
            if not sys.stdin.isatty():
                _print_noninteractive_license_help(model_ref, license_name, full_url)
                sys.exit(1)

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
                if message:
                    print(f"ERROR: {message}")
                else:
                    print("ERROR: License acceptance on the hub is required for this model.")
                print("The hub requires a logged-in license acceptance for this model.")
                print(f"Log in on {registry}, accept the license for {org}/{name}, then retry the import.")
                sys.exit(1)
            if code or message:
                print(f"ERROR: Failed to get download URL ({status}): {message or code}")
                if code and message:
                    print(f"   Server error code: {code}")
                sys.exit(1)

        print(f"ERROR: Failed to get download URL: {e}")
        sys.exit(1)

    download_url = download_info.get("url")
    file_name = download_info.get("fileName", f"{name}.nbx")

    if not download_url:
        print("ERROR: Registry did not return a download URL.")
        print("The model may not have a .nbx file uploaded yet.")
        sys.exit(1)

    # 3. Download .nbx to store/
    print(f"\n[3/4] Downloading {file_name}...")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    store_path = STORE_DIR / file_name

    try:
        resp = requests.get(download_url, stream=True, timeout=30)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) or file_size
        with open(store_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=file_name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        actual_size = store_path.stat().st_size
        print(f"   Saved: {store_path} ({format_size(actual_size)})")

    except requests.HTTPError as e:
        print(f"ERROR: Download failed: {e}")
        if store_path.exists():
            store_path.unlink()
        sys.exit(1)
    except requests.ConnectionError:
        print("ERROR: Connection lost during download.")
        if store_path.exists():
            store_path.unlink()
        sys.exit(1)

    # 4. Extract to cache/
    print(f"\n[4/4] Extracting to cache...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        import shutil
        shutil.rmtree(cache_path)

    import zipfile
    if zipfile.is_zipfile(store_path):
        cache_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(store_path, 'r') as zf:
            for member in zf.namelist():
                member_resolved = os.path.realpath(os.path.join(cache_path, member))
                if not member_resolved.startswith(os.path.realpath(str(cache_path)) + os.sep) and member_resolved != os.path.realpath(str(cache_path)):
                    raise ValueError(f"Security: path traversal detected in archive member: {member}")
            zf.extractall(cache_path)
        print(f"   Extracted: {cache_path}")
    else:
        print(f"ERROR: Downloaded file is not a valid .nbx (ZIP) archive.")
        sys.exit(1)

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


def cmd_list(args):
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
            if not model_dir.is_dir():
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
    """Remove a model from cache, store, or both."""
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
            removed = True

        if not store_matches and not removed:
            print(f"No .nbx file found for '{model_name}' in store.")

    if not removed:
        print(f"Model '{model_name}' not found.")
        # Show what's available
        available = []
        if CACHE_DIR.exists():
            available += [f"{d.name} (cache)" for d in CACHE_DIR.iterdir()
                         if d.is_dir() and (d / "manifest.json").exists()]
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
            if d.is_dir() and (d / "manifest.json").exists():
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
    import urllib.request
    import urllib.error
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
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": f"neurobrix-cli/{__version__}",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # HTTPError subclasses URLError, so it used to land in the branch
        # below and every rejected request was reported as a connectivity
        # failure — sending a user who had merely mistyped a category off to
        # check their firewall. The registry answered, and it answered
        # usefully: the body carries the reason and the vocabulary.
        detail = {}
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            pass
        if e.code < 500 and detail:
            print(f"\nERROR: {detail.get('error', e)}")
            valid = detail.get("validCategories")
            if valid:
                print(f"  Valid categories: {', '.join(valid)}")
            sys.exit(2)
        print(f"\nERROR: registry returned HTTP {e.code} for {url}")
        print(f"  {e}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\nERROR: Cannot connect to registry at {registry}")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    remote_models = data.get("models", [])
    total_count = data.get("total", len(remote_models))

    if not remote_models:
        print("\nNo models found.")
        if args.category or args.search:
            print("Try without filters: neurobrix hub")
        sys.exit(0)

    # Check which are already installed locally
    installed = set()
    if CACHE_DIR.exists():
        for d in CACHE_DIR.iterdir():
            if d.is_dir() and (d / "manifest.json").exists():
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
