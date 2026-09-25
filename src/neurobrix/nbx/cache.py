# core/nbx/cache.py
"""
NBX Cache Manager.

Extracts NBX containers to flat directory for fast loading.
Provisoire solution until NBX v2 format.

OPTIMIZATION:
  OLD: NBX (ZIP) -> zipfile.read() -> CPU RAM -> .to(cuda) -> VRAM = 3 copies
  NEW: NBX (ZIP) -> Extract once -> ~/.neurobrix/cache/ -> safetensors.load_file(device="cuda") = 1 copy

Parallel extraction using ThreadPoolExecutor
  - ~80% faster extraction for large models
  - 8 workers default (I/O bound, not CPU bound)
"""

import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class NBXCache:
    """
    Manages extracted NBX cache for fast loading.

    Cache structure: ~/.neurobrix/cache/<model_name>/

    Benefits:
    - First run: Extract NBX to flat directory
    - Subsequent runs: Direct file access, no ZIP overhead
    - safetensors can load directly to GPU via mmap
    """

    #: The machine's cache location, from the one door (`core.paths`).
    #: This was a second literal for the same thing until 2026-09-17.
    @staticmethod
    def _default_cache_dir():
        from neurobrix.core.paths import cache_dir
        return cache_dir()

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, nbx_path: Path) -> Path:
        """Get cache directory path for an NBX file."""
        # Use model name from path (parent directory name)
        model_name = Path(nbx_path).parent.name
        return self.cache_dir / model_name

    def is_cached(self, nbx_path: Path) -> bool:
        """Check if NBX is already extracted and valid."""
        cache_path = self.get_cache_path(nbx_path)
        manifest_path = cache_path / "manifest.json"
        meta_path = cache_path / ".cache_meta.json"

        if not manifest_path.exists():
            return False

        if not meta_path.exists():
            return False

        # Verify cache is newer than NBX file
        nbx_path = Path(nbx_path)
        if nbx_path.exists():
            nbx_mtime = nbx_path.stat().st_mtime
            cache_mtime = manifest_path.stat().st_mtime
            return cache_mtime >= nbx_mtime

        return True

    def get_cache_info(self, nbx_path: Path) -> Optional[Dict[str, Any]]:
        """Get cache metadata if exists."""
        cache_path = self.get_cache_path(nbx_path)
        meta_path = cache_path / ".cache_meta.json"

        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    #: The named opening on the replacement door below. Deliberate, never silent.
    REPLACE_ENV = "NBX_ALLOW_CACHE_REPLACE"

    def _refuse_incidental_replacement(self, nbx_path: Path, cache_path: Path,
                                       declared: bool) -> None:
        """Refuse to REPLACE a cached container as a side effect of reading one.

        `NBXContainer.load()`, `NBXLoader` and `core/runtime/factory` all call
        `ensure_extracted()` — they are READ paths, and none of them means "replace the
        canonical copy of this model". But `extract()` re-unpacks whenever the `.nbx` is
        newer than the cache, so handing any of them a freshly built container silently
        overwrote the installed one.

        Measured 2026-09-22 17:44: a call made to GATE a new build's symbolic dims printed
        `[Cache] Extracting model.nbx -> ~/.neurobrix/<cache>/mochi-1-preview` and replaced
        the canonical container, 43 files and 41.05 GB, with the new build. The repository's
        blocking hook could not see it: the hook refuses a SHELL COMMAND that names the cache
        path, and this write came from inside a library that resolved the destination itself
        from a path in `nbx/builds/`. A door that watches the command line cannot see that.

        THE HARMFUL STATE, named rather than the outcome: a container this cache did not get
        from THIS `.nbx` is replaced by it. So:

        * nothing cached yet -> a fresh install, allowed;
        * `.cache_meta.json` records THIS same source -> the artefact was rebuilt in place,
          which is the ordinary update path, allowed;
        * anything else -> a replacement, and it must be declared.

        Only 2 of the 59 containers in this cache carry `.cache_meta.json` at all; the other
        57 arrived by another route, and a missing record is exactly the case where the
        replacement is least intended. Absent metadata therefore refuses rather than assumes.

        The opening is named and reads as deliberate: `force=True` / `allow_replace=True` at
        the call, or `NBX_ALLOW_CACHE_REPLACE=1` in the environment for a CLI that installs.
        """
        if declared or os.environ.get(self.REPLACE_ENV) == "1":
            return
        if not (cache_path / "manifest.json").exists():
            return                                  # fresh install, nothing to lose

        recorded = ""
        meta_path = cache_path / ".cache_meta.json"
        if meta_path.exists():
            try:
                recorded = str((json.loads(meta_path.read_text()) or {}).get("source") or "")
            except Exception:                       # noqa: BLE001 — unreadable == unknown
                recorded = ""
        if recorded:
            try:
                if Path(recorded).resolve() == Path(nbx_path).resolve():
                    return                          # same artefact, rebuilt in place
            except Exception:                       # noqa: BLE001
                pass

        raise RuntimeError(
            f"ZERO FALLBACK: the cached container at {cache_path}\n"
            f"  did not come from this .nbx, and this call would either replace it or\n"
            f"  silently serve it in place of what was asked for.\n"
            f"  with a different .nbx: {nbx_path}\n"
            f"  the cache records its source as: {recorded or '(no .cache_meta.json — unknown)'}\n"
            f"\n"
            f"  Reading a container must not overwrite the installed one. The cache is the\n"
            f"  canonical source the census reads, and 57 of its 59 containers carry no\n"
            f"  record of where they came from, so an unrecorded source is refused rather\n"
            f"  than assumed to be the same artefact.\n"
            f"\n"
            f"  To READ this .nbx without touching the installed container, point the cache\n"
            f"  elsewhere for the call:\n"
            f"      NEUROBRIX_CACHE=<a scratch dir> ...\n"
            f"  To REPLACE it on purpose, say so:\n"
            f"      {self.REPLACE_ENV}=1 ...      (or extract(..., allow_replace=True))"
        )

    def extract(self, nbx_path: Path, force: bool = False,
                allow_replace: bool = False) -> Path:
        """
        Extract NBX to cache directory.

        Args:
            nbx_path: Path to .nbx file
            force: Force re-extraction even if cached

        Returns:
            Path to extracted cache directory
        """
        nbx_path = Path(nbx_path)
        cache_path = self.get_cache_path(nbx_path)

        # THE DOOR, ABOVE the cached short-circuit. Both directions are harmful and only
        # one of them is a replacement:
        #   * cache OLDER than this .nbx -> it is re-extracted, and the installed container
        #     is REPLACED as a side effect of a read (the 2026-09-22 incident);
        #   * cache NEWER -> `is_cached()` returns True and this returns the cached tree,
        #     which is a DIFFERENT container under the requested name. The caller asked for
        #     one artefact and silently received another, with no line printed.
        # `get_cache_path` keys the slot on the .nbx's PARENT DIRECTORY NAME, so two builds
        # of the same model in different trees always collide here.
        self._refuse_incidental_replacement(nbx_path, cache_path,
                                            declared=bool(force or allow_replace))

        if self.is_cached(nbx_path) and not force:
            print(f"[Cache] Using cached: {cache_path}")
            return cache_path

        print(f"[Cache] Extracting {nbx_path.name} -> {cache_path}")

        # Extraction goes to a private staging tree and is swapped in at the end.
        #
        # This used to remove the live tree and then extract IN PLACE at the final
        # name, for as long as the container took to unpack. A `manifest.json`
        # appears from the first member on, so for those minutes any other reader —
        # a `run` on this machine, or on another machine sharing this mounted cache
        # — saw a half-written model as a complete one, and the model it had been
        # about to load was gone. `cli/commands/registry.py` had already been fixed
        # for exactly this (Studio request 6); the same bug written twice is a
        # missing brick, and `nbx/atomic_install` is that brick.
        from neurobrix.nbx.atomic_install import installing
        with installing(cache_path, label=f"cache extract {nbx_path.name}") as staging:
            return self._extract_into(nbx_path, staging, cache_path)

    def _extract_into(self, nbx_path: Path, cache_path: Path,
                      final_path: Path) -> Path:
        """Unpack `nbx_path` into `cache_path`, which is a staging directory.

        `final_path` is where the result will be visible once the caller swaps it
        in, and is used only for the lines this prints — a progress message naming
        a staging directory would send a reader looking for a path that is about
        to stop existing.
        """

        # SPRINT 0 - R0.1: Parallel extraction using ThreadPoolExecutor
        # I/O bound operation benefits from parallel workers
        #
        # OPTIMIZATION (January 2026):
        # - Buffer size increased from 64KB (default) to 8MB for large files
        # - 8MB buffer is optimal for NVMe/SSD sequential writes
        # - Small files (<1MB) use smaller buffer to avoid memory waste
        LARGE_FILE_THRESHOLD = 1 * 1024 * 1024  # 1MB
        LARGE_BUFFER_SIZE = 8 * 1024 * 1024     # 8MB for safetensors
        SMALL_BUFFER_SIZE = 256 * 1024          # 256KB for small files

        # ONE ZipFile PER THREAD. Eight workers reading members through a
        # single shared ZipFile race on the underlying handle's seek/read —
        # CPython's zipfile does not support concurrent member reads on one
        # object — and the corruption is real, not theoretical: Sana 4Kpx's
        # transformer shard_001 failed its CRC-32 twice through this pool
        # (2026-09-21, over NFS, where latency widens the window) while a
        # single-reader `unzip -t` of the same member over the same mount
        # passed. The CRC caught it; the race owned it.
        _tl = threading.local()

        def _zf_for_thread():
            z = getattr(_tl, "zf", None)
            if z is None:
                z = zipfile.ZipFile(nbx_path, 'r')
                _tl.zf = z
                _all_zfs.append(z)
            return z

        _all_zfs = []
        with zipfile.ZipFile(nbx_path, 'r') as zf:
            members = zf.namelist()
            total = len(members)
            total_bytes = sum(zf.getinfo(m).file_size for m in members)

            # Thread-safe progress tracking
            progress_lock = threading.Lock()
            extracted_bytes = [0]  # Use list for mutable reference
            last_progress = [0]

            def extract_member(member: str) -> int:
                """Extract a single member and return its size."""
                # Security: validate against path traversal
                member_path = cache_path / member
                if not str(member_path.resolve()).startswith(str(cache_path.resolve())):
                    raise ValueError(f"Security: path traversal detected in archive member: {member}")
                if member.endswith('/'):
                    member_path.mkdir(parents=True, exist_ok=True)
                    return 0

                member_path.parent.mkdir(parents=True, exist_ok=True)

                # Get file size to choose optimal buffer
                zft = _zf_for_thread()
                file_size = zft.getinfo(member).file_size
                buffer_size = LARGE_BUFFER_SIZE if file_size > LARGE_FILE_THRESHOLD else SMALL_BUFFER_SIZE

                # Extract file with optimized buffer
                with zft.open(member) as src:
                    with open(member_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst, length=buffer_size)

                return file_size

            def update_progress(member: str, member_bytes: int):
                """Thread-safe progress update."""
                with progress_lock:
                    extracted_bytes[0] += member_bytes
                    progress = int((extracted_bytes[0] / total_bytes) * 100)
                    if progress >= last_progress[0] + 10 or 'safetensors' in member:
                        size_gb = extracted_bytes[0] / 1e9
                        print(f"[Cache] {progress}% ({size_gb:.2f}GB) - {member[:60]}...")
                        last_progress[0] = progress

            # Use 8 workers (I/O bound, not CPU bound)
            max_workers = min(8, len(members))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all extraction tasks
                future_to_member = {
                    executor.submit(extract_member, member): member
                    for member in members
                }

                # Process completed tasks
                for future in as_completed(future_to_member):
                    member = future_to_member[future]
                    try:
                        member_bytes = future.result()
                        update_progress(member, member_bytes)
                    except Exception as e:
                        print(f"[Cache] ERROR extracting {member}: {e}")
                        raise
            for _z in _all_zfs:
                try:
                    _z.close()
                except Exception:                      # noqa: BLE001
                    pass

        # Write cache metadata
        cache_meta = {
            "source": str(nbx_path.absolute()),
            "extracted_at": datetime.now().isoformat(),
            "file_count": total,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1e9, 2),
        }
        with open(cache_path / ".cache_meta.json", 'w') as f:
            json.dump(cache_meta, f, indent=2)

        print(f"[Cache] Done: {total} files, {total_bytes/1e9:.2f}GB extracted")
        # The naming door at extraction: the slot is keyed on the .nbx's parent directory
        # (`get_cache_path`), so a container built under a hand-chosen directory would install
        # under that name. Refused here, in staging, before the swap makes it visible.
        with open(cache_path / "manifest.json") as f:
            refuse_misnamed(final_path, json.load(f))
        return final_path

    def clear(self, model_name: Optional[str] = None):
        """Clear cache for a model or all models."""
        if model_name:
            cache_path = self.cache_dir / model_name
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"[Cache] Cleared: {model_name}")
            else:
                print(f"[Cache] Not found: {model_name}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print("[Cache] Cleared all")

    def list_cached(self) -> List[Dict[str, Any]]:
        """List all cached models with metadata."""
        if not self.cache_dir.exists():
            return []

        from neurobrix.nbx.atomic_install import is_install_artifact
        cached = []
        for d in self.cache_dir.iterdir():
            # A staging tree, a lock, or a tree renamed aside mid-swap is not a
            # cached model, however complete its contents look from outside.
            if d.is_dir() and not is_install_artifact(d.name):
                meta_path = d / ".cache_meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    cached.append({
                        "name": d.name,
                        "path": str(d),
                        **meta
                    })
                else:
                    cached.append({
                        "name": d.name,
                        "path": str(d),
                        "extracted_at": "unknown",
                    })

        return cached

    def get_size(self, model_name: Optional[str] = None) -> int:
        """Get cache size in bytes."""
        if model_name:
            cache_path = self.cache_dir / model_name
            if not cache_path.exists():
                return 0
            return sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        else:
            if not self.cache_dir.exists():
                return 0
            return sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())


# Global instance
_cache = NBXCache()


def get_cache() -> NBXCache:
    """Get global cache instance."""
    return _cache


def refuse_misnamed(cache_path, manifest: dict) -> None:
    """A container's directory carries its manifest's model name, or the engine refuses it.

    The name of a model is the name of its Hugging Face repository — never invented, never
    hand-suffixed, never renamed — and two traces of one repository under two names are a
    duplication whatever the graphs (the owner, 2026-09-26). The shared cache held four such
    directories (`PixArt-XL-1024` over a manifest declaring `PixArt-XL-2-1024-MS`, the Sigma
    and Sana pairs, a `.pre-G-backup`), each a second trace nobody could tell from the first
    by name. A census says "not this time"; this door says "never": a directory whose name is
    not the name its manifest declares is refused at every entry — extraction, the directory
    short-cut, and the runtime loader — so the state cannot be reached by any path.
    """
    declared = manifest.get("model_name")
    if not declared:
        raise RuntimeError(
            f"ZERO FALLBACK: the manifest at {Path(cache_path) / 'manifest.json'} declares no "
            f"model_name; a container without its model's name cannot be told from another.")
    actual = Path(cache_path).name
    if actual != declared:
        raise RuntimeError(
            f"NeuroBrix refuses the container at {cache_path}: its directory is named "
            f"{actual!r} but its manifest declares model_name {declared!r}. A container carries "
            f"its model's name — the Hugging Face repository's — and one repository under two "
            f"names is a duplicate, not two models. Rename the directory to {declared!r}, or "
            f"remove it if {declared!r} already exists (`neurobrix remove {actual}`).")


def ensure_extracted(nbx_path: Path) -> Path:
    """Ensure NBX is extracted and return cache path.

    Handles both:
    - .nbx ZIP files: Extract to cache and return cache path
    - Already extracted directories: Validate and return directly
    """
    nbx_path = Path(nbx_path)

    # If it's a directory with manifest.json, it's already extracted
    if nbx_path.is_dir():
        manifest_path = nbx_path / "manifest.json"
        if manifest_path.exists():
            import json as _json
            refuse_misnamed(nbx_path, _json.loads(manifest_path.read_text()))
            return nbx_path
        else:
            raise FileNotFoundError(
                f"Directory '{nbx_path}' is not a valid NBX cache (no manifest.json)"
            )

    # Otherwise, extract the .nbx file
    return _cache.extract(nbx_path)
