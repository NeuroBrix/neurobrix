"""`python -m nbxtensor install-metal`: install Triton for the Apple GPU — the project's own wheels (Triton built with
TRITON_EXT_ENABLED=1, and triton-ext's AppleGPU backend as `nbx-triton-apple`), downloaded from the release of the
project's fork, github.com/NeuroBrix/triton-ext. The AppleGPU backend is imperatormk's (triton-lang/triton-ext#126).

PyPI publishes no Triton for macOS, so a Mac has no Triton after `pip install nbxtensor`. The wheels are fetched from
the URLs written below, each checked against the sha256 written beside it (`pip --require-hashes`), with `--no-deps`:
never through an extra index and never by a package name pip would look up on PyPI — anyone can publish a package
named "triton-apple-backend" there (dependency confusion). A machine other than an Apple-silicon Mac, or a Python the
table does not cover, is refused by name.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
from typing import List, Tuple

#: the only place the wheels are fetched from: the releases of the project's fork of triton-ext
RELEASES = "https://github.com/NeuroBrix/triton-ext/releases/download/"

#: the Metal wheels per CPython tag: (URL of the release asset, sha256 of the file), Triton first within each build;
#: a tag absent here is refused by name. Two releases of the project's fork: apple-v0.1.0+triton3.8.0 (2026-11-07),
#: Triton 3.8.0 (git 4a15f415) built with TRITON_EXT_ENABLED=1 and nbx-triton-apple 0.1.0 for CPython 3.12, macOS 15.0
#: arm64; apple-v0.1.0+triton3.8.0-macos14 (2026-11-09), the same sources for CPython 3.10-3.13, macOS 14.0 arm64. A Mac
#: on macOS 15 or later under CPython 3.12 installs the 15.0 build, any other covered Mac the 14.0 build (`wheels`).
#: sha256 read on the downloaded assets (each matches the release's SHA256SUMS).
METAL_WHEELS: dict = {
    "cp310": (
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/triton-3.8.0%2Bgit4a15f415.nbx1-cp310-cp310-macosx_14_0_arm64.whl",
         "77d9c1e43c77a490264ad810a0a624f15963b120eff96384bf2c921748352f8c"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/nbx_triton_apple-0.1.0-cp310-cp310-macosx_14_0_arm64.whl",
         "4ac872b376f113e40c15772aa81a3c95b2e83a412f4bbfb12211129cec03b284"),
    ),
    "cp311": (
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/triton-3.8.0%2Bgit4a15f415.nbx1-cp311-cp311-macosx_14_0_arm64.whl",
         "8af8dbce2acb4d32de26b3e745696bd52fea6f54723fe96cb229c9feee70668c"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/nbx_triton_apple-0.1.0-cp311-cp311-macosx_14_0_arm64.whl",
         "fd6a9e309d0f37c719d33e921267dc6413edd406920055fed1563ad942062148"),
    ),
    "cp312": (
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0/triton-3.8.0%2Bgit4a15f415.nbx1-cp312-cp312-macosx_15_0_arm64.whl",
         "b9084dc93ae7f1468062a03e713d49c9add9a1c9a609fae753a7b32a13b7a223"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0/nbx_triton_apple-0.1.0-cp312-cp312-macosx_15_0_arm64.whl",
         "b9ed95c983b220b1fbf1300481b1e2f52a0435c10ee618ef7ccf60a132fe263c"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/triton-3.8.0%2Bgit4a15f415.nbx1-cp312-cp312-macosx_14_0_arm64.whl",
         "daad78ec3e8a8cfe28dc531f9372aaf7d5a16a347b59d2df2759866f78c7c78f"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/nbx_triton_apple-0.1.0-cp312-cp312-macosx_14_0_arm64.whl",
         "f09c969635255f67350a3ca4f9ef2fa416934193cea3dd5b2b930ff9d78c6bab"),
    ),
    "cp313": (
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/triton-3.8.0%2Bgit4a15f415.nbx1-cp313-cp313-macosx_14_0_arm64.whl",
         "4099a5199cc4b0114aae75a325a139acdc737b2d7ed56493c27a3ca4e7a5a703"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0-macos14/nbx_triton_apple-0.1.0-cp313-cp313-macosx_14_0_arm64.whl",
         "21d96a90e3d74c6bd4a478610de4135a2fb71bed245f10684ba5eab7829528c7"),
    ),
}


class InstallMetalRefused(RuntimeError):
    """This machine or this Python has no Metal wheels in this version's table."""


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def wheels(version: str) -> List[Tuple[str, str]]:
    """[(url, sha256)] of the wheels this machine installs, or a named refusal."""
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        raise InstallMetalRefused(
            f"install-metal installs Triton for the Apple GPU and runs on an Apple-silicon Mac only (this machine: "
            f"{platform.system()} {platform.machine()}). Elsewhere Triton comes from PyPI with nbxtensor.")
    tag = _python_tag()
    if tag not in METAL_WHEELS:
        covered = ", ".join(sorted(METAL_WHEELS)) or "none yet"
        raise InstallMetalRefused(
            f"nbxtensor {version} carries no Metal wheels for Python {sys.version_info.major}.{sys.version_info.minor} "
            f"({tag}); covered: {covered}.")
    # A tag may carry one build per minimum macOS (the wheels' platform tag, e.g. macosx_15_0 and macosx_14_0): the
    # Mac installs the NEWEST build its macOS satisfies; one too old for every build is refused by name.
    builds: dict = {}
    for url, sha in METAL_WHEELS[tag]:
        builds.setdefault(_min_macos(url), []).append((url, sha))
    have = _macos_version()
    fits = [need for need in builds if have >= need]
    if not fits:
        need = min(builds)
        raise InstallMetalRefused(
            f"macOS {'.'.join(map(str, need))} required for now: the Metal wheels of nbxtensor {version} are built for "
            f"macOS {'.'.join(map(str, need))} and later (this Mac: macOS {'.'.join(map(str, have))}).")
    return builds[max(fits)]


def _min_macos(url: str) -> Tuple[int, int]:
    """The lowest macOS a wheel installs on, read from its platform tag (`macosx_15_0_arm64` -> (15, 0))."""
    m = re.search(r"-macosx_(\d+)_(\d+)_arm64\.whl$", url)
    if m is None:
        raise InstallMetalRefused(f"not a macOS arm64 wheel: {url}")
    return int(m.group(1)), int(m.group(2))


def _macos_version() -> Tuple[int, int]:
    parts = [int(p) for p in (platform.mac_ver()[0] or "0").split(".")[:2]]
    return (parts + [0])[0], (parts + [0])[1]


def pip_command(requirements: str) -> List[str]:
    """pip on this interpreter, from a requirements file of URLs and hashes only: no index is read, nothing else."""
    return [sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes", "--no-index", "-r", requirements]


def _wheel_identity(url: str) -> Tuple[str, str]:
    """(distribution, version) a wheel installs, read from its file name (PEP 427: `{name}-{version}-...whl`, the
    version's `+` URL-quoted in a release asset)."""
    from urllib.parse import unquote
    name, version = unquote(url.rsplit("/", 1)[1]).split("-")[:2]
    return name.replace("_", "-"), version


def _installed_version(dist: str):
    """The version of `dist` installed in this interpreter, or None."""
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def requirements_text(entries: List[Tuple[str, str]]) -> str:
    return "".join(f"{url} --hash=sha256:{sha}\n" for url, sha in entries)


def main(version: str, dry_run: bool = False) -> int:
    try:
        entries = wheels(version)
    except InstallMetalRefused as exc:
        print(f"install-metal: {exc}", file=sys.stderr)
        return 1
    wanted = [_wheel_identity(url) for url, _sha in entries]
    if all(_installed_version(d) == v for d, v in wanted):
        # the pinned wheels are this interpreter's already: nothing is downloaded (131 MB on every call before)
        print(f"Triton for Metal already installed ({', '.join(f'{d} {v}' for d, v in wanted)}).")
        return 0
    text = requirements_text(entries)
    if dry_run:
        print(text, end="")
        print(" ".join(pip_command("<these lines>")))
        return 0
    fd, path = tempfile.mkstemp(prefix="nbxtensor-metal-", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        print(f"Installing Triton for Metal ({', '.join(f'{d} {v}' for d, v in wanted)})...")
        # pip's own Collecting/Downloading lines only with NBX_LOG=debug; its progress bar stays
        quiet = [] if logging.getLogger(__name__).isEnabledFor(logging.DEBUG) else ["-q"]
        cmd = pip_command(path)
        cmd[4:4] = quiet                    # after `pip install`, the requirements file last
        return subprocess.call(cmd)
    finally:
        os.unlink(path)
