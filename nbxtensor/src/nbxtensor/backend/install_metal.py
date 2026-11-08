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

#: the Metal wheels per CPython tag: (URL of the release asset, sha256 of the file), Triton first; a tag absent here is
#: refused by name. The fork's release apple-v0.1.0+triton3.8.0 (2026-11-07): Triton 3.8.0 (git 4a15f415) built with
#: TRITON_EXT_ENABLED=1 and nbx-triton-apple 0.1.0, both macOS 15.0 arm64 (the prebuilt LLVM Triton links is minos 15.0);
#: sha256 read on the downloaded assets.
METAL_WHEELS: dict = {
    "cp312": (
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0/triton-3.8.0%2Bgit4a15f415.nbx1-cp312-cp312-macosx_15_0_arm64.whl",
         "b9084dc93ae7f1468062a03e713d49c9add9a1c9a609fae753a7b32a13b7a223"),
        (RELEASES + "apple-v0.1.0%2Btriton3.8.0/nbx_triton_apple-0.1.0-cp312-cp312-macosx_15_0_arm64.whl",
         "b9ed95c983b220b1fbf1300481b1e2f52a0435c10ee618ef7ccf60a132fe263c"),
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
    entries = list(METAL_WHEELS[tag])
    need = max(_min_macos(url) for url, _sha in entries)
    have = _macos_version()
    if have < need:
        raise InstallMetalRefused(
            f"macOS {'.'.join(map(str, need))} required for now: the Metal wheels of nbxtensor {version} are built for "
            f"macOS {'.'.join(map(str, need))} and later (this Mac: macOS {'.'.join(map(str, have))}).")
    return entries


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
