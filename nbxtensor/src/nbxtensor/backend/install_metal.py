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

import os
import platform
import subprocess
import sys
import tempfile
from typing import List, Tuple

#: the only place the wheels are fetched from: the releases of the project's fork of triton-ext
RELEASES = "https://github.com/NeuroBrix/triton-ext/releases/download/"

#: the Metal wheels per CPython tag: (URL of the release asset, sha256 of the file), Triton first. Filled from the
#: release the fork publishes; a tag absent here is refused by name.
METAL_WHEELS: dict = {}


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
    return list(METAL_WHEELS[tag])


def pip_command(requirements: str) -> List[str]:
    """pip on this interpreter, from a requirements file of URLs and hashes only: no index is read, nothing else."""
    return [sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes", "--no-index", "-r", requirements]


def requirements_text(entries: List[Tuple[str, str]]) -> str:
    return "".join(f"{url} --hash=sha256:{sha}\n" for url, sha in entries)


def main(version: str, dry_run: bool = False) -> int:
    try:
        entries = wheels(version)
    except InstallMetalRefused as exc:
        print(f"install-metal: {exc}", file=sys.stderr)
        return 1
    text = requirements_text(entries)
    if dry_run:
        print(text, end="")
        print(" ".join(pip_command("<these lines>")))
        return 0
    fd, path = tempfile.mkstemp(prefix="nbxtensor-metal-", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return subprocess.call(pip_command(path))
    finally:
        os.unlink(path)
