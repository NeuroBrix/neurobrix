"""The wheel's build: the crate compiled by its pinned toolchain (`cargo build --release --locked`) and its shared
library placed inside the package. The library is a C ABI loaded with ctypes, not a CPython extension, so one wheel per
platform serves every Python 3: tagged py3-none-<platform>. NBXGRAMMAR_CARGO_TARGET names a cross target (macOS x86_64
built on arm64: x86_64-apple-darwin); unset, the host's."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                                   # setuptools older than 70.1
    from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
LIB = {"darwin": "libnbxgrammar.dylib", "win32": "nbxgrammar.dll"}.get(sys.platform, "libnbxgrammar.so")


class BuildWithCrate(build_py):
    def run(self):
        target = os.environ.get("NBXGRAMMAR_CARGO_TARGET")
        cmd = ["cargo", "build", "--release", "--locked"] + (["--target", target] if target else [])
        # the library carries no path of the building machine, and two builds of one tree give one library: the
        # crate's root remapped in any embedded source path, and on macOS an @rpath install name (the default is the
        # absolute build path, which also moved the UUID and the signature from one directory to another)
        flags = [f"--remap-path-prefix={ROOT}=nbxgrammar"]
        if sys.platform == "darwin":
            flags.append(f"-Clink-arg=-Wl,-install_name,@rpath/{LIB}")
        env = dict(os.environ, RUSTFLAGS=" ".join(filter(None, [os.environ.get("RUSTFLAGS", "")] + flags)))
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
        built = ROOT / "target" / (target or "") / "release" / LIB
        shutil.copy2(built, ROOT / "python" / "nbxgrammar" / LIB)
        super().run()


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False                     # the library is a platform's

    def get_tag(self):
        plat = super().get_tag()[2]
        if sys.platform == "darwin":                  # the macOS the library is built for, not the build Python's
            major, minor = (os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0").split(".") + ["0"])[:2]
            target = os.environ.get("NBXGRAMMAR_CARGO_TARGET", "")
            arch = "x86_64" if target.startswith("x86_64") else "arm64" if target.startswith("aarch64") \
                else plat.rsplit("_", 1)[-1] if plat.endswith(("arm64", "x86_64")) else plat
            plat = f"macosx_{major}_{minor}_{'x86_64' if arch.endswith('x86_64') else 'arm64'}"
        return ("py3", "none", plat)                  # no CPython ABI is used: one wheel for every Python 3


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True                                   # the package carries a platform's library: platlib, not purelib


setup(cmdclass={"build_py": BuildWithCrate, "bdist_wheel": PlatformWheel}, distclass=BinaryDistribution,
      package_data={"nbxgrammar": [LIB]})
