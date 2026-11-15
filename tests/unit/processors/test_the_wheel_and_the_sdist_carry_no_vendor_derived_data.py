"""Builds the wheel and the sdist and lists them: under neurobrix/processors/ only code, README and LICENSE ship, and
no path carries equality data (the vendor-derived corpus and its stored outputs are GPL-3.0-or-later: outside the package)."""
import importlib.util
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = (".py",)
ALLOWED_NAMES = ("README", "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt")


def _bad(paths):
    out = []
    for p in paths:
        parts = p.replace("\\", "/")
        if "neurobrix/processors/" in parts:
            name = parts.rsplit("/", 1)[-1]
            if parts.endswith("/") or name.endswith(ALLOWED) or name in ALLOWED_NAMES:
                pass
            else:
                out.append(p)
        if parts.endswith("/"):
            continue
        # equality DATA: any path with an `equality` component that is not one of our .py tools
        if "equality" in parts.lower() and "neurobrix/processors/" in parts and not parts.endswith(".py") \
                and not parts.rsplit("/", 1)[-1].startswith("README"):
            out.append(p)
    return sorted(set(out))


def test_the_packaging_rule_names_a_data_file_and_lets_code_pass():
    ok = ["neurobrix/processors/espeak_ng/equality/grade.py", "neurobrix/processors/espeak_ng/README.md",
          "neurobrix/processors/espeak_ng/LICENSE"]
    assert _bad(ok) == []
    assert _bad(ok + ["neurobrix/processors/espeak_ng/equality/corpus/rules.txt"]) == [
        "neurobrix/processors/espeak_ng/equality/corpus/rules.txt"]


@pytest.mark.skipif(importlib.util.find_spec("build") is None, reason="needs the `build` package (pip install build)")
def test_the_wheel_and_the_sdist_carry_only_code_readme_and_licence_under_processors(tmp_path):
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", str(tmp_path), str(ROOT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    wheels, sdists = list(tmp_path.glob("*.whl")), list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1
    with zipfile.ZipFile(wheels[0]) as z:
        wheel_paths = z.namelist()
    with tarfile.open(sdists[0]) as t:
        sdist_paths = t.getnames()
    (tmp_path / "wheel.list").write_text("\n".join(wheel_paths))
    (tmp_path / "sdist.list").write_text("\n".join(sdist_paths))
    assert any(p.endswith("neurobrix/processors/espeak_ng/equality/grade.py") for p in wheel_paths + sdist_paths)
    assert _bad(wheel_paths) == [] and _bad(sdist_paths) == []
    import os
    d = os.environ.get("PACKAGING_LISTING_DIR")
    if d:
        Path(d, "wheel.list").write_text("\n".join(wheel_paths) + "\n")
        Path(d, "sdist.list").write_text("\n".join(sdist_paths) + "\n")
