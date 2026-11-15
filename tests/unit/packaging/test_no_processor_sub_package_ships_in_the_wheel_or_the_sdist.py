"""Nothing but the engine's generic processor files ships under neurobrix/processors/, in the wheel or in the sdist, and
no shipped path anywhere has an equality or data part, a vendor licence or a processor artifact.

A processor's source sub-package (`src/neurobrix/processors/<name>/`: its reader, its tests, its equality report) is
built into its own artifact (tools/build_processor.py) and fetched on demand; the vendor data it reads (GPL-3.0 for
one) never enters the engine's package. One pyproject line per target keeps the sub-packages out.

What this test does if the packaging is wrong: a target without the exclusion line - red at the first test; a
sub-package planted in a copy of the package's sources and built with this pyproject.toml shows up in the wheel or
the sdist - red at the second; this tree's own wheel or sdist (hatch's own file selection) would carry a refused path
- red at the third. The rule itself is held by the fourth. The builds need the build backend (hatchling, the
`build-system` of pyproject.toml); without it the two build tests are skipped by that name.

Run: python -m pytest tests/unit/packaging/test_no_processor_sub_package_ships_in_the_wheel_or_the_sdist.py
"""
from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLACE = "neurobrix/processors"
EXCLUSION = '"/src/neurobrix/processors/*/"'
SUFFIX = ".nbxp"
#: the files beside the package that the sdist and the readme hook read
PROJECT_FILES = ("pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", "NOTICE")


def refused(paths):
    """The shipped paths the rule refuses: a path under neurobrix/processors/ deeper than its generic files, a path
    with a `data` or `equality...` part anywhere, a vendor licence (COPYING) or a processor artifact."""
    out = []
    for raw in paths:
        p = PurePosixPath(raw.replace("\\", "/"))
        parts = p.parts
        names = [x.lower() for x in parts]
        rel = "/".join(parts)
        cut = rel.find(PLACE + "/")
        if cut >= 0 and "/" in rel[cut + len(PLACE) + 1:]:
            out.append(raw)
        elif any(x == "data" or x.startswith("equality") for x in names):
            out.append(raw)
        elif names and (names[-1].startswith("copying") or names[-1].endswith(SUFFIX)):
            out.append(raw)
    return sorted(out)


def _section(name):
    text = (ROOT / "pyproject.toml").read_text()
    start = text.index(f"[tool.hatch.build.targets.{name}]")
    end = text.find("\n[", start + 1)
    return text[start:end if end >= 0 else None]


def test_both_targets_keep_every_processor_sub_package_out():
    for target in ("wheel", "sdist"):
        assert EXCLUSION in _section(target), \
            f"pyproject.toml [tool.hatch.build.targets.{target}] does not exclude {EXCLUSION}: a processor's source " \
            f"sub-package would ship in the {target}"


def _builders():
    pytest.importorskip("hatchling.builders.wheel", reason="needs the build backend hatchling (pyproject.toml "
                                                           "build-system) to build the wheel and the sdist")
    from hatchling.builders.sdist import SdistBuilder
    from hatchling.builders.wheel import WheelBuilder
    return WheelBuilder, SdistBuilder


def _plant(root: Path) -> list:
    """A copy of the project files, the package marker and the generic processor files, with one processor
    sub-package planted in it; the planted files' paths."""
    for name in PROJECT_FILES:
        shutil.copyfile(ROOT / name, root / name)
    pkg = root / "src" / "neurobrix"
    (pkg / "processors").mkdir(parents=True)
    shutil.copyfile(ROOT / "src" / "neurobrix" / "__init__.py", pkg / "__init__.py")
    for f in (ROOT / "src" / PLACE).iterdir():
        if f.is_file():
            shutil.copyfile(f, pkg / "processors" / f.name)
    planted = {"planted/processor.json": "{}", "planted/LICENSE": "x", "planted/EQUALITY.md": "x",
               "planted/README.md": "x", "planted/reader/__init__.py": "", "planted/reader/words.py": "X = 1\n",
               "planted/tests/test_words.py": "", "planted/equality/grade.py": "", "planted/equality/corpus/a.txt": "x"}
    for rel, body in planted.items():
        (pkg / "processors" / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / "processors" / rel).write_text(body)
    return sorted(planted)


def test_a_planted_sub_package_is_left_out_of_the_built_wheel_and_sdist(tmp_path):
    WheelBuilder, SdistBuilder = _builders()
    root, out = tmp_path / "project", tmp_path / "dist"
    root.mkdir()
    _plant(root)
    wheel = Path(next(WheelBuilder(str(root)).build(directory=str(out), versions=["standard"])))
    sdist = Path(next(SdistBuilder(str(root)).build(directory=str(out), versions=["standard"])))
    with zipfile.ZipFile(wheel) as z:
        wheel_paths = z.namelist()
    with tarfile.open(sdist) as t:
        sdist_paths = [m.name for m in t.getmembers() if m.isfile()]
    for kind, paths in (("wheel", wheel_paths), ("sdist", sdist_paths)):
        assert refused(paths) == [], f"the built {kind} carries {refused(paths)}"
        assert any(p.endswith(f"{PLACE}/store.py") for p in paths), f"the built {kind} lost the generic files"


def test_this_trees_wheel_and_sdist_ship_nothing_the_rule_refuses():
    WheelBuilder, SdistBuilder = _builders()
    lists = {}
    for kind, builder in (("wheel", WheelBuilder), ("sdist", SdistBuilder)):
        lists[kind] = sorted(f.distribution_path for f in builder(str(ROOT)).recurse_included_files())
        assert refused(lists[kind]) == [], f"this tree's {kind} would carry {refused(lists[kind])}"
        assert any(p.endswith(f"{PLACE}/store.py") for p in lists[kind]), f"the {kind} lost the generic files"
    listing = os.environ.get("PACKAGING_LISTING_DIR")
    if listing:
        for kind, paths in lists.items():
            Path(listing, f"{kind}.selected.list").write_text("\n".join(paths) + "\n")


def test_the_rule_refuses_a_sub_package_and_data_anywhere_and_lets_the_generic_files_pass():
    generic = [f"{PLACE}/store.py", f"{PLACE}/README.md", "neurobrix/kernels/metadata_ops.py",
               "neurobrix-0.6.16/src/neurobrix/processors/__init__.py", "neurobrix-0.6.16.dist-info/METADATA"]
    bad = [f"{PLACE}/x/reader/words.py", "neurobrix-0.6.16/src/neurobrix/processors/x/README.md",
           "neurobrix/config/data/en_rules", "neurobrix/tools/equality/corpus.txt", "neurobrix/tools/EQUALITY.md",
           "neurobrix/config/COPYING", f"neurobrix/config/x-1.0.0{SUFFIX}"]
    assert refused(generic) == []
    assert refused(generic + bad) == sorted(bad)
