"""THIRD_PARTY_NOTICES.md is the package's account of the code it took from other projects, and the wheel ships it.

What would this test do if the notice were wrong? A kernel file naming a source the notice does not list for it, a
listed file that no longer exists, a file citing its source under another license than the notice's (ops/rope.py said
"Liger-Kernel (Apache 2.0)"; the repository is BSD-2-Clause), a derived project without its license text, or a wheel
built without the notice: each fails here by name. The projects and their files are READ from the notice's own table.
"""
from __future__ import annotations

import pathlib
import re

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

PKG = pathlib.Path(__file__).resolve().parents[1]
NOTICE = PKG / "THIRD_PARTY_NOTICES.md"
SRC = PKG / "src" / "nbxtensor"

_ROW = re.compile(r"^\| \[(?P<repo>[^\]]+)\]\([^)]+\) \| (?P<license>[^|]+) \| (?P<files>.+) \|$")


def _norm(license_name: str) -> str:
    return re.sub(r"[\s-]+", "", license_name).lower()


def _derived():
    text = NOTICE.read_text()
    section = text.split("## Code derived from third-party projects", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        m = _ROW.match(line.strip())
        if m:
            files = set(re.findall(r"`([^`]+\.py)`", m["files"]))
            rows[m["repo"]] = (m["license"].strip(), files)
    assert rows, "the notice's derived-code table holds no row"
    return text, rows


def test_every_file_naming_a_derived_source_is_listed_under_it():
    _, rows = _derived()
    for repo, (_, listed) in rows.items():
        name = repo.split("/")[-1]
        citing = {str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if name in p.read_text()}
        missing = citing - listed
        assert not missing, f"{repo}: named in {sorted(missing)} but not listed in THIRD_PARTY_NOTICES.md"


def test_every_listed_file_exists():
    _, rows = _derived()
    for repo, (_, listed) in rows.items():
        gone = [f for f in listed if not (SRC / f).is_file()]
        assert not gone, f"{repo}: THIRD_PARTY_NOTICES.md lists {gone}, absent from src/nbxtensor"


def test_a_file_citing_its_source_cites_the_notices_license():
    _, rows = _derived()
    for repo, (license_name, listed) in rows.items():
        name = re.escape(repo.split("/")[-1])
        for f in listed:
            for cited in re.findall(name + r"[^()\n]{0,40}\(([^)]*)\)", (SRC / f).read_text()):
                head = cited.split(";")[0].replace("license", "").strip()
                if re.search(r"apache|mit|bsd", head, re.I):
                    assert _norm(head) == _norm(license_name), (
                        f"{f} cites {repo} under {head!r}; THIRD_PARTY_NOTICES.md (and the repository) says "
                        f"{license_name}")


def test_every_derived_project_carries_its_license_text():
    text, rows = _derived()
    for repo in rows:
        assert f"### {repo} " in text, f"{repo}: no license text in THIRD_PARTY_NOTICES.md"


def test_the_wheel_ships_the_notice():
    conf = tomllib.loads((PKG / "pyproject.toml").read_text())
    shipped = conf["tool"]["hatch"]["build"]["targets"]["wheel"].get("force-include", {})
    assert shipped.get("THIRD_PARTY_NOTICES.md") == "nbxtensor/THIRD_PARTY_NOTICES.md", (
        "the wheel is built without THIRD_PARTY_NOTICES.md (pyproject [tool.hatch.build.targets.wheel.force-include])")
