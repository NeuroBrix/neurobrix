"""The engine only slims: its tensor mechanics (slimming findings, scattered definitions) may only go down.

`tools/engine_slim_scan.py` lists what the engine carries that nbxtensor makes unnecessary, scanning the engine against
the INSTALLED library (the release pyproject.toml pins). Every finding is on the engine's own list,
`tools/engine_slim_list.tsv`, with the one sentence that says why it stays; the list is the output of `--write-list`.

1. A finding not on the list fails: something was written into the engine that the library makes unnecessary
   (delete it, or move it into nbxtensor).
2. A listed finding the scan no longer finds fails: it was resolved (`--write-list` deletes its line).
3. A finding listed without its reason fails.
4. The number of findings never exceeds the list's header, and the scattered tensor definitions
   (`tools/scatter_scan.py`) never exceed `tools/scatter_list.tsv`'s header. The engine's code-line count
   (`tools/brick_ratchet.code_lines`) is a measure, printed against the header, never a failure (ruling 17/11 16:11).
5. The list was scanned against the release the engine pins, read from the installed package, never from this tree.

What would this file do if the engine grew? A new dead or duplicated definition: rules 1 and 4 RED; a new scattered
definition: rule 4 RED; new plain lines: printed, green; a list regenerated against another release of the library: rule 5 RED."""
from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("engine_slim_scan", REPO / "tools" / "engine_slim_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
_scatter_spec = importlib.util.spec_from_file_location("scatter_scan", REPO / "tools" / "scatter_scan.py")
SC = importlib.util.module_from_spec(_scatter_spec)
_scatter_spec.loader.exec_module(SC)


@functools.lru_cache(maxsize=1)
def _found():
    return {(k, p, s) for k, p, s, _ in S.scan(REPO, S.installed_library())}


@functools.lru_cache(maxsize=1)
def _list():
    return S.read_list(S.LIST)


def _header(name: str) -> int:
    head = _list()[0]
    assert name in head, f"{S.LIST.name}: its header states no {name}"
    return int(head[name])


def test_every_finding_is_on_the_engines_list():
    new = sorted(_found() - set(_list()[1]))
    assert not new, ("Written into the engine though nbxtensor makes it unnecessary: delete it, or move it into the "
                     "library: " + "; ".join(f"{k} {p}:{s}" for k, p, s in new[:40]) + (" ..." if len(new) > 40 else ""))


def test_nothing_listed_is_already_resolved():
    gone = sorted(set(_list()[1]) - _found())
    assert not gone, (f"Resolved: run `python tools/engine_slim_scan.py --write-list` to delete their lines from "
                      f"{S.LIST.name}: " + "; ".join(f"{k} {p}:{s}" for k, p, s in gone[:40]))


def test_every_listed_finding_says_why_it_stays():
    mute = sorted(k for k, why in _list()[1].items() if not why.strip())
    assert not mute, f"listed without the reason it stays: {mute[:40]}"


def test_the_engine_and_its_findings_only_go_down():
    listed, lines = len(_list()[1]), S.engine_lines()
    print(f"engine code lines: {lines} (the list's header: {_header('engine_lines')}, a measure)")
    assert listed == _header("findings"), f"the list holds {listed} findings, its header says {_header('findings')}"
    new = sorted(_found() - set(_list()[1]))
    assert len(_found()) <= _header("findings"), (
        f"the engine holds {len(_found())} findings (the list's header: {_header('findings')}): it only slims; "
        f"new: " + "; ".join(f"{k} {p}:{s}" for k, p, s in new[:40]))
    head = SC.read_list()[0]
    assert "definitions" in head, f"{SC.LIST.name}: its header states no definitions count"
    scattered = sorted({(r[0], r[1]) for r in SC.scan()})
    assert len(scattered) <= int(head["definitions"]), (
        f"the engine holds {len(scattered)} scattered tensor definitions (the header of {SC.LIST.name}: "
        f"{head['definitions']}): it only shrinks; found: " + "; ".join(f"{p}:{s}" for p, s in scattered))


def test_the_list_was_scanned_against_the_installed_pinned_release():
    lib = S.installed_library()
    assert not lib.is_relative_to(REPO), lib
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    pin = next(re.fullmatch(r"nbxtensor==(\S+)", d.replace(" ", "")).group(1) for d in deps if re.match(r"nbxtensor\b", d))
    assert S.library_version(lib) == importlib.metadata.version("nbxtensor") == pin == _list()[0].get("nbxtensor"), (
        f"the scan read nbxtensor {S.library_version(lib)} at {lib}, pyproject pins {pin}, the list was scanned "
        f"against {_list()[0].get('nbxtensor')}: install the pinned release, then run --write-list")
