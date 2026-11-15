"""build_processor.py <name> --out <folder> [--source-root <folder>] [--data-from <folder>]: ONE processor artifact
(`<name>-<version>.nbxp`) from its source sub-package `src/neurobrix/processors/<name>/` and its pinned data.

The artifact is a plain zip, deterministic: members sorted by name, every member stored (no compression, so no zlib
version changes a byte), one fixed time and one fixed mode for every member. The same sources and pins give the same
sha256 on any host.

    processor.json     the source's processor.json, plus `members`: the sha256 of every other member
    code/              our code, the source's `code.source` folder, with the sub-package's LICENSE (`code.licence`)
    data/              the pinned files, unchanged, under their pinned relative paths; one of them is the licence of
                       the tree (`data.licence_file`, licence `data.licence`)
    EQUALITY.md        the sub-package's equality report

The source's processor.json (every key named, none defaulted):

    name, version, kind, interface, entry ("module:Class" of the code tree), variants,
    code: {source: <folder of the sub-package>, licence: <SPDX id>},
    data: {licence: <SPDX id>, licence_file: <a pinned path>,
           files: [{path, url, revision, sha256, size}, ...]}

The code tree is Python source only, and imports nothing but the standard library and itself (relative imports):
any other import, a dynamic import, or a file that is not Python source is refused by file, line and name, before a
byte is fetched. A pinned file is fetched from its url (or read from `--data-from <folder>/<path>`) and refused unless
its size and sha256 are the pinned ones. Every fetched or read pin is said on stderr (path, sha256, where from); the
result (name, version, file, size, sha256) on stdout. No processor is named here: the name is the argument.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from neurobrix.processors.declaration import ARTIFACT_SUFFIX  # noqa: E402

#: where the source sub-packages are
SOURCE_ROOT = REPO / "src" / "neurobrix" / "processors"
#: every member's time (the zip format's first representable instant) and mode (a plain readable file)
MEMBER_TIME = (1980, 1, 1, 0, 0, 0)
MEMBER_MODE = 0o100644
#: the sub-package's files the artifact carries beside its trees
CODE_LICENCE_FILE = "LICENSE"
EQUALITY_FILE = "EQUALITY.md"
#: how long a pinned file's host may stay silent before the build is refused
FETCH_TIMEOUT_SECONDS = 60

_TEXT = ("name", "version", "kind", "entry")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")


class Refused(Exception):
    """The build cannot give the artifact: `str()` says what is wrong and where."""


def _need(rec: dict, key: str, kind, where: str):
    value = rec.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) or (kind is not int and not value):
        raise Refused(f"{where} has no {kind.__name__} '{key}'")
    return value


def _relative(path: str, where: str) -> str:
    """A pinned or source path: relative, posix, no '.' or '..' part."""
    p = PurePosixPath(path)
    if p.is_absolute() or "\\" in path or any(part in ("", ".", "..") for part in path.split("/")):
        raise Refused(f"{where} names the path {path!r}, which is not a plain relative path")
    return str(p)


def read_source(sub: Path) -> dict:
    """The sub-package's processor.json, every key it needs present."""
    where = f"{sub.name}/processor.json"
    if not (sub / "processor.json").is_file():
        raise Refused(f"there is no {where}: no processor source at {sub}")
    rec = json.loads((sub / "processor.json").read_text(encoding="utf-8"))
    for key in _TEXT:
        _need(rec, key, str, where)
    if rec["name"] != sub.name:
        raise Refused(f"{where} names the processor {rec['name']!r}, its folder is {sub.name!r}")
    if _need(rec, "interface", int, where) <= 0:
        raise Refused(f"{where} has an 'interface' that is not a positive integer")
    if not _ENTRY.match(rec["entry"]):
        raise Refused(f"{where} has an 'entry' that is not 'module:Class'")
    variants = _need(rec, "variants", list, where)
    if not all(isinstance(v, str) and v for v in variants):
        raise Refused(f"{where} has a 'variants' that is not a list of texts")
    code = _need(rec, "code", dict, where)
    _relative(_need(code, "source", str, f"{where} code"), f"{where} code")
    _need(code, "licence", str, f"{where} code")
    data = _need(rec, "data", dict, where)
    _need(data, "licence", str, f"{where} data")
    files = _need(data, "files", list, f"{where} data")
    paths = []
    for i, pin in enumerate(files):
        at = f"{where} data.files[{i}]"
        if not isinstance(pin, dict):
            raise Refused(f"{at} is not a record")
        paths.append(_relative(_need(pin, "path", str, at), at))
        for key in ("url", "revision"):
            _need(pin, key, str, at)
        if not _SHA256.match(_need(pin, "sha256", str, at)):
            raise Refused(f"{at} has a 'sha256' that is not 64 lower-case hex digits")
        if _need(pin, "size", int, at) <= 0:
            raise Refused(f"{at} has a 'size' that is not a positive integer")
    if len(set(paths)) != len(paths):
        raise Refused(f"{where} pins the same data path twice")
    if _need(data, "licence_file", str, f"{where} data") not in paths:
        raise Refused(f"{where} data.licence_file {data['licence_file']!r} is not one of the pinned files")
    if "members" in rec:
        raise Refused(f"{where} carries 'members': the build writes it, the source never does")
    return rec


def _module_of(call: ast.Call):
    """The name a dynamic import call asks for, or None when the call is not one."""
    f = call.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    return name if name in ("__import__", "import_module") else None


def non_standard_imports(path: Path, shown: str) -> list:
    """Every import of `path` that is neither the standard library nor relative, as 'file:line name'."""
    tree = ast.parse(path.read_bytes(), filename=shown)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"{shown}:{node.lineno} imports {a.name}" for a in node.names
                    if a.name.split(".")[0] not in sys.stdlib_module_names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                and node.module.split(".")[0] not in sys.stdlib_module_names:
            out.append(f"{shown}:{node.lineno} imports {node.module}")
        elif isinstance(node, ast.Call) and _module_of(node):
            out.append(f"{shown}:{node.lineno} imports by {_module_of(node)}() (a dynamic import)")
    return out


def code_members(sub: Path, rec: dict) -> dict:
    """`code/<relative path>` -> bytes: the code tree, Python source of the standard library only, and its licence."""
    source = sub / rec["code"]["source"]
    if not source.is_dir():
        raise Refused(f"{sub.name}/processor.json code.source {rec['code']['source']!r} is not a folder of {sub.name}")
    licence = sub / CODE_LICENCE_FILE
    if not licence.is_file():
        raise Refused(f"{sub.name} has no {CODE_LICENCE_FILE}: the code tree's licence ({rec['code']['licence']}) "
                      f"is stated by that file")
    members, refused = {f"code/{CODE_LICENCE_FILE}": licence.read_bytes()}, []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if "__pycache__" in rel.parts or path.is_dir():
            continue
        shown = f"{sub.name}/{rec['code']['source']}/{rel.as_posix()}"
        if path.is_symlink() or path.suffix != ".py":
            refused.append(f"{shown} is not Python source (the code tree holds only .py files)")
            continue
        refused += non_standard_imports(path, shown)
        members[f"code/{rel.as_posix()}"] = path.read_bytes()
    if refused:
        raise Refused("the code tree is the standard library only: " + "; ".join(refused))
    module = rec["entry"].split(":")[0].replace(".", "/")
    if f"code/{module}.py" not in members and f"code/{module}/__init__.py" not in members:
        raise Refused(f"{sub.name}/processor.json entry {rec['entry']!r}: the code tree has no module {module}")
    return members


def _fetch(pin: dict, data_from) -> tuple:
    """The pinned file's bytes and where they came from."""
    if data_from is not None:
        path = Path(data_from) / pin["path"]
        if not path.is_file():
            raise Refused(f"the pinned file {pin['path']} is not in {data_from}")
        return path.read_bytes(), str(path)
    try:
        with urllib.request.urlopen(pin["url"], timeout=FETCH_TIMEOUT_SECONDS) as reply:
            return reply.read(), pin["url"]
    except OSError as exc:
        raise Refused(f"the pinned file {pin['path']} could not be fetched from {pin['url']}: {exc}") from exc


def data_members(rec: dict, data_from=None) -> dict:
    """`data/<pinned path>` -> bytes: each pinned file, its size and sha256 the pinned ones."""
    members = {}
    for pin in rec["data"]["files"]:
        body, origin = _fetch(pin, data_from)
        sha = hashlib.sha256(body).hexdigest()
        if len(body) != pin["size"] or sha != pin["sha256"]:
            raise Refused(f"the pinned file {pin['path']} from {origin} is {len(body)} bytes, sha256 {sha}; the pin "
                          f"says {pin['size']} bytes, sha256 {pin['sha256']}")
        print(f"pinned {pin['path']} {sha} from {origin}", file=sys.stderr)
        members[f"data/{pin['path']}"] = body
    return members


def deterministic_zip(members: dict) -> bytes:
    """`members` (name -> bytes) as a zip whose bytes depend on the names and bytes alone."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, MEMBER_TIME)
            info.external_attr = MEMBER_MODE << 16
            info.create_system = 3
            zf.writestr(info, members[name], compress_type=zipfile.ZIP_STORED)
    return out.getvalue()


def build(name: str, out: Path, source_root: Path = SOURCE_ROOT, data_from=None) -> Path:
    """Write `<name>-<version>.nbxp` into `out` and give its path."""
    sub = Path(source_root) / name
    rec = read_source(sub)
    if not (sub / EQUALITY_FILE).is_file():
        raise Refused(f"{name} has no {EQUALITY_FILE}: the artifact carries its equality report")
    members = code_members(sub, rec)
    members.update(data_members(rec, data_from))
    members[EQUALITY_FILE] = (sub / EQUALITY_FILE).read_bytes()
    own = dict(rec, members={m: hashlib.sha256(b).hexdigest() for m, b in sorted(members.items())})
    members["processor.json"] = (json.dumps(own, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    body = deterministic_zip(members)
    Path(out).mkdir(parents=True, exist_ok=True)
    dest = Path(out) / f"{name}-{rec['version']}{ARTIFACT_SUFFIX}"
    dest.write_bytes(body)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="the processor's sub-package name")
    ap.add_argument("--out", required=True, help="the folder the artifact is written to")
    ap.add_argument("--source-root", default=str(SOURCE_ROOT), help="the folder holding the sub-packages")
    ap.add_argument("--data-from", default=None, help="read the pinned files from this folder instead of their urls")
    args = ap.parse_args(argv)
    try:
        dest = build(args.name, Path(args.out), Path(args.source_root), args.data_from)
    except Refused as exc:
        print(f"build_processor: {exc}", file=sys.stderr)
        return 1
    body = dest.read_bytes()
    rec = json.loads(zipfile.ZipFile(io.BytesIO(body)).read("processor.json"))
    print(f"name {rec['name']}\nversion {rec['version']}\nfile {dest}\nsize {len(body)}\n"
          f"sha256 {hashlib.sha256(body).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
