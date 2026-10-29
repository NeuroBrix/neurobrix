#!/usr/bin/env python3
"""Split a vendor's files into the LIBRARY's device files and the ENGINE's vendor files, on one key list
(`docs/nbxtensor/device_keys.txt`, the device-key census) — Hocine 2026-10-29 16:28, API.md section 11.

nbxtensor reads the kernel-relevant description of a device (tiles, shared memory, registers, warp size, the matrix
unit, the dispatch limits, the identification a profile is matched by); NeuroBrix reads the rest (Prism's measured
rates and budgets, the PyTorch branch's allocator figures). A vendor file held both. This moves, line by line and with
the comments written above each key (a value's source is part of the value), every key the list names:

- a top-level key the list names WITHOUT naming any of its children moves whole (a section a reader takes whole);
- a section whose children the list names moves those children (a child named through a deeper path moves whole);
  the section's header is written in both files — unless every child moves, and then the section moves whole;
- everything else stays in the engine's file.

The proof is the tool's own: each file is written only when the two halves, loaded, merge back to EXACTLY the
original (no key in both, no value changed or lost); otherwise nothing is written and the file is named. A vendor
already split takes the keys the list gained: they are appended to its library file, and the proof is then that the
new halves merge back to the old ones.

    python3 tools/split_vendor_files.py --vendor apple --keys docs/nbxtensor/device_keys.txt \\
        --engine src/neurobrix/config/vendors --library nbxtensor/src/nbxtensor/backend/devices [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

TOP = re.compile(r"^([A-Za-z_][\w\-]*)\s*:")
SUB = re.compile(r"^  ([A-Za-z_][\w\-]*)\s*:")


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _blocks(lines, lo, hi, pattern):
    """[(key, start, key_line, end)] of the keys `pattern` matches in lines[lo:hi]; a block starts at the comment
    lines written right above its key (no blank line between) and ends where the next one starts."""
    keys = [(i, pattern.match(lines[i]).group(1)) for i in range(lo, hi) if pattern.match(lines[i])]
    starts = []
    for i, _ in keys:
        s = i
        while s - 1 >= lo and _is_comment(lines[s - 1]):
            s -= 1
        starts.append(s)
    return [(k, starts[n], i, starts[n + 1] if n + 1 < len(keys) else hi) for n, (i, k) in enumerate(keys)]


def split(text: str, keys: set):
    """(library text, engine text, moved key paths) of one vendor file."""
    lines = text.splitlines(keepends=True)
    lib, eng, moved = [], [], []
    tops = _blocks(lines, 0, len(lines), TOP)
    eng.extend(lines[:tops[0][1]] if tops else lines)
    for top, start, at, end in tops:
        children = {k[len(top) + 1:].split(".")[0] for k in keys if k.startswith(top + ".")}
        if top in keys and not children:
            lib.extend(lines[start:end])
            moved.append(top)
            continue
        if not children:
            eng.extend(lines[start:end])
            continue
        subs = _blocks(lines, at + 1, end, SUB)
        if subs and all(b[0] in children for b in subs):          # every child moves: the section moves whole
            lib.extend(lines[start:end])
            moved.append(top)
            continue
        head_end = subs[0][1] if subs else end
        eng.extend(lines[start:head_end])
        took = [b for b in subs if b[0] in children]
        if took:
            lib.append(lines[at] if lines[at].endswith("\n") else lines[at] + "\n")
        for sub, s, _i, e in subs:
            (lib if sub in children else eng).extend(lines[s:e])
            if sub in children:
                moved.append(f"{top}.{sub}")
    return "".join(lib), "".join(eng), moved


def _join(held: str, new: str) -> str:
    """`new` (moved blocks) joined to a library file already written: a section the file already has takes the new
    children at its end (a second header of the same name would REPLACE the first when loaded); any other block is
    appended."""
    lines = held.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    new_lines = new.splitlines(keepends=True)
    tail = []
    for top, start, at, end in _blocks(new_lines, 0, len(new_lines), TOP):
        mine = [b for b in _blocks(lines, 0, len(lines), TOP) if b[0] == top]
        if mine:
            _, _, _, hend = mine[0]
            body = new_lines[at + 1:end]
            lines[hend:hend] = body
        else:
            tail.extend(new_lines[start:end])
    out = "".join(lines)
    return out + ("\n" + "".join(tail) if tail else "")


def _merge(a: dict, b: dict, where=""):
    out = dict(a)
    for k, v in b.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v, f"{where}{k}.")
            else:
                raise ValueError(f"{where}{k} is in both halves")
        else:
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--check", action="store_true", help="prove every file, write nothing")
    a = ap.parse_args(argv)
    keys = {k.strip() for k in Path(a.keys).read_text().split() if k.strip()}
    src, dst = Path(a.engine) / a.vendor, Path(a.library) / a.vendor
    files = sorted(src.glob("*.yml"))
    if not files:
        raise SystemExit(f"{src}: no vendor file")
    plans, bad = [], []
    for f in files:
        text = f.read_text()
        lib, eng, moved = split(text, keys)
        held = dst / f.name
        held_text = held.read_text() if held.is_file() else ""      # a split already made: the new keys join it
        if held_text and lib:
            lib = _join(held_text, lib)
        elif held_text:
            lib = held_text
        try:
            whole = _merge(yaml.safe_load(held_text) or {}, yaml.safe_load(text) or {})
            back = _merge(yaml.safe_load(lib) or {}, yaml.safe_load(eng) or {})
            if back != whole:
                raise ValueError("the halves do not merge back to the original")
        except Exception as e:  # noqa: BLE001 — a file that cannot be proven is named and nothing is written
            bad.append(f"{f}: {e}")
            continue
        plans.append((f, lib, eng, moved))
    for line in bad:
        print(f"REFUSED {line}")
    if bad:
        return 1
    for f, lib, eng, moved in plans:
        print(f"{f.name}: {len(moved)} key(s) to the library ({', '.join(moved)})")
        if a.check:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        if (dst / f.name).is_file():
            (dst / f.name).write_text(lib)                           # the note is already at its head
            f.write_text(eng)
            continue
        note = (f"# The library's half of {a.vendor}/{f.name}: the keys nbxtensor reads (docs/nbxtensor/device_keys.txt),\n"
                f"# moved with their sources by tools/split_vendor_files.py. NeuroBrix's half stays in\n"
                f"# src/neurobrix/config/vendors/{a.vendor}/{f.name}; the engine reads the two merged.\n\n")
        (dst / f.name).write_text(note + lib)
        f.write_text(eng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
