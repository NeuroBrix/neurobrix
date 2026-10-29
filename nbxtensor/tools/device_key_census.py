#!/usr/bin/env python3
"""The device-key census: every key of a vendor file the nbxtensor LIBRARY reads — the list that splits a vendor file
into the library's device file (`nbxtensor/backend/devices/<vendor>/<arch>.yml`) and the engine's policy file
(`config/vendors/<vendor>/<arch>.yml`), one key list for every vendor (API.md section 11).

Three passes, unioned:
- static: the library's code read with `ast` — `_arch_param("section", "key", ...)`, `vendor_param(v, a, "section",
  "key", ...)`, and chains `.get("a").get("b")` / `["a"]["b"]` on a vendor configuration;
- named: every key of a vendor file (top level, or a section's key) whose name the library's code spells as a string
  literal — a reader the tests do not reach, or one that copies a section before reading it;
- dynamic: the keys the library actually read while a test run went through it (`device_key_trace.py`).

    python3 nbxtensor/tools/device_key_census.py --vendors src/neurobrix/config/vendors --trace <trace file> \\
        --out docs/nbxtensor/device_keys.txt
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "src" / "nbxtensor"


def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _chain(node):
    """The literal key chain of `x.get("a", ..).get("b", ..)` / `x["a"]["b"]`, innermost first, and its root."""
    keys = []
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and node.args and _literal(node.args[0]) is not None:
            keys.append(_literal(node.args[0]))
            node = node.func.value
        elif isinstance(node, ast.Subscript) and _literal(node.slice) is not None:
            keys.append(_literal(node.slice))
            node = node.value
        elif isinstance(node, ast.BoolOp) and node.values:     # `(cfg.get("memory") or {})`
            node = node.values[0]
        else:
            return list(reversed(keys)), node


def static_keys() -> set:
    found = set()
    for p in sorted(LIB.rglob("*.py")):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                name = getattr(n.func, "id", getattr(n.func, "attr", ""))
                lits = [_literal(a) for a in n.args]
                if name == "_arch_param" and len(lits) >= 2 and lits[0] and lits[1]:
                    found.add(f"{lits[0]}.{lits[1]}")
                elif name in ("vendor_param", "_vendor_param") and len(lits) >= 4 and lits[2] and lits[3]:
                    found.add(f"{lits[2]}.{lits[3]}")
            if isinstance(n, (ast.Call, ast.Subscript)):
                keys, root = _chain(n)
                if not keys:
                    continue
                rootname = getattr(getattr(root, "func", root), "id", "") or getattr(getattr(root, "func", root), "attr", "")
                if any(t in str(rootname) for t in ("vendor_config", "get_vendor_config", "cfg", "_vendor_memory")):
                    found.add(".".join(keys[:2]))
    return found


def _vendor_files(vendors_dir: Path):
    """Every vendor file of both halves: the application's (`vendors_dir`) and the library's device files — a key the
    split already moved is still a key the library reads, so the census never shrinks as the split goes."""
    return sorted(vendors_dir.glob("*/*.yml")) + sorted((LIB / "backend" / "devices").glob("*/*.yml"))


def vendor_sections(vendors_dir: Path) -> set:
    """Every top-level key some vendor file carries (a static chain whose first key is none of them read another
    configuration — a launch config's `num_warps`, a call's `kwargs`)."""
    import yaml
    tops = set()
    for f in _vendor_files(vendors_dir):
        tops |= set((yaml.safe_load(f.read_text()) or {}).keys())
    return tops


def named_keys(vendors_dir: Path) -> set:
    """Vendor-file keys (`top` or `section.key`) whose NAME the library spells as a string literal."""
    import re
    import yaml
    text = "\n".join(p.read_text() for p in LIB.rglob("*.py"))
    spelled = lambda name: re.search(r"[\"']%s[\"']" % re.escape(name), text) is not None
    found = set()
    for f in _vendor_files(vendors_dir):
        for top, value in (yaml.safe_load(f.read_text()) or {}).items():
            subs = list(value) if isinstance(value, dict) else []
            found |= {f"{top}.{k}" for k in subs if spelled(str(k))}
            if not subs and spelled(str(top)):
                found.add(str(top))
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vendors", required=True, help="the vendor files' directory (config/vendors)")
    a = ap.parse_args(argv)
    tops = vendor_sections(Path(a.vendors))
    keys = {k for k in static_keys() if k.split(".")[0] in tops} | named_keys(Path(a.vendors))
    for t in a.trace:
        keys |= {line.strip() for line in Path(t).read_text().splitlines() if line.strip()}
    # `_vendor`, `_profile`, `_exact`: names the resolver adds to the profile it records, not keys of a file
    keys = {k for k in keys if not k.startswith("_")}
    Path(a.out).write_text("".join(f"{k}\n" for k in sorted(keys)))
    print(f"[device-key census] {len(keys)} keys -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
