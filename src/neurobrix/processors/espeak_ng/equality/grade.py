"""Grades a candidate reader against the stored reference: equal / total per layer and variant, every unequal line listed.

    python grade.py --candidate DIR [--archive FILE | --data-dir DIR] [--report EQUALITY.md]
    python grade.py --candidate DIR --reference DIR [--report EQUALITY.md]

The reference is the vendor stack's stored reading of the corpus. It is vendor-derived data (GPL-3.0-or-later) and
lives outside the repository in one archive named by `archive_pin.py` (name, size, sha256). The archive is given
with --archive, or looked up by the pin's name in --data-dir; it is refused by name when absent or when its size or
sha256 differs from the pin. --reference DIR reads an already unpacked folder (the held-out set, a test corpus).

Both folders have <variant>/<layer>.jsonl.gz (or .jsonl): one JSON row [text, units] per corpus line. A candidate
row that is missing, extra, or reads another text than the reference's counts as unequal. Equality is exact on the
unit sequence. Exit 0 only when every line of every layer is equal.
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile

LAYERS = ["in_list", "rules", "affixes", "numbers", "abbreviations", "symbols", "unknown", "stress_context"]


def read_rows(folder, variant, layer):
    for name, op in ((layer + ".jsonl.gz", gzip.open), (layer + ".jsonl", open)):
        p = os.path.join(folder, variant, name)
        if os.path.exists(p):
            with op(p, "rt", encoding="utf-8") as f:
                return [json.loads(ln) for ln in f.read().split("\n") if ln]
    return None


def grade_layer(ref, cand):
    """Returns (equal, total, unequal) where unequal lists (line number, text, reference units, candidate units)."""
    bad = []
    n = max(len(ref), len(cand) if cand is not None else 0)
    for i in range(len(ref)):
        want = ref[i][1]
        got = None if cand is None or i >= len(cand) or cand[i][0] != ref[i][0] else cand[i][1]
        if got != want:
            bad.append((i + 1, ref[i][0], want, got))
    if cand is not None and len(cand) > len(ref):
        for i in range(len(ref), len(cand)):
            bad.append((i + 1, cand[i][0], None, cand[i][1]))
    return len(ref) - sum(1 for b in bad if b[2] is not None), n, bad


PIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive_pin.py")


def load_pin(path=PIN):
    ns = {}
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)
    pin = ns.get("PIN", {})
    for k in ("name", "size", "sha256"):
        if k not in pin:
            raise SystemExit("the pin file %s has no key %r: it needs name, size and sha256" % (path, k))
    return pin


def check_archive(path, pin):
    """Refuses by name an archive that is absent or that is not the pinned one."""
    if not path or not os.path.isfile(path):
        raise SystemExit("the equality archive %s is absent: give --archive FILE or --data-dir DIR holding it" % (
            path or pin["name"]))
    if os.path.getsize(path) != pin["size"]:
        raise SystemExit("the equality archive %s has size %d, the pin says %d: fetch the pinned archive" % (
            path, os.path.getsize(path), pin["size"]))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    if h.hexdigest() != pin["sha256"]:
        raise SystemExit("the equality archive %s has sha256 %s, the pin says %s: fetch the pinned archive" % (
            path, h.hexdigest(), pin["sha256"]))


def unpack_reference(archive, into, pin=None):
    """Checks the archive against the pin, unpacks it into `into`, checks every file against its MANIFEST.sha256;
    returns the folder holding <variant>/<layer>.jsonl.gz."""
    pin = pin or load_pin()
    check_archive(archive, pin)
    z = subprocess.run(["zstd", "-dc", archive], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if z.returncode:
        raise SystemExit("the equality archive %s cannot be decompressed (zstd): %s" % (archive, z.stderr.decode()))
    t = subprocess.run(["tar", "-x", "-C", into], input=z.stdout, stderr=subprocess.PIPE)
    if t.returncode:
        raise SystemExit("the equality archive %s cannot be unpacked: %s" % (archive, t.stderr.decode()))
    root = os.path.join(into, os.path.basename(pin["name"])[: -len(".tar.zst")])
    man = os.path.join(root, "MANIFEST.sha256")
    if not os.path.isfile(man):
        raise SystemExit("the equality archive %s has no MANIFEST.sha256" % archive)
    for ln in open(man, encoding="utf-8").read().split("\n"):
        if ln:
            digest, rel = ln.split("  ", 1)
            with open(os.path.join(root, rel), "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != digest:
                    raise SystemExit("the equality archive file %s differs from its MANIFEST sha256" % rel)
    return os.path.join(root, "reference")


def grade(reference, candidate, variants=("us", "gb")):
    out = {}
    for v in variants:
        for layer in LAYERS:
            ref = read_rows(reference, v, layer)
            if ref is None:
                raise SystemExit("the reference has no %s/%s: build it first (build_reference.py)" % (v, layer))
            out[(v, layer)] = grade_layer(ref, read_rows(candidate, v, layer))
    return out


def report(result, title="Equality report"):
    lines = ["# %s" % title, "", "| variant | layer | equal | total | figure |", "|---|---|---|---|---|"]
    for (v, layer), (eq, tot, _) in result.items():
        lines.append("| %s | %s | %d | %d | %.2f %% |" % (v, layer, eq, tot, 100.0 * eq / tot if tot else 100.0))
    te = sum(r[0] for r in result.values())
    tt = sum(r[1] for r in result.values())
    lines += ["", "Overall: %d / %d lines equal." % (te, tt), ""]
    for (v, layer), (_, _, bad) in result.items():
        if bad:
            lines += ["## Unequal lines: %s / %s (%d)" % (v, layer, len(bad)), ""]
            for no, text, want, got in bad:
                lines.append("- line %d `%s`: vendor `%s` / reader `%s`" % (no, text, want, got))
            lines.append("")
    return "\n".join(lines) + "\n", te == tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", help="an unpacked reference folder (<variant>/<layer>.jsonl.gz)")
    ap.add_argument("--archive", help="the pinned equality archive file")
    ap.add_argument("--data-dir", help="folder holding the pinned archive by its pin name")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.reference:
        text, ok = report(grade(a.reference, a.candidate))
    else:
        pin = load_pin()
        archive = a.archive or (os.path.join(a.data_dir, pin["name"]) if a.data_dir else None)
        if archive is None:
            raise SystemExit("no reference: give --reference DIR, --archive FILE or --data-dir DIR")
        with tempfile.TemporaryDirectory() as tmp:
            text, ok = report(grade(unpack_reference(archive, tmp, pin), a.candidate))
    if a.report:
        open(a.report, "w", encoding="utf-8").write(text)
    print("\n".join(text.split("\n")[:14]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
