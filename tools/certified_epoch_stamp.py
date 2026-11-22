"""Replace a shipped certified file by its epoch re-proof, only when the replacement loses nothing unsaid.

The certifier's epoch re-prove (`neurobrix autotune certify --reprove-epoch-from <shipped> --out <draft>`, one run per
memory class) writes into a DRAFT tree the stored configurations proven again under the library's bits epoch, and a
ledger (`<draft>/reprove_epoch_ledger.tsv`) of every key it tried. This tool walks the shipped tree (no vendor, card or
file named) and, per file, counts each certification owed (one key in one memory class, the unit a card re-proves):

  re-proven  the draft holds it for that class, every proof of it at the library's epoch, the configuration
             byte-identical to the shipped one, and the same matrix-unit tile (or none on both);
  failed     the ledger's last line for it says failed (its re-prove raised, or it could not be filed);
  pending    neither — not yet tried, or tried by no ledger line, or a draft configuration or unit tile that differs.

A file is REPLACED (`--apply`) only when nothing is pending: every certification it owes is re-proven or failed, and
the draft file passes the library's own gate (`validate`: the epoch, the format, every proof). The replacement is the
draft file as it stands: it holds the re-proven certifications only, and is never mixed. THE FAILED CERTIFICATIONS
LEAVE THE FILE until their full recertification: each one is appended to `--recert-list` (file, key, memory_class,
reason) — never dropped silently, never retuned here. A file with every certification failed is not replaced (its
replacement would be empty): it is said, and left for a ruling.

One row per file reached (a draft file or a ledger line) is written to `--table` (file, keys, re_proven, failed,
seconds), replacing that file's earlier row. A file's recert rows and table row are written BEFORE it is replaced, and
the replacement is whole or nothing (a copy beside it, renamed over it): a stop at any point loses no failed key. Run with the frozen library's python: the library reads the files.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

from nbxtensor.ops import autotune_certified as C
from neurobrix.kernels.autotune_certify import unit_tile

LEDGER = "reprove_epoch_ledger.tsv"
TABLE_COLS = ["file", "keys", "re_proven", "failed", "seconds"]
RECERT_COLS = ["file", "key", "memory_class", "reason"]


def _norm(ktext):
    key = C.parse_key(ktext)
    return C.key_repr(key) if key is not None else ktext


def _class_epochs(head):
    return {p.get("bits_epoch") for c in C.class_certifications(head) for p in C.certification_proofs(c)}


def read_ledger(path):
    """{(file relative to the tree, key text, memory class GB): its LAST line} — a key tried again keeps only its latest outcome."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[(r["file"], _norm(r["key"]), int(str(r["memory_class"]).rstrip("g")))] = r
    return out


def owed_heads(entries):
    """[(key text, memory class GB, the shipped head)] of every certification not at the library's bits epoch."""
    out = []
    for ktext, e in entries.items():
        for cls in sorted(C.covered_memory_gbs(e)):
            head = C.entry_for_memory_gb(e, cls)
            if head and not _class_epochs(head) <= {C.BITS_EPOCH}:
                out.append((_norm(ktext), cls, head))
    return out


def judge(shipped_path, draft_path, ledger, rel):
    """The file's verdict: {'keys', 're_proven', 'failed' [(key, class, reason)], 'pending' [(key, class, why)],
    'seconds', 'problems' (the draft's gate), 'reached'}."""
    doc = json.loads(Path(shipped_path).read_text(encoding="utf-8"))
    owed = owed_heads(doc.get("entries") or {})
    draft_doc = json.loads(Path(draft_path).read_text(encoding="utf-8")) if Path(draft_path).exists() else None
    draft = {_norm(k): e for k, e in ((draft_doc or {}).get("entries") or {}).items()}
    fname = str(rel)
    v = {"keys": len(owed), "re_proven": 0, "failed": [], "pending": [], "seconds": 0.0, "problems": [],
         "reached": draft_doc is not None or any(k[0] == fname for k in ledger)}
    owed_ids = set()
    for ktext, cls, head in owed:
        owed_ids.add((ktext, cls))
        line = ledger.get((fname, ktext, cls))
        if line is not None:
            v["seconds"] += float(line.get("seconds") or 0)
        got = C.entry_for_memory_gb(draft.get(ktext), cls)
        if got is not None:
            if not _class_epochs(got) <= {C.BITS_EPOCH}:
                v["pending"].append((ktext, cls, "the draft's proof is not of the library's bits epoch"))
            elif got.get("config") != head.get("config"):
                v["pending"].append((ktext, cls, f"the draft's configuration {got.get('config')} is not the shipped "
                                                 f"{head.get('config')}"))
            elif unit_tile(got) != unit_tile(head):
                v["pending"].append((ktext, cls, f"the draft's unit tile {unit_tile(got)} is not the shipped "
                                                 f"{unit_tile(head)}"))
            else:
                v["re_proven"] += 1
        elif line is not None and line.get("outcome") == "failed":
            v["failed"].append((ktext, cls, line.get("reason") or "failed"))
        else:
            v["pending"].append((ktext, cls, "not re-proven yet"))
    for ktext, e in draft.items():                 # the draft holds nothing the shipped file did not owe
        for cls in C.covered_memory_gbs(e):
            if (ktext, cls) not in owed_ids:
                v["pending"].append((ktext, cls, "in the draft, not owed by the shipped file"))
    if draft_doc is not None:
        v["problems"] = C.validate(draft_doc, Path(draft_path))
    return v


def _upsert(path, cols, rows, key):
    path = Path(path)
    old = []
    if path.exists():
        with open(path, newline="") as fh:
            old = list(csv.DictReader(fh, delimiter="\t"))
    keep = {key(r): r for r in old}
    for r in rows:
        keep[key(r)] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in keep.values():
            w.writerow({c: r.get(c, "") for c in cols})
    tmp.replace(path)


def _replace(src, dst):
    """`dst` becomes a copy of `src` whole or stays as it was: copied beside it, then renamed over it (a stop
    mid-copy leaves the shipped file untouched, never truncated)."""
    dst = Path(dst)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def run(shipped, draft, table, recert_list, apply=False, say=print):
    shipped, draft = Path(shipped), Path(draft)
    ledger = read_ledger(draft / LEDGER)
    rows, recert, replaced = [], [], []
    for path in C.files(shipped):
        rel = path.relative_to(shipped)
        v = judge(path, draft / rel, ledger, rel)
        if not v["keys"] or not v["reached"]:
            continue
        row = {"file": str(rel), "keys": v["keys"], "re_proven": v["re_proven"], "failed": len(v["failed"]),
               "seconds": f"{v['seconds']:.1f}"}
        rows.append(row)
        ready = not v["pending"] and not v["problems"] and v["re_proven"] > 0
        state = ("READY" if ready else "ALL FAILED: not replaced, left for a ruling" if not v["pending"]
                 and not v["re_proven"] else f"NOT READY: {len(v['pending'])} pending"
                 + (f", draft gate: {v['problems'][0]}" if v["problems"] else ""))
        say(f"{rel}: {v['keys']} owed, {v['re_proven']} re-proven, {len(v['failed'])} failed, "
            f"{v['seconds']:.1f} s — {state}")
        for ktext, cls, why in v["pending"][:3]:
            say(f"  pending {ktext} {cls}g: {why}")
        mine = []
        if ready:
            for ktext, cls, why in v["failed"]:
                mine.append({"file": str(rel), "key": ktext, "memory_class": f"{cls}g", "reason": why})
                say(f"  LEAVES the file until its full recertification: {ktext} {cls}g ({why})")
            recert.extend(mine)
        # THIS FILE's rows are on disk BEFORE its replacement: a stop after it must not leave a failed key that has
        # left the shipped file and joined no list (qa 1227-2) — a re-run finds the replaced file owing nothing
        if mine:
            _upsert(recert_list, RECERT_COLS, mine, key=lambda r: (r["file"], r["key"], r["memory_class"]))
        _upsert(table, TABLE_COLS, [row], key=lambda r: r["file"])
        if ready and apply:
            _replace(draft / rel, path)
            replaced.append(str(rel))
            say(f"  replaced {path} by its draft")
    return {"rows": rows, "recert": recert, "replaced": replaced}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shipped", required=True, help="the certified tree re-proven from (vendor/profile/*.json)")
    ap.add_argument("--draft", required=True, help="the draft tree the re-prove wrote (and its ledger)")
    ap.add_argument("--table", required=True, help="the stamp table (TSV: file, keys, re_proven, failed, seconds)")
    ap.add_argument("--recert-list", required=True, help="the full-recertification list the failed keys join (TSV)")
    ap.add_argument("--apply", action="store_true", help="replace every READY shipped file by its draft")
    a = ap.parse_args(argv)
    if Path(a.draft).resolve() == Path(a.shipped).resolve():
        sys.exit("the draft tree is the shipped tree: refused")
    run(a.shipped, a.draft, a.table, a.recert_list, apply=a.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
