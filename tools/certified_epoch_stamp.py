"""Replace a shipped certified file by its epoch re-proof, only when the replacement loses nothing unsaid.

The certifier's epoch re-prove (`neurobrix autotune certify --reprove-epoch-from <shipped> --out <draft>`, one run per
memory class) writes into a DRAFT tree the stored configurations proven again under the library's bits epoch, and a
ledger (`<draft>/reprove_epoch_ledger.tsv`) of every key it tried. This tool walks the shipped tree (no vendor, card or
file named) and, per file, counts each certification owed (one key in one memory class, the unit a card re-proves):

  re-proven  the draft holds it for that class, every proof of it at the library's epoch, the configuration
             byte-identical to the shipped one, and the same matrix-unit tile (or none on both);
  failed     the ledger's last line for it says failed (its re-prove raised, or it could not be filed);
  retired    a `--retired` list names it (verdict `retired_*`): a key the library gives no launch, so there is nothing
             to re-prove or recertify (the draft must not hold it);
  pending    neither — not yet tried, or tried by no ledger line, or a draft configuration or unit tile that differs;
             and, both ways, a draft certification the file did not owe or a shipped one already of the
             library's epoch that the draft does not hold as it stands.

A file is REPLACED (`--apply`) only when nothing is pending: every certification it owes is re-proven or failed, and
the draft file passes the library's own gate (`validate`: the epoch, the format, every proof). The replacement is the
draft file as it stands: it holds the re-proven certifications only, and is never mixed. THE FAILED CERTIFICATIONS
LEAVE THE FILE until their full recertification: each one is appended to `--recert-list` (file, key, memory_class,
reason) — never dropped silently, never retuned here. THE RETIRED ONES leave the file too, counted apart (the table's
`retired`) and kept off `--recert-list`: the list that names them is their record. The re-prove runs under one code generator: a re-proven
certification's proofs under OTHER generators (`also_proven`, alternates) leave with the replacement too, and each is
appended to `--recert-list` with its `generator`, to be re-proven under it. A file with every certification failed is not replaced (its
replacement would be empty): it is said, and left for a ruling. A file with every certification RETIRED (and none
pending or failed) is REMOVED by `--apply`: no key the library forms is left in it, and an empty file is not shipped.

ADD FROM A SUBSET DRAFT (`--add-from DRAFT ...`, no ledger): a recertification draft (`autotune certify --keys --out`)
holds a SUBSET of a file's keys, each certified on ONE memory class at the library's bits epoch. Each of its
certifications is ADDED to the shipped file: a key the file lacks takes the draft's entry as it stands; a key the file
holds for the other class takes this class in its variant slot, the other class untouched. It NEVER replaces: a
certification of the same key and class already in the file (or added by an earlier draft of the same run) is refused
by name, as is a draft entry of no single class or of another bits epoch, and a file with any refusal, or whose result
fails the library's `validate`, is not written. A row per file (file, added, variants, refused) goes to `--table` first;
with `--apply` the file is then written whole (beside it, renamed over it), in the certifier's own format.

One row per file reached (a draft file or a ledger line) is written to `--table` (file, keys, re_proven, failed,
seconds, generators_dropped, retired), replacing that file's earlier row. A file's recert rows and table row are written BEFORE it is replaced, and
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
TABLE_COLS = ["file", "keys", "re_proven", "failed", "seconds", "generators_dropped", "retired"]
RECERT_COLS = ["file", "key", "memory_class", "reason", "generator"]
ADD_COLS = ["file", "added", "variants", "refused"]


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


def read_retired(paths):
    """{(file, key text, memory class GB)} a `--retired` list names with a verdict `retired_*` (columns file, key,
    memory_class, verdict; any other verdict retires nothing)."""
    out = set()
    for path in paths or []:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if str(r.get("verdict") or "").startswith("retired"):
                    out.add((r["file"], _norm(r["key"]), int(str(r["memory_class"]).rstrip("g"))))
    return out


def _generators(head):
    """The code generators (`C.proof_backend`) one memory class's certifications are proven under: every proof of its
    head and of its alternates (`also_proven` included)."""
    return {C.proof_backend(p) for c in C.class_certifications(head) for p in C.certification_proofs(c)}


def owed_heads(entries):
    """[(key text, memory class GB, the shipped head)] of every certification not at the library's bits epoch."""
    out = []
    for ktext, e in entries.items():
        for cls in sorted(C.covered_memory_gbs(e)):
            head = C.entry_for_memory_gb(e, cls)
            if head and not _class_epochs(head) <= {C.BITS_EPOCH}:
                out.append((_norm(ktext), cls, head))
    return out


def judge(shipped_path, draft_path, ledger, rel, retired=frozenset()):
    """The file's verdict: {'keys', 're_proven', 'failed' [(key, class, reason)], 'pending' [(key, class, why)],
    'retired', 'seconds', 'problems' (the draft's gate), 'reached'}."""
    doc = json.loads(Path(shipped_path).read_text(encoding="utf-8"))
    owed = owed_heads(doc.get("entries") or {})
    draft_doc = json.loads(Path(draft_path).read_text(encoding="utf-8")) if Path(draft_path).exists() else None
    draft = {_norm(k): e for k, e in ((draft_doc or {}).get("entries") or {}).items()}
    fname = str(rel)
    v = {"keys": len(owed), "re_proven": 0, "failed": [], "pending": [], "seconds": 0.0, "problems": [],
         "generators_dropped": [], "retired": 0,
         "reached": draft_doc is not None or any(k[0] == fname for k in ledger)}
    owed_ids = set()
    for ktext, cls, head in owed:
        owed_ids.add((ktext, cls))
        line = ledger.get((fname, ktext, cls))
        if line is not None:
            v["seconds"] += float(line.get("seconds") or 0)
        got = C.entry_for_memory_gb(draft.get(ktext), cls)
        if (fname, ktext, cls) in retired:
            if got is None:
                v["retired"] += 1
            else:
                v["pending"].append((ktext, cls, "named retired, yet the draft holds it"))
        elif got is not None:
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
                # the re-prove ran under ONE generator: the shipped proofs of the others (`also_proven`, alternates)
                # leave with the replacement — each is said, never dropped silently (qa 1227-4)
                for gen in sorted(_generators(head) - _generators(got), key=str):
                    v["generators_dropped"].append((ktext, cls, gen, ", ".join(sorted(map(str, _generators(got))))))
        elif line is not None and line.get("outcome") == "failed":
            v["failed"].append((ktext, cls, line.get("reason") or "failed"))
        else:
            v["pending"].append((ktext, cls, "not re-proven yet"))
    # BOTH WAYS (qa 1227-6): the draft holds nothing the shipped file did not owe, and every shipped certification not
    # owed (already of the library's epoch) is in the draft as it stands — the replacement loses nothing unsaid
    shipped_entries = {_norm(k): e for k, e in (doc.get("entries") or {}).items()}
    for ktext, e in draft.items():
        for cls in C.covered_memory_gbs(e):
            if (ktext, cls) not in owed_ids and (C.entry_for_memory_gb(e, cls)
                                                 != C.entry_for_memory_gb(shipped_entries.get(ktext), cls)):
                v["pending"].append((ktext, cls, "in the draft, not owed by the shipped file"))
    for ktext, e in shipped_entries.items():
        for cls in C.covered_memory_gbs(e):
            if (ktext, cls) not in owed_ids and (C.entry_for_memory_gb(draft.get(ktext), cls)
                                                 != C.entry_for_memory_gb(e, cls)):
                v["pending"].append((ktext, cls, "of the library's epoch in the shipped file and not in the draft as "
                                                 "it stands: the replacement would lose it"))
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


def run(shipped, draft, table, recert_list, apply=False, say=print, retired=()):
    shipped, draft = Path(shipped), Path(draft)
    ledger, retired = read_ledger(draft / LEDGER), read_retired(retired)
    rows, recert, replaced, removed = [], [], [], []
    for path in C.files(shipped):
        rel = path.relative_to(shipped)
        v = judge(path, draft / rel, ledger, rel, retired)
        if not v["keys"] or not v["reached"]:
            continue
        row = {"file": str(rel), "keys": v["keys"], "re_proven": v["re_proven"], "failed": len(v["failed"]),
               "seconds": f"{v['seconds']:.1f}", "generators_dropped": len(v["generators_dropped"]),
               "retired": v["retired"]}
        rows.append(row)
        ready = not v["pending"] and not v["problems"] and v["re_proven"] > 0
        gone = not v["pending"] and not v["failed"] and not v["re_proven"] and v["retired"] == v["keys"]
        state = ("READY" if ready else "EVERY KEY RETIRED: removed, nothing left to ship" if gone else "ALL FAILED: not replaced, left for a ruling" if not v["pending"]
                 and not v["re_proven"] else f"NOT READY: {len(v['pending'])} pending"
                 + (f", draft gate: {v['problems'][0]}" if v["problems"] else ""))
        say(f"{rel}: {v['keys']} owed, {v['re_proven']} re-proven, {len(v['failed'])} failed, "
            f"{v['retired']} retired, {v['seconds']:.1f} s — {state}")
        for ktext, cls, why in v["pending"][:3]:
            say(f"  pending {ktext} {cls}g: {why}")
        mine = []
        if ready:
            for ktext, cls, why in v["failed"]:
                mine.append({"file": str(rel), "key": ktext, "memory_class": f"{cls}g", "reason": why})
                say(f"  LEAVES the file until its full recertification: {ktext} {cls}g ({why})")
            for ktext, cls, gen, ran in v["generators_dropped"]:
                mine.append({"file": str(rel), "key": ktext, "memory_class": f"{cls}g", "generator": gen,
                             "reason": f"re-proven under {ran} only: its proof under {gen} leaves with the replacement"})
            if v["generators_dropped"]:
                say(f"  {len(v['generators_dropped'])} proof(s) of other generators leave with the replacement "
                    f"(on the recert list, by generator)")
            recert.extend(mine)
        # THIS FILE's rows are on disk BEFORE its replacement: a stop after it must not leave a failed key that has
        # left the shipped file and joined no list (qa 1227-2) — a re-run finds the replaced file owing nothing
        if mine:
            _upsert(recert_list, RECERT_COLS, mine,
                    key=lambda r: (r["file"], r["key"], r["memory_class"], r.get("generator") or ""))
        _upsert(table, TABLE_COLS, [row], key=lambda r: r["file"])
        if ready and apply:
            _replace(draft / rel, path)
            replaced.append(str(rel))
            say(f"  replaced {path} by its draft")
        if gone and apply:
            path.unlink()
            removed.append(str(rel))
            say(f"  removed {path}: every key it owed is retired")
    return {"rows": rows, "recert": recert, "replaced": replaced, "removed": removed}


def _add_one(entries, ktext, cert):
    """Place one draft certification in `entries` (keyed by normalised key text): "added", "variant", or the refusal."""
    classes = C.covered_memory_gbs(cert)
    if len(classes) != 1 or cert.get("variants"):
        return f"covers {sorted(classes)}, not one memory class"
    if not _class_epochs(cert) <= {C.BITS_EPOCH}:
        return "a proof not of the library's bits epoch"
    cls = next(iter(classes))
    cur = entries.get(ktext)
    if C.entry_for_memory_gb(cur, cls) is not None:
        return f"the file already holds {ktext} for {cls}g: an existing certification is never overwritten"
    if cur is None:
        entries[ktext] = cert
        return "added"
    cur.setdefault("variants", {})[f"{cls}g"] = cert
    return "variant"


def _write_doc(doc, dst):
    """`dst` becomes `doc` in the certifier's own format, whole or not at all (written beside it, renamed over it)."""
    dst = Path(dst)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def add(shipped, drafts, table, apply=False, say=print):
    """Add every certification of the subset drafts to the shipped files (module docstring); returns the rows and the
    files written."""
    shipped = Path(shipped)
    plan = {}                        # rel -> [(ktext, cert, draft)]
    for d in map(Path, drafts):
        for path in C.files(d):
            doc = json.loads(path.read_text(encoding="utf-8"))
            plan.setdefault(path.relative_to(d), []).extend(
                (_norm(k), e, d) for k, e in (doc.get("entries") or {}).items())
    rows, written = [], []
    for rel, certs in sorted(plan.items(), key=lambda x: str(x[0])):
        target = shipped / rel
        if not target.exists():
            say(f"{rel}: REFUSED, no shipped file to add to")
            rows.append({"file": str(rel), "added": 0, "variants": 0, "refused": len(certs)})
            continue
        doc = json.loads(target.read_text(encoding="utf-8"))
        entries = {_norm(k): e for k, e in (doc.get("entries") or {}).items()}
        n = {"added": 0, "variant": 0}
        refused = []
        for ktext, cert, d in certs:
            got = _add_one(entries, ktext, json.loads(json.dumps(cert)))
            if got in n:
                n[got] += 1
            else:
                refused.append(got)
                say(f"  REFUSED {rel} {ktext} (from {d}): {got}")
        merged = {**doc, "entries": entries}
        problems = C.validate(merged, target) if not refused else []
        row = {"file": str(rel), "added": n["added"], "variants": n["variant"], "refused": len(refused)}
        rows.append(row)
        _upsert(table, ADD_COLS, [row], key=lambda r: r["file"])
        ok = not refused and not problems
        say(f"{rel}: {n['added']} added, {n['variant']} beside the other class, {len(refused)} refused — "
            + ("READY" if ok else "NOT WRITTEN" + (f", gate: {problems[0]}" if problems else "")))
        if ok and apply:
            _write_doc(merged, target)
            written.append(str(rel))
            say(f"  wrote {target}")
    return {"rows": rows, "written": written}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shipped", required=True, help="the certified tree re-proven from (vendor/profile/*.json)")
    ap.add_argument("--draft", help="the draft tree the re-prove wrote (and its ledger)")
    ap.add_argument("--table", required=True, help="the stamp table (TSV: file, keys, re_proven, failed, seconds)")
    ap.add_argument("--recert-list", help="the full-recertification list the failed keys join (TSV)")
    ap.add_argument("--apply", action="store_true", help="replace every READY shipped file by its draft, remove "
                    "every file whose keys are all retired")
    ap.add_argument("--retired", nargs="+", default=[], help="TSV(s) of keys the library gives no launch (file, key, "
                    "memory_class, verdict retired_*): they leave the file, counted apart, off --recert-list")
    ap.add_argument("--add-from", nargs="+", default=[], help="subset recertification draft(s): ADD their certifications "
                    "(never replacing one); no ledger, no --draft or --recert-list")
    a = ap.parse_args(argv)
    if a.add_from:
        if any(Path(d).resolve() == Path(a.shipped).resolve() for d in a.add_from):
            sys.exit("a draft tree is the shipped tree: refused")
        add(a.shipped, a.add_from, a.table, apply=a.apply)
        return 0
    if not (a.draft and a.recert_list):
        ap.error("--draft and --recert-list are required (or --add-from)")
    if Path(a.draft).resolve() == Path(a.shipped).resolve():
        sys.exit("the draft tree is the shipped tree: refused")
    run(a.shipped, a.draft, a.table, a.recert_list, apply=a.apply, retired=a.retired)
    return 0


if __name__ == "__main__":
    sys.exit(main())
