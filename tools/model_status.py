#!/usr/bin/env python3
"""THE table the owner reads the project from (the owner via the supervisor, 2026-10-03 20:52):
one row per container in the catalogue, every cell read from a record, never typed by hand.

    python tools/model_status.py --cache <catalogue> --rc <release-candidate tree> \
        --derivation <derived_census RESULTS.txt> [--records <records.json>] --out docs/reference/model-status.md

Per container: what the container is (format, NeuroTax version, trace date, the Forge revision when
the container records one); the census (the committed table's rows per memory class, and whether
the DERIVATION reproduces the walked keys exactly in both modes); the keys of that table certified
for each memory class over the keys it holds (the certifier's own coverage question,
`autotune_certified.entry_covers`); the last certified-only confirmation, whether its output was
judged from outside, its cold-run time; the hub artifact; the status. A cell with no record says
so — a blank and a zero read the same, and only one of them is honest. The second table is the
derivation per family. What either table shows as missing is the queue: derivation,
certification, confirmation, cold-run time.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CLASSES = (16, 32)
_LINE = re.compile(r"^(?P<model>\S+) (?P<mode>triton(?:-sequential)?) \| walked (?P<w>\d+) derived (?P<d>\d+): "
                   r"reproduced (?P<r>\d+), missed (?P<m>\d+), extra (?P<x>\d+) \| walked-with-op (?P<wo>\d+): "
                   r"reproduced (?P<ro>\d+), missed (?P<mo>\d+) \| not-yet (?P<ny>\d+)")


def containers(cache: Path):
    out = []
    if not cache.is_dir():
        raise SystemExit(f"model_status: {cache} is not a directory — no container to read, refused")
    for d in sorted(cache.iterdir(), key=lambda p: p.name.lower()):
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        m = json.loads(mf.read_text())
        out.append({"name": d.name, "family": m.get("family"), "nbx": m.get("nbx_version"),
                    "neurotax": m.get("neurotax_version"), "traced": (m.get("created_at") or "")[:10] or None,
                    "forge": m.get("forge_commit") or m.get("forge_sha")})
    if not out:
        raise SystemExit(f"model_status: {cache} holds no container (no */manifest.json) — refused")
    return out


def derivation(path: Path):
    """{model: {mode: verdict dict}} from a derived-census RESULTS file."""
    res = collections.defaultdict(dict)
    for line in path.read_text().splitlines():
        m = _LINE.match(line.strip())
        if m:
            g = {k: (int(v) if v.isdigit() else v) for k, v in m.groupdict().items()}
            res[g["model"]][g["mode"]] = g
    if not res:
        raise SystemExit(f"model_status: {path} holds no derivation line — refused")
    return res


def derivation_verdict(per_mode):
    if not per_mode:
        return "not derived"
    bad = []
    for mode in ("triton", "triton-sequential"):
        g = per_mode.get(mode)
        if g is None:
            bad.append(f"{mode} not run")
        elif g["m"] or g["x"] or g["mo"]:
            bad.append(f"{mode}: {g['r']}/{g['w']} keys, {g['m']} missed, {g['x']} extra")
        elif g["ny"]:
            bad.append(f"{mode}: exact on its {g['w']} keys, {g['ny']} op kind(s) not yet derived")
    return "exact" if not bad else "walked — " + "; ".join(bad)


def table_keys(rc: Path, cls: int):
    """{model: {(kernel, key text)}} from the committed census table of this class."""
    from neurobrix.kernels import autotune_certified as C
    path = rc / "src/neurobrix/config/census/nvidia/volta" / f"{cls}g.jsonl"
    out = collections.defaultdict(set)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = C.parse_key(row["key"])
        if key is not None:
            out[row["model"]].add((row["kernel"], C.key_repr(key)))
    return out


def certified_entries(rc: Path):
    """{kernel short name: entries} — every dtype file of a kernel, merged (a key text names its dtypes)."""
    out = collections.defaultdict(dict)
    for f in (rc / "src/neurobrix/config/autotune/nvidia/volta").glob("*.json"):
        out[f.name.split(".")[0]].update(json.loads(f.read_text()).get("entries") or {})
    return out


def coverage(keys, entries, cls):
    from neurobrix.kernels import autotune_certified as C
    have = sum(1 for k, t in keys if C.entry_covers(entries.get(C.kernel_short(k), {}), t, cls))
    return have, len(keys)


def registry_repos(path: Path):
    """{registry name: (Hugging Face repo id, checkpoint file or None)} from Forge's model registry. A
    `.pth` distribution is one checkpoint of a repository, so its identity is the pair."""
    import yaml
    reg = yaml.safe_load(path.read_text())
    out = {}
    for fam, models in reg.items():
        if fam.startswith("_") or not isinstance(models, dict):
            continue
        for name, entry in models.items():
            if isinstance(entry, dict) and entry.get("hf_repo"):
                out[name] = (entry["hf_repo"], entry.get("checkpoint_file"))
    return out


def _tokens(x):
    return [t for t in re.split(r"[^a-z0-9]+", x.lower()) if t]


def repo_of(name, repos):
    """The container's identity in the registry: its own entry, a repository it is named after (a
    `-Diffusers` suffix), or the one checkpoint whose name's tokens the container's name carries."""
    def ident(r):
        return r[0] + (f"::{r[1]}" if r[1] else "")
    if name in repos:
        return ident(repos[name])
    hit = {ident(r) for r in repos.values() if r[0].split("/")[-1].lower() == name.lower()}
    if len(hit) == 1:
        return hit.pop()
    tok, flat = set(_tokens(name)), "".join(_tokens(name))
    hit = {ident(r) for r in repos.values() if r[1] and (
        set(_tokens(Path(r[1]).stem)) <= tok or "".join(_tokens(Path(r[1]).stem)) == flat)}
    return hit.pop() if len(hit) == 1 else None


def census_movers(log: Path):
    """[(model, verdict, keys, tail)] from a census run's summary lines."""
    out = []
    for line in log.read_text().splitlines():
        m = re.match(r"^\[census\] (\S+)\s+(\w+)\s+(\w+)\s+(\d+) key\(s\)\s*(.*)$", line)
        if m:
            out.append(m.groups())
    return out


def _last(entries):
    if not entries:
        return None
    return entries[-1] if isinstance(entries, list) else entries


def _short(x, n=110):
    x = " ".join(str(x).split())
    return x if len(x) <= n else x[:n - 1] + "…"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rc", type=Path, required=True, help="the tree holding the census tables and certified directory of record")
    ap.add_argument("--derivation", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True, help="Forge's model_registry.yml (the Hugging Face repo ids)")
    ap.add_argument("--records", type=Path, required=True,
                    help="confirmations, oracle comparisons, judgments, hub artefacts, retraces — read from the campaign records")
    ap.add_argument("--census-log", type=Path, action="append", default=[], help="a census run's log (the movers' verdicts)")
    ap.add_argument("--notes", type=Path, default=REPO / "docs/reference/model-status-notes.json",
                    help="measured causes no record field carries, one per container, each with its source")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = containers(a.cache)
    der = derivation(a.derivation)
    rec = json.loads(a.records.read_text())
    notes = {k: v for k, v in json.loads(a.notes.read_text()).items() if not k.startswith("_")} if a.notes.exists() else {}
    unknown = sorted(set(notes) - {c["name"] for c in rows})
    if unknown:
        raise SystemExit(f"model_status: notes name no container of the catalogue: {unknown} — refused")
    repos = registry_repos(a.registry)
    keys = {c: table_keys(a.rc, c) for c in CLASSES}
    ent = certified_entries(a.rc)
    fam = collections.defaultdict(collections.Counter)
    L = [
        "# Model status — one row per container (generated by `tools/model_status.py`, never edited by hand)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} from the catalogue `{a.cache}` ({len(rows)} containers), the census "
        f"tables and certified directory of `{a.rc}`, the derivation `{a.derivation}`, the registry `{a.registry}` and the "
        f"campaign records `{a.records.name}` (every cell names its record there).",
        "",
        "How to read it. **Census**: whether the derivation (`tools/derived_census.py`, no execution) reproduces the walked "
        "census exactly in both modes — the committed tables still hold the walk's rows. **Certified**: keys of the committed "
        "table certified for that card class / keys it holds for the model. The **oracle ladder** (the owner's method, "
        "2026-10-03): the vendor's PyTorch pipeline at the same request is the oracle of the MODEL; triton-sequential, the "
        "container's graph op by op, is the oracle of the CONTAINER (it matches the vendor: the trace is right); the Triton "
        "compiled mode is validated against both. A model that looks broken is never retraced on sight — the ladder first. "
        "A certified-only run at zero misses is itself the proof that no autotune cost is left at runtime. "
        "*no record* means none exists on this rack.",
        "",
        "| container · repo · format/NeuroTax · traced · Forge | census | certified 16 GB | certified 32 GB | vendor oracle "
        "| sequential oracle | Triton compiled (certified-only) | judged from outside | hub artefact | status |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    queue = collections.defaultdict(list)
    seen_repo = collections.defaultdict(list)
    for c in rows:
        n = c["name"]
        r = rec.get(n) or {}
        repo = repo_of(n, repos)
        if repo is None and isinstance(r.get("hub_artifact_5_0"), dict) and r["hub_artifact_5_0"].get("slug"):
            # named by the hub record's slug when the registry name does not resolve — and said so
            repo = f"{r['hub_artifact_5_0']['slug']} (hub slug; the registry name does not resolve)"
        if repo:
            seen_repo[repo].append(n)
        dv = derivation_verdict(der.get(n))
        fam[c["family"]]["containers"] += 1
        fam[c["family"]]["exact" if dv == "exact" else "walked"] += 1
        cov = {cls: coverage(keys[cls].get(n, set()), ent, cls) for cls in CLASSES}
        vo, so = _last(r.get("vendor_oracle")), _last(r.get("sequential_oracle"))
        vo_t = f"{vo.get('date')}: {_short(vo.get('result'), 70)}" if vo else "no record"
        so_t = f"{so.get('date')}: {_short(so.get('result'), 70)}" if so else "no record"
        cm = r.get("compiled_mode") or {}
        cm_t = "; ".join(f"{cls} {_short(v.get('result'), 40)} ({str(v.get('date'))[:10]})" for cls, v in sorted(cm.items())
                         if isinstance(v, dict)) or "no record"
        jd = r.get("judged_from_outside") or []
        jd = jd if isinstance(jd, list) else [jd]
        jl = jd[-1] if jd else None
        jd_t = f"{str(jl.get('date'))[:10]}: {_short(jl.get('verdict'), 60)}" if jl else "no record"
        hub = (r.get("hub_artifact_5_0") or {}).get("value", "no record") if isinstance(r.get("hub_artifact_5_0"), dict) else "no record"
        reds = r.get("red_cells_row") or []
        cm_ok = bool(cm) and all(isinstance(v, dict) and "zero miss" in str(v.get("result")) for v in cm.values())
        if reds:
            status = "; ".join(f"{x.get('class', '?')} defect (red-cells L{x.get('line')}: {_short(x.get('what'), 60)})" for x in reds)
        elif cm_ok and jl and "PASS" in str(jl.get("verdict", "")).upper():
            status = "functional"
        elif cm_ok:
            status = "runs at zero miss, output not judged on record"
        else:
            status = "not confirmed"
        cert = {cls: (f"{h}/{t}" if t else "no rows") for cls, (h, t) in cov.items()}
        what = " · ".join(str(x) for x in (repo or "repo not in the registry", f"{c['nbx']}/{c['neurotax']}", c["traced"],
                                             c["forge"] or "Forge sha not recorded in the container"))
        L.append(f"| `{n}` · {what} | {dv} | {cert[16]} | {cert[32]} | {vo_t} | {so_t} | {cm_t} | {jd_t} | {hub} | {status} |")
        if dv != "exact":
            queue["derivation"].append(n)
        if any(h < t or t == 0 for h, t in cov.values()):
            queue["certification"].append(n)
        if not vo or not so:
            queue["oracle ladder"].append(n)
        if not cm_ok:
            queue["Triton compiled confirmation"].append(n)
    dup = {r_: ns for r_, ns in seen_repo.items() if len(ns) > 1}
    L += ["", f"**Distinct models:** {len(rows)} containers, {len(seen_repo)} with a registry repo id"
          + (f"; containers sharing one repo id: {dup}" if dup else "; no two containers share a repo id") + "."]
    L += ["", "## The derivation per family", "", "| family | containers | derivation exact | still walked |", "|---|---:|---:|---:|"]
    for f, cnt in sorted(fam.items()):
        L.append(f"| {f} | {cnt['containers']} | {cnt['exact']} | {cnt['walked']} |")
    L += ["", "## The derivation's open rows (each with what the record says)", "", "| container | mode | derived vs walked |", "|---|---|---|"]
    for n, modes in sorted(der.items()):
        for mode, g in sorted(modes.items()):
            if g["m"] or g["x"] or g["mo"] or g["ny"]:
                L.append(f"| `{n}` | {mode} | walked {g['w']}, derived {g['d']}: reproduced {g['r']}, missed {g['m']}, extra {g['x']}; "
                         f"pairs missed {g['mo']}; op kinds not yet derived {g['ny']} |")
    raw = a.derivation.read_text().splitlines()
    for n in [c["name"] for c in rows if c["name"] not in der]:
        said = [l_ for l_ in raw if l_.startswith(n + " ")]
        if not said:
            L.append(f"| `{n}` | both | no derivation line |")
        for l_ in said:
            mode = l_.split()[1]
            err = l_.split("| ERR", 1)[1].strip() if "| ERR" in l_ else l_[len(n) + len(mode) + 2:]
            L.append(f"| `{n}` | {mode} | not derived: {_short(err, 200)} |")
    if a.census_log:
        L += ["", "## The census of the movers", "", "| container | family | verdict | keys | what the census said |", "|---|---|---|---:|---|"]
        for lg in a.census_log:
            for m, f, v, k, tail in census_movers(lg):
                L.append(f"| `{m}` | {f} | {v} | {k} | {lg.parent.name}/{lg.name}: {tail or '—'} |")
    if notes:
        L += ["", "## Causes measured (docs/reference/model-status-notes.json)", ""]
        for n, why in sorted(notes.items()):
            L.append(f"- **`{n}`** — {why}")
    rt = rec.get("_retraces") or []
    if rt:
        L += ["", "## Retraces of the last month, and whether the oracle ladder preceded them", "",
              "| container | date | outcome | stated reason | ladder on record before the retrace |", "|---|---|---|---|---|"]
        for x in rt:
            L.append(f"| `{x.get('model')}` | {x.get('date')} | {_short(x.get('outcome'), 70)} | {_short(x.get('reason') or 'no reason on record', 80)} "
                     f"| {_short(x.get('ladder_on_record'), 90)} |")
    L += ["", "## The queue, in the owner's order", ""]
    for step in ("derivation", "certification", "oracle ladder", "Triton compiled confirmation"):
        L.append(f"- **{step}** ({len(queue[step])}): " + (", ".join(f"`{x}`" for x in queue[step]) or "none"))
    a.out.write_text("\n".join(L) + "\n")
    print(f"model_status: {len(rows)} containers -> {a.out}")


if __name__ == "__main__":
    main()
