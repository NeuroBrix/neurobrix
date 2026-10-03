#!/usr/bin/env python3
"""THE table the owner reads the project from (the owner via the supervisor, 2026-10-03 20:52):
one row per container in the catalogue, every cell read from a record, never typed by hand.

    python tools/model_status.py --cache <catalogue> --rc <release-candidate tree> \
        --derivation <derived_census RESULTS.txt> [--records <records.json>] --out docs/reference/model-status.md

Per container: what the container is (format, NeuroTax version, trace date, the build revision when
the container records one); the census (the committed table's rows per memory class, and whether
the DERIVATION reproduces the walked keys exactly in both modes); the keys of that table certified
for each memory class over the keys it holds (the certifier's own coverage question,
`autotune_certified.entry_covers`); the last certified-only confirmation, whether its output was
judged from outside, its cold-run time; the hub artifact; the dimensions the trace left at its own
value (`tools/frozen_dim_scan.py`, a static read of every graph); the status. A cell with no record says
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


def frozen_scan(path: Path, names: set) -> dict:
    """{container: record} from `tools/frozen_dim_scan.py`'s JSONL. A refused scan, or a record naming
    no container of the catalogue, is refused here — a stale or foreign scan is not a column."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("refused") and not r.get("container"):
            raise SystemExit(f"model_status: the frozen scan {path} is a refusal ({r['refused']}) — refused")
        out[r["container"]] = r
    foreign = sorted(set(out) - names)
    if foreign:
        raise SystemExit(f"model_status: the frozen scan names no container of the catalogue: {foreign} — refused")
    return out


def _pattern(p) -> str:
    notes = ["in the arguments" if p.get("in_args") else "annotation only"]
    if p.get("dropped"):
        notes.append("the symbol dropped at this op")
    if not p.get("live"):
        notes.append("unused output")
    if p.get("origins", 1) > 1:
        notes.append(f"x{p['origins']}")
    return f"{p['component']}/`{p['first']}` dim {p['position']} = {p['value']} ~ `{p['matches']}` ({', '.join(notes)})"


def frozen_cell(r) -> str:
    if r is None:
        return "no record"
    if r.get("refused"):
        return f"refused: {_short(r['refused'], 60)}"
    c = r.get("counts") or {}
    if r.get("verdict") == "UNREADABLE":
        return f"UNREADABLE ({c.get('unreadable')} graph(s))"
    if r.get("patterns"):
        p = r["patterns"][0]
        return (f"FROZEN: {c.get('FROZEN_patterns')} pattern(s), {c.get('FROZEN_patterns_live')} live — first "
                f"{p['component']}/{p['first']} = {p['value']} ({_short(p['matches'], 40)})")
    if r.get("verdict") == "AMBIGUOUS":
        return f"clean of FROZEN; {c.get('AMBIGUOUS')} ambiguous"
    return "clean"


def frozen_section(path: Path, rows, frozen: dict) -> list:
    L = ["", "## Frozen dimensions — the static scan (`tools/frozen_dim_scan.py`)", "",
         f"Read from `{path}`; nothing executed. The field read is each tensor's symbolic annotation "
         "(`symbolic_shape.dims`), never the trace record (`output_shapes`); for each hit the producer's arguments are "
         "read too. A dimension is **FROZEN** when it is a plain integer where the same graph carries that number as a "
         "symbol or a symbolic expression (a product, sum or floordiv of symbols the model computes), and the number is "
         "not an extent of a weight nor an integer of the model's configuration. **Ambiguous**: the number is also a "
         "weight extent or a configuration integer, or the match runs only through a symbol traced at 0, 1 or 2 (R39) "
         "or through a trace extent of an input axis the graph keeps literal — read one by one, never counted. "
         "A **pattern** is one freeze repeated per block (component, op type, position, value); **live** means a "
         "tensor carrying it is consumed; *the symbol dropped at this op*: the op consumes a tensor carrying the value "
         "symbolically and writes it back literal, the strongest witness a static read gives. *in the arguments*: the runtime evaluates the literal; *annotation only*: the "
         "arguments are symbolic or inferred (`-1`) and the annotation the derivation reads is literal.", "",
         "| container | verdict | FROZEN patterns (live) | ambiguous groups | the patterns |", "|---|---|---:|---:|---|"]
    for c in rows:
        r = frozen.get(c["name"])
        if r is None:
            L.append(f"| `{c['name']}` | no record | | | |")
            continue
        k = r.get("counts") or {}
        pats = "; ".join(_pattern(p) for p in (r.get("patterns") or [])[:4])
        more = len(r.get("patterns") or []) - 4
        L.append(f"| `{c['name']}` | {r.get('verdict')} | {k.get('FROZEN_patterns', 0)} ({k.get('FROZEN_patterns_live', 0)}) | "
                 f"{k.get('AMBIGUOUS', 0)} | {pats or '—'}{f'; and {more} more' if more > 0 else ''} |")
    return L


def _short(x, n=110):
    x = " ".join(str(x).split())
    return x if len(x) <= n else x[:n - 1] + "…"


def _public(text: str) -> str:
    """The table is a PUBLIC document: it says what the build did, never names the private build toolchain
    (rules/docs-and-language.md, two spheres). Campaign records and the registry path carry its name; the
    rendered text replaces a path through its tree by the file's own name and the word by "the build"."""
    import re
    text = re.sub(r"(?:/[\w.-]+)*/forge/((?:[\w.-]+/)*)([\w.-]+)", r"the build's \2", text)
    text = re.sub(r"\bForge\b", "the build", text)
    return re.sub(r"\bforge\b", "the build", text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rc", type=Path, required=True, help="the tree holding the census tables and certified directory of record")
    ap.add_argument("--derivation", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True, help="Forge's model_registry.yml (the Hugging Face repo ids)")
    ap.add_argument("--records", type=Path, required=True,
                    help="confirmations, oracle comparisons, judgments, hub artefacts, retraces — read from the campaign records")
    ap.add_argument("--census-log", type=Path, action="append", default=[], help="a census run's log (the movers' verdicts)")
    ap.add_argument("--judged", type=Path, action="append", default=[],
                    help="a JUDGED.md of certified-only confirmations judged from outside (| date | container | class | mode | how | verdict |)")
    ap.add_argument("--neurotax", type=Path, help="the parser's own check per container (neurotax_check.jsonl)")
    ap.add_argument("--frozen-scan", type=Path,
                    help="tools/frozen_dim_scan.py's JSONL, one record per container (dims the trace left at its value)")
    ap.add_argument("--notes", type=Path, default=REPO / "docs/reference/model-status-notes.json",
                    help="measured causes no record field carries, one per container, each with its source")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = containers(a.cache)
    der = derivation(a.derivation)
    rec = json.loads(a.records.read_text())
    judged = {}
    for jf in a.judged:
        for line in jf.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6 or not re.match(r"\d{4}-\d{2}-\d{2}", cells[0]):
                continue
            for name in re.split(r"\s*/\s*", cells[1]):
                name = name.split(" (")[0].split(" — ")[0].strip()
                if name:
                    judged[name] = {"date": cells[0], "class": cells[2], "mode": cells[3], "how": cells[4],
                                    "verdict": cells[5], "file": jf.name}
    notes = {k: v for k, v in json.loads(a.notes.read_text()).items() if not k.startswith("_")} if a.notes.exists() else {}
    unknown = sorted(set(notes) - {c["name"] for c in rows})
    if unknown:
        raise SystemExit(f"model_status: notes name no container of the catalogue: {unknown} — refused")
    frozen = frozen_scan(a.frozen_scan, {c["name"] for c in rows}) if a.frozen_scan else None
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
        "| container · repo · format/NeuroTax · traced · build | census | certified 16 GB | certified 32 GB | vendor oracle "
        "| sequential oracle | Triton compiled (certified-only) | judged from outside | validated tonight (compiled Triton, "
        "certified-only, judged) | hub artefact |" + (" frozen dims (static scan) |" if frozen is not None else "") + " status |",
        "|---|---|---:|---:|---|---|---|---|---|---|" + ("---|" if frozen is not None else "") + "---|",
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
                                             c["forge"] or "build sha not recorded in the container"))
        jn = judged.get(n)
        jn_t = f"{jn['date']} {jn['class']}: {_short(jn['verdict'], 90)}" if jn else "not yet"
        fz = (f" {frozen_cell(frozen.get(n))} |" if frozen is not None else "")
        L.append(f"| `{n}` · {what} | {dv} | {cert[16]} | {cert[32]} | {vo_t} | {so_t} | {cm_t} | {jd_t} | {jn_t} | {hub} |{fz} {status} |")
        if not jn or not jn["verdict"].upper().startswith(("PASS", "MATCHES")):
            queue["validated tonight"].append(n)
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
    if a.neurotax and a.neurotax.exists():
        nt = {json.loads(l)["container"]: json.loads(l) for l in a.neurotax.read_text().splitlines() if l.strip()}
        L += ["", "## NeuroTax 5.0 weight keys — the parser's own check (normalize_strict(key) == key)", "",
              f"Read-only, `{a.neurotax}`. Canonical: the parser returns the key unchanged. Raw: it renames the key or "
              "refuses one of its tokens (a vendor token the synonym registry does not hold)."
              + (f" {json.loads(a.notes.read_text()).get('_neurotax_decision', '')}" if a.notes.exists() else ""), "",
              "| container | keys | raw | raw by component | first raw keys |", "|---|---:|---:|---|---|"]
        for c in rows:
            r_ = nt.get(c["name"])
            if r_ is None:
                L.append(f"| `{c['name']}` | no record | | | |")
                continue
            L.append(f"| `{c['name']}` | {r_['keys']} | {(str(r_['raw']) + ' — rename owed after validation') if r_['raw'] else 'fully canonical'} | "
                     f"{', '.join(f'{k} {v}' for k, v in r_['raw_by_component'].items()) or '—'} | "
                     f"{'; '.join(r_['samples'][:3]) or '—'} |")
    if frozen is not None:
        L += frozen_section(a.frozen_scan, rows, frozen)
    rt = rec.get("_retraces") or []
    if rt:
        L += ["", "## Retraces of the last month, and whether the oracle ladder preceded them", "",
              "| container | date | outcome | stated reason | ladder on record before the retrace |", "|---|---|---|---|---|"]
        for x in rt:
            L.append(f"| `{x.get('model')}` | {x.get('date')} | {_short(x.get('outcome'), 70)} | {_short(x.get('reason') or 'no reason on record', 80)} "
                     f"| {_short(x.get('ladder_on_record'), 90)} |")
    L += ["", "## The queue, in the owner's order", ""]
    for step in ("derivation", "certification", "oracle ladder", "Triton compiled confirmation", "validated tonight"):
        L.append(f"- **{step}** ({len(queue[step])}): " + (", ".join(f"`{x}`" for x in queue[step]) or "none"))
    a.out.write_text(_public("\n".join(L)) + "\n")
    print(f"model_status: {len(rows)} containers -> {a.out}")


if __name__ == "__main__":
    main()
