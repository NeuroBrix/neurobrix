"""The device facts of the card models of a machine that are not card 0's: each model's own block, with the
expandable pool's chunk measured on its own first card."""
from typing import Any, Dict


def measured_of_others(devices, measured_of, added, q, chunks=None) -> Dict[Any, dict]:
    """{model: its device facts} of every card model of the machine but card 0's: its own block (its vendor file's),
    with the expandable pool's chunk measured on its own first card (by its `index`, as the library reads it) where
    that block states none. A mixed machine's 32 GB cards were left without a chunk and every run placed there was
    refused (2026-11-17). `chunks`: {model: its sweep's answer} already swept side by side (`autodetect.
    _sweep_chunks`); a model not in it is swept here. An unconcluded sweep's spread is kept beside its chunk."""
    from neurobrix.core.prism import facts_probe as P
    others: Dict[Any, dict] = {}
    for d in (d for d in devices if d.get("model") != devices[0].get("model")):
        own = others.setdefault(d.get("model"), measured_of(d))
        if "vmm_chunk_bytes" not in own:
            if "index" not in d:
                raise ValueError(f"profile {d.get('model')!r}: a device of this card model has no 'index' fact, so its "
                                 f"expandable-pool chunk cannot be measured on its own card; add 'index' to every device "
                                 f"of the profile's devices list")
            got = chunks[d.get("model")] if chunks and d.get("model") in chunks else \
                P.vmm_chunk_bytes(int(d["index"]), q["vmm_reps"], q["vmm_ks"])
            own["vmm_chunk_bytes"] = added.setdefault("vmm_chunk_bytes_of", {})[int(d["index"])] = int(got or 0)
            if getattr(got, "spread", None) is not None:
                own["vmm_chunk_spread"] = dict(got.spread)
    return others


def place(devices, first, others) -> None:
    """Card 0's model's facts onto its cards (they share what card 0 measured), each other model's own onto its cards."""
    for d in devices:
        d["measured"] = dict(first if d.get("model") == devices[0].get("model") else others[d.get("model")])


#: the device-entry facts a card's runtime gives (`facts_probe.complete_runtime_card_facts`): the same on every card of
#: one model
RUNTIME_CARD_FACTS = ("runtime_total_mb", "runtime_context_mb")


def sibling_profiles(directory, exclude=None):
    """The GENERATED profiles of `directory` (`default-*.yml`: one machine's, each seen through its card set) but
    `exclude`, read; unreadable ones named in the result's second item."""
    import yaml
    from pathlib import Path
    out, skipped = [], []
    for p in sorted(Path(directory).glob("default-*.yml")):
        if exclude is not None and p.resolve() == Path(exclude).resolve():
            continue
        try:
            out.append((p, yaml.safe_load(p.read_text()) or {}))
        except (OSError, yaml.YAMLError) as exc:
            skipped.append(f"{p.name}: {exc}")
    return out, skipped


def from_siblings(path, data, devices) -> dict:
    """Class facts measured ONCE per card model and machine, reused by every card set (the launch-time cure,
    2026-11-19: a fresh home measured the same 16 GB class again for its 1-, 2- and 4-card profiles, 24 replay children
    each time). Onto each device of `devices` (the profile at `path`, `data` its whole content) that lacks them: the
    `measured` facts and the runtime's card facts of a device of the SAME model — earlier in this profile or in a
    sibling generated profile (`sibling_profiles`); onto the profile: the host's measured facts (`host_measured`, key by
    key, the reload rates per engine and stored dtype) and the engine's base (`cpu.runtime_base_mb`) a sibling holds.
    Identical cards are never measured again; a fact no sibling holds is left for the launch to measure. Returns
    {"class_facts_from": [what came from where]} or {}."""
    from pathlib import Path
    siblings, _skipped = sibling_profiles(Path(path).parent, exclude=path)
    took = []
    donors = [(path.name if hasattr(path, "name") else str(path), d) for d in devices] + \
             [(p.name, d) for p, s in siblings for d in (s.get("devices") or [])]
    for i, d in enumerate(devices):
        mine = dict(d.get("measured") or {})
        for name, donor in donors:
            if donor is d or donor.get("model") != d.get("model"):
                continue
            theirs = donor.get("measured") or {}
            missing = {k: v for k, v in theirs.items() if k not in mine and k != "source"}
            if missing:
                src = theirs.get("source")
                mine.update(missing)
                mine["source"] = (f"{mine['source']}; " if mine.get("source") else "") + \
                    f"{sorted(missing)} of {d.get('model')} from {name}" + (f" ({src})" if src else "")
                took.append(f"card {d.get('index', i)}: {sorted(missing)} from {name}")
            for k in RUNTIME_CARD_FACTS:
                if d.get(k) is None and donor.get(k) is not None:
                    d[k] = donor[k]
                    took.append(f"card {d.get('index', i)}: {k} from {name}")
        if mine != dict(d.get("measured") or {}):
            d["measured"] = mine
    for p, s in siblings:
        theirs = s.get("host_measured") or {}
        if theirs:
            host = data.setdefault("host_measured", {})
            for k, v in theirs.items():
                if k == "weight_reload_gbps" and isinstance(v, dict):
                    # a rate this profile lacks comes with its spread when the sibling kept it unsteady
                    for eng, by in v.items():
                        for st, val in (by or {}).items():
                            if st in ((host.get(k) or {}).get(eng) or {}):
                                continue
                            host.setdefault(k, {}).setdefault(eng, {})[st] = val
                            spread = ((theirs.get("weight_reload_spread") or {}).get(eng) or {}).get(st)
                            if spread is not None:
                                host.setdefault("weight_reload_spread", {}).setdefault(eng, {})[st] = spread
                            took.append(f"host: weight_reload_gbps.{eng}.{st} from {p.name}")
                elif k == "weight_reload_source":
                    have = host.setdefault(k, [])
                    have.extend(x for x in (v or []) if x not in have)
                elif k != "weight_reload_spread" and k not in host:
                    host[k] = v
                    took.append(f"host: {k} from {p.name}")
        base = (s.get("cpu") or {}).get("runtime_base_mb")
        cpu = data.get("cpu")
        if base and isinstance(cpu, dict) and not cpu.get("runtime_base_mb"):
            cpu["runtime_base_mb"] = dict(base)
            took.append(f"cpu: runtime_base_mb from {p.name}")
    return {"class_facts_from": took} if took else {}


def base_from_siblings(directory, exclude=None) -> dict:
    """The engine's base (`cpu.runtime_base_mb`) a generated profile of `directory` (but `exclude`) measured, or {}: a
    machine fact, the same whatever card set the profile sees."""
    for _p, s in sibling_profiles(directory, exclude)[0]:
        base = (s.get("cpu") or {}).get("runtime_base_mb")
        if base:
            return dict(base)
    return {}
