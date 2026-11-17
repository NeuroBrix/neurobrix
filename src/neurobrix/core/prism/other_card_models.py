"""The device facts of the card models of a machine that are not card 0's: each model's own block, with the
expandable pool's chunk measured on its own first card."""
from typing import Any, Dict


def measured_of_others(devices, measured_of, added, q) -> Dict[Any, dict]:
    """{model: its device facts} of every card model of the machine but card 0's: its own block (its vendor file's),
    with the expandable pool's chunk measured on its own first card (by its `index`, as the library reads it) where
    that block states none. A mixed machine's 32 GB cards were left without a chunk and every run placed there was
    refused (2026-11-17)."""
    from neurobrix.core.prism import facts_probe as P
    others: Dict[Any, dict] = {}
    for d in (d for d in devices if d.get("model") != devices[0].get("model")):
        own = others.setdefault(d.get("model"), measured_of(d))
        if "vmm_chunk_bytes" not in own:
            if "index" not in d:
                raise ValueError(f"profile {d.get('model')!r}: a device of this card model has no 'index' fact, so its "
                                 f"expandable-pool chunk cannot be measured on its own card; add 'index' to every device "
                                 f"of the profile's devices list")
            own["vmm_chunk_bytes"] = added.setdefault("vmm_chunk_bytes_of", {})[int(d["index"])] = int(
                P.vmm_chunk_bytes(int(d["index"]), q["vmm_reps"], q["vmm_ks"]) or 0)
    return others


def place(devices, first, others) -> None:
    """Card 0's model's facts onto its cards (they share what card 0 measured), each other model's own onto its cards."""
    for d in devices:
        d["measured"] = dict(first if d.get("model") == devices[0].get("model") else others[d.get("model")])
